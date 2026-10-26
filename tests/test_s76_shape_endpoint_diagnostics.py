"""Regression: shape match must not be preconditioned on endpoint distances."""
import inspect
import numpy as np
import pytest
from bubble_chamber.s35_fragment_link import optimal_endpoints
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s76_shape_endpoint_diagnostics import (
    shape_only_match,endpoint_pair_distance,audit_shape_vs_endpoint)


def synthetic_ref():
    out=[]
    for y in (35,95):
        for left in (20,70,120):
            out.append(np.column_stack((np.linspace(left,left+27,45),np.full(45,y))))
    out.append(np.column_stack((np.linspace(20,47,45),np.full(45,157))))
    return out


def bonded(a,b,i,j):
    pick=optimal_endpoints(a,b)
    return Edge(i,j,int(pick['endpoint_index_a']),int(pick['endpoint_index_b']),.92)


def test_endpoint_far_but_shape_matched_not_hidden_as_unmatched():
    ref=synthetic_ref()
    # Thin segment would be insufficient source coverage for a 27 px reference.
    # Use 18px middle segment, shape supported well, ends > 8 px away.
    src=np.column_stack((np.linspace(20,70,101),np.full(101,185.)))
    partial=np.column_stack((np.linspace(31,59,55),np.full(55,185.)))
    pieces=[partial]
    ref[6]=src
    matches,qa,_=shape_only_match(pieces,ref)
    assert matches=={0:6} and qa['matched']==1
    assert max(endpoint_pair_distance(partial,src))>8
    measured=audit_shape_vs_endpoint({'polylines':pieces,'chosen':[]},ref)
    assert measured['shape_matched_one_or_both_endpoints_far']==1
    assert measured['shape_decoy_one_or_both_endpoints_far']==1
    assert measured['shape_true_one_or_both_endpoints_far']==0
    assert measured['shape_unmatched_polylines']==0


def test_reversed_point_storage_is_irrelevant():
    ref=synthetic_ref()
    a=ref[:3]
    edges=[bonded(a[0],a[1],0,1),bonded(a[1],a[2],1,2)]
    inference={'polylines':a,'chosen':edges,'pair_count':2,'solver_success':True}
    first=audit_shape_vs_endpoint(inference,ref)
    assert first['matched_neighbor_correct_port']==2
    assert first['shape_endpoint_separated_complete_chains']==1
    for mask in range(8):
        new=[p[::-1].copy() if mask&(1<<i) else p.copy() for i,p in enumerate(ref)]
        newedges=[bonded(new[0],new[1],0,1),bonded(new[1],new[2],1,2)]
        report=audit_shape_vs_endpoint({'polylines':new[:3],'chosen':newedges},new)
        assert report['matched_neighbor_correct_port']==2
        assert report['shape_endpoint_separated_complete_chains']==1
        assert report['physical_track_connections_authorized'] is False


def test_wrong_selected_port_separated_from_bad_endpoint_localization():
    ref=synthetic_ref();pieces=ref[:3]
    correct=bonded(pieces[0],pieces[1],0,1)
    flipped=Edge(0,1,0 if correct.end_i==-1 else -1,correct.end_j,.95)
    qa=audit_shape_vs_endpoint({'polylines':pieces,'chosen':[flipped]},ref)
    assert qa['matched_neighbor_wrong_port']==1 and qa['matched_neighbor_endpoint_unlocalized']==0
    cut=pieces[0].copy()
    cut[:,0]-=9   # shape overlap remains; selected gap endpoint now 9px short
    assert shape_only_match([cut],ref)[0].get(0)==0  # still shape overlap on the source
    qa=audit_shape_vs_endpoint({'polylines':[cut,pieces[1]],'chosen':[Edge(0,1,-1,0,.8)]},ref)
    assert qa['matched_neighbor_endpoint_unlocalized']==1


def test_non_neighbor_pair_and_unmatched_are_not_called_endpoint_failures():
    ref=synthetic_ref()
    qa=audit_shape_vs_endpoint({'polylines':[ref[0],ref[2]],'chosen':[Edge(0,1,-1,0,.8)]},ref)
    assert qa['matched_wrong_identity_pair']==1
    extra=np.column_stack((np.linspace(12,42,31),np.full(31,180.)))
    qa=audit_shape_vs_endpoint({'polylines':[ref[0],extra],'chosen':[Edge(0,1,-1,0,.8)]},ref)
    assert qa['unmatched_or_ambiguous_endpoint_blind']==1


def test_diagnostic_only_no_inference_api_and_original_s69_unmodified():
    ref=synthetic_ref()
    inf={'polylines':ref[:3],'chosen':[bonded(ref[0],ref[1],0,1)],'pair_count':1,'solver_success':True}
    before=audit_orientation_invariant_synthetic(inf,ref)
    _=audit_shape_vs_endpoint(inf,ref)
    after=audit_orientation_invariant_synthetic(inf,ref)
    assert before==after
    assert set(inspect.signature(audit_shape_vs_endpoint).parameters)=={'inference','injected'}
    with pytest.raises(ValueError):shape_only_match([np.ones((5,2))],ref)


def test_ambiguous_shape_and_duplicate_identity_abstention():
    ref=synthetic_ref()
    # Two identical artificial GT segments cannot establish an identity.
    ref[1]=ref[0].copy()
    m,q,_=shape_only_match([ref[0]],ref)
    assert m=={} and q['ambiguous']==1
    # Two image hypotheses cannot both claim the same single synthetic source.
    ref=synthetic_ref()
    m,q,_=shape_only_match([ref[0],ref[0].copy()],ref)
    assert len(m)==1 and q['unmatched']==1


def test_real_shape_near_but_endpoint_far_is_not_a_new_true_link():
    ref=synthetic_ref()
    shifted=ref[0].copy()
    shifted[:,0]-=9
    near=ref[1]
    m,q,_=shape_only_match([shifted,near],ref)
    assert m=={0:0,1:1}
    edge=Edge(0,1,-1,0,.99)
    out=audit_shape_vs_endpoint({'polylines':[shifted,near],'chosen':[edge]},ref)
    assert out['matched_neighbor_endpoint_unlocalized']==1
    assert out['matched_neighbor_correct_port']==0
    assert out['shape_true_one_or_both_endpoints_far']==1
