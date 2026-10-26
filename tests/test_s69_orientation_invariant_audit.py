"""Golden regression: flipping stored synthetic points cannot flip the GT."""
import inspect
import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene
from bubble_chamber.s69_orientation_invariant_audit import (
    audit_orientation_invariant_synthetic, reference_facing_ports,
)


def _synthetic_nodes():
    nodes=[]
    for y in (36,96):
        for left in (20,70,120):
            nodes.append(np.column_stack((np.linspace(left,left+27,31),np.full(31,y))))
    nodes.append(np.column_stack((np.linspace(20,47,31),np.full(31,157))))
    return nodes


def _get_edges(src):
    from bubble_chamber.s35_fragment_link import optimal_endpoints
    o=[]
    for i,j in ((0,1),(1,2)):
        pick=optimal_endpoints(src[i],src[j])
        o.append(Edge(i,j,int(pick['endpoint_index_a']),int(pick['endpoint_index_b']),.92))
    return o


def test_all_reference_reversals_keep_identical_synthetic_truth():
    ref=_synthetic_nodes()
    for mask in range(8):
        source=[p[::-1].copy() if mask&(1<<k) else p.copy() for k,p in enumerate(ref)]
        pred={'polylines':source[:3],'chosen':_get_edges(source),'pair_count':2,'solver_success':True}
        corrected=audit_orientation_invariant_synthetic(pred,source)
        assert corrected['selected_synthetic_correct_links']==2
        assert corrected['selected_synthetic_wrong_endpoint']==0
        assert corrected['complete_three_piece_paths_synthetic']==1
        assert corrected['synthetic_noncorrect_selected']==0
        assert corrected['physical_verified_links']==0
    # Historic S64 evaluator is provably NOT orientation invariant.
    rev=ref[:];rev=[p.copy() for p in ref];rev[1]=rev[1][::-1]
    old=audit_synthetic_scene(dict(polylines=rev[:3],chosen=_get_edges(rev),pair_count=2,solver_success=True),rev)
    assert old['selected_synthetic_correct_links']==0
    assert old['selected_synthetic_wrong_endpoint']==2


def test_wrong_ports_and_wrong_pairs_still_rejected():
    src=_synthetic_nodes()
    pred={'polylines':src[:3],'pair_count':1,'solver_success':True,
          'chosen':[Edge(0,1,0,-1,.97)]}
    a=audit_orientation_invariant_synthetic(pred,src)
    assert a['selected_synthetic_correct_links']==0
    assert a['selected_synthetic_wrong_endpoint']==1
    pred['chosen']=[Edge(0,2,-1,0,.97)]
    b=audit_orientation_invariant_synthetic(pred,src)
    assert b['selected_synthetic_wrong_identified']==1
    assert b['selected_synthetic_correct_links']==0


def test_gap_ports_nearest_endpoints_and_inference_separation():
    src=_synthetic_nodes()
    assert reference_facing_ports(src[0],src[1])==(-1,0)
    assert reference_facing_ports(src[0][::-1],src[1][::-1])==(0,-1)
    assert reference_facing_ports(src[0][::-1],src[1])==(0,0)
    with pytest.raises(ValueError):reference_facing_ports(src[0],src[1][:,:1])
    signature=set(inspect.signature(audit_orientation_invariant_synthetic).parameters)
    assert signature=={'inference','injected'}
