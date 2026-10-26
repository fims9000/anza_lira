"""Frozen S76 diagnostic uses new seeds, locked source code, never changes inference."""
import inspect
from pathlib import Path
from hashlib import sha256
import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from scripts.research.bubble_chamber import s76_blind_shape_endpoint_benchmark as s76


def test_frozen_source_cal_and_fresh_test_seeds():
    assert sha256(s76.SOURCE_CAL.read_bytes()).hexdigest()==s76.LOCKED_S72_CAL_SHA256
    assert s76.SEEDS=={'005':(20267603,20267604),'006':(20267613,20267614)}
    assert set(sum((list(v) for v in s76.SEEDS.values()),[])).isdisjoint(
        {20267503,20267504,20267513,20267514,20267403,20267404,20267413,20267414})
    assert s76.N_TEST==24
    assert s76.IMAGE_CONDITIONS==('clean','blur_sigma_11','blur_sigma_16')
    assert 'injected' not in inspect.signature(s76.blind_pixel_predictions).parameters


def test_official_audit_score_is_never_replaced_by_new_diagnostic():
    import bubble_chamber.s69_orientation_invariant_audit as a
    original=a.audit_orientation_invariant_synthetic
    from tests.test_s76_shape_endpoint_diagnostics import synthetic_ref,bonded
    ref=synthetic_ref()
    pred={'polylines':ref[:3],'chosen':[bonded(ref[0],ref[1],0,1)],
          'pair_count':1,'solver_success':True}
    row=s76.measure_prediction(pred,ref)
    assert row['official_s69_correct_links']==1
    assert row['selected_proposals']==1
    assert a.audit_orientation_invariant_synthetic is original


def test_synthetic_only_schema_and_no_private_trajectories():
    assert s76.SCHEMA.endswith('_V1')
    assert set(s76.FIELDS).isdisjoint({'injected','frame_id','photo_name','polylines'})
    assert not any(x in inspect.signature(s76.blind_pixel_predictions).parameters
                   for x in ('synthetic_proxy','reference','injected','source_track_id'))


def test_sum_bootstrap_pairing_and_wrong_port_incomplete_accounting():
    a={k:0 for k in s76.FIELDS};a['selected_proposals']=2
    b=dict(a);b['matched_neighbor_wrong_port']=1
    with pytest.raises(ValueError):s76.paired([],[],42)
    stats=s76.paired([a,a],[b,b],42)
    assert stats['matched_neighbor_wrong_port']['paired_difference_per_scene']==1
    assert stats['matched_neighbor_wrong_port']['scene_bootstrap_95pct']==[1.,1.]
