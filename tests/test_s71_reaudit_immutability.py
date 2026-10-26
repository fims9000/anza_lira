"""S71: old vs new synthetic evaluation changes truth accounting only."""
import numpy as np
import pytest

from bubble_chamber.s56_global_path_assembly import Edge
from scripts.research.bubble_chamber import s71_reaudit_s65_s67_historic as m
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic


def reference():
    curves=[]
    for y in (30,95):
        for x in (20,70,120):
            curves.append(np.column_stack((np.linspace(x,x+27,31),np.full(31,y))))
    curves.append(np.column_stack((np.linspace(20,47,31),np.full(31,160))))
    return curves


def test_reversal_fixes_only_wrong_orientation_not_prediction():
    raw=reference()
    ref=[p.copy() for p in raw]
    ref[1]=ref[1][::-1].copy()  # generator stores middle synthetic curve in reverse
    chosen=[raw[0],raw[1],raw[2]]
    p={'polylines':chosen,'pair_count':2,'chosen':[Edge(0,1,-1,0,.95),Edge(1,2,-1,0,.95)],
       'solver_success':True}
    o=audit_synthetic_scene(p,ref)
    n=audit_orientation_invariant_synthetic(p,ref)
    m.assert_stable(o,n)
    assert n['selected_synthetic_correct_links']==2
    assert n['complete_three_piece_paths_synthetic']==1
    assert o['selected_synthetic_correct_links']==0
    assert len(p['chosen'])==2


def test_change_to_proposals_or_old_correct_is_rejected():
    ref=reference()
    p={'polylines':[ref[0],ref[1]],'pair_count':1,'chosen':[], 'solver_success':True}
    old=audit_synthetic_scene(p,ref)
    new=dict(old)
    new['candidate_polylines']=3
    with pytest.raises(AssertionError,match='Inference drift'):
        m.assert_stable(old,new)
    new=dict(old);new['selected_synthetic_correct_links']=-1
    with pytest.raises(AssertionError,match='Corrected evaluator lost'):
        m.assert_stable(old,new)


def test_dual_evaluator_restores_original_on_error():
    from scripts.research.bubble_chamber import s66_test_image_only_endpoint_extensions as mod
    old=mod.audit_synthetic_scene
    with pytest.raises(RuntimeError):
        with m.dual_evaluator(mod):
            assert mod.audit_synthetic_scene is not old
            raise RuntimeError('failure during benchmark must not persist patch')
    assert mod.audit_synthetic_scene is old


def test_no_historical_metric_overwrite_and_snapshot_locks():
    m.verify_locks()
    assert set(m.ALLOWED_DIFFERENT)=={'selected_synthetic_correct_links',
        'selected_synthetic_wrong_endpoint','synthetic_noncorrect_selected',
        'scene_any_noncorrect','complete_three_piece_paths_synthetic'}
    assert m.EXPERIMENT_ID=='ANZA-LIRA-BC-S71-20261026'
