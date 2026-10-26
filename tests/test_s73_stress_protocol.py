"""S73 contracts prevent retuning, cross-domain leakage and false native GT."""
import inspect
import json
from pathlib import Path
import pytest
from scripts.research.bubble_chamber import s73_stress_test_pixel_graph as s73
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72


def test_new_scene_seeds_are_disjoint_and_immutable():
    assert s73.SEEDS=={'005':(20267303,20267304),'006':(20267313,20267314)}
    assert s73.N_SCENES==24 and s73.BOOTSTRAP==1200
    assert set(sum((list(v) for v in s73.SEEDS.values()),[])).isdisjoint(
      set(sum((list(v) for v in s72.TEST_SEEDS.values()),[])))
    assert s73.PROTOCOL.exists()


def test_lock_cal_is_prior_s72_and_rejects_drift(tmp_path):
    local=Path(__file__).resolve().parents[1]/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
    assert s73.digest(local)==s73.FROZEN_S72_CAL_SHA256
    x=s73.load_locked_cal(local)
    assert x['directions']['005_to_006']['CAL_selected']=='s68_branch_endpoint_refine'
    assert x['directions']['006_to_005']['CAL_selected']=='s39_original'
    changed=tmp_path/'broken.json';changed.write_bytes(local.read_bytes()+b' ')
    with pytest.raises(ValueError):s73.load_locked_cal(changed)


def test_robustness_engine_never_accepts_native_truth():
    a=list(inspect.signature(s73.run).parameters)
    assert a==['visualmetro','tiles','calibration','smoke']
    source=inspect.getsource(s73.run)
    assert source.index('s68.infer_pixel_only')<source.index('audit_orientation_invariant_synthetic')
    assert 'native_H50Z_verified_GT_count' in source
    assert 'native_physical_auto_joins_authorized' in source


def test_paired_scene_ci_is_zero_for_identical_predicted_scenes():
    rows=[{k:1 for k in s73.PRIMARY_METRICS} for _ in range(6)]
    v=s73._paired(rows,rows,20267399)
    assert all(x['delta_total']==0 and x['paired_scene_bootstrap_95pct']==[0.,0.]
               for x in v.values())
    with pytest.raises(ValueError):s73._paired(rows,rows[:-1],8)
