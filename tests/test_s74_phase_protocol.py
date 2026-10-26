"""S74 only reprocesses original pixels; frozen CAL before fresh TEST."""
import inspect
import json
from pathlib import Path
import pytest
from scripts.research.bubble_chamber import s74_calibrate_then_test_unsharp as s74


def test_frozen_s74_seeds_and_complete_test():
    assert s74.CAL_SEEDS=={'005':20267401,'006':20267411}
    assert s74.TEST_SEEDS=={'005':(20267403,20267404),'006':(20267413,20267414)}
    assert s74.N_CAL==s74.N_TEST==24
    assert s74.CONDITIONS==('clean','blur_sigma_11')
    assert s74.PROTOCOL.exists()
    assert s74.PREDICTOR.exists()


def test_no_oracle_data_in_inference_or_transforms():
    s=inspect.getsource(s74.sample)
    assert s.index('s68.infer_pixel_only')<s.index('audit_orientation_invariant_synthetic')
    assert 'scene.nodes' in s[s.index('audit_orientation_invariant_synthetic'):]
    assert 'scene.nodes' not in s[:s.index('audit_orientation_invariant_synthetic')]


def test_previous_s72_cal_lock_exact():
    root=Path(__file__).resolve().parents[1]
    j=s74.freeze_originals(root/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json')
    assert j['phase']=='cal'
    assert j['directions']['005_to_006']['CAL_selected']=='s68_branch_endpoint_refine'
