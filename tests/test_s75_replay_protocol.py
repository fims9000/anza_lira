"""Data leakage and frozen input contracts for S75 CAL/TEST replay."""
import inspect
from pathlib import Path
import numpy as np
import pytest
from scripts.research.bubble_chamber.s75_calibrate_then_test_local_ridges import (
    CAL_SEEDS,TEST_SEEDS,N_CAL,N_TEST,CAL_CONDITIONS,TEST_CONDITIONS,
    condition_pixels,SCHEMA,sample,run)
from bubble_chamber.s75_local_ridge_rescue import preprocess


def test_no_reused_old_seeds_and_fixed_folds():
    old={20267203,20267204,20267213,20267214,20267303,20267304,20267313,
         20267314,20267403,20267404,20267413,20267414}
    assert set(sum((list(x) for x in TEST_SEEDS.values()),[])).isdisjoint(old)
    assert not set(sum((list(x) for x in TEST_SEEDS.values()),[])).intersection(CAL_SEEDS.values())
    assert (N_CAL,N_TEST)==(24,24)
    assert CAL_CONDITIONS==('clean','blur_sigma_11')
    assert TEST_CONDITIONS==('clean','blur_sigma_11','blur_sigma_16')
    assert SCHEMA.endswith('SYNTHETIC_ONLY_V1')


def test_both_blur_and_clean_pixel_stressors_fixed_and_image_only():
    gray=np.full((192,192),120,np.uint8)
    gray[90:93,30:160]=15
    assert np.array_equal(condition_pixels(gray,'clean'),gray)
    assert not np.array_equal(condition_pixels(gray,'blur_sigma_11'),gray)
    assert not np.array_equal(condition_pixels(gray,'blur_sigma_16'),gray)
    assert list(inspect.signature(preprocess).parameters)==['gray','arm']


def test_missing_frozen_s72_data_fails_before_CAL_or_TEST():
    with pytest.raises((FileNotFoundError,ValueError)):
        run(Path('/nonexistent/visualmetro'),Path('/nonexistent/tiles'),
            Path('/nonexistent/S72_CAL.json'),'test',frozen_s75_cal=None)


def test_source_code_contains_no_source_to_predicted_geometry_backdoor():
    source=Path(__file__).resolve().parents[1]/'bubble_chamber/s75_local_ridge_rescue.py'
    s=source.read_text()
    assert 'scene.nodes' not in s
    assert 'generator_nodes' not in s
    assert 'native_physics_auto_join_authorized' in s
