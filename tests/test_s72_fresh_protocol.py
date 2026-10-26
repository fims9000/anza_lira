"""S72: orientation-correct from calibration before any independent test."""
import json
from pathlib import Path
import pytest
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber.s72_fresh_corrected_branch_benchmark import (
    LOCKS,CAL_SEEDS,TEST_SEEDS,CAL_SCENES,TEST_SCENES,
    corrected_audit_only,digest,PROTOCOL,ROOT,SCHEMA)
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic


def test_frozen_source_locks_and_new_test_ids():
    assert PROTOCOL.is_file() and 'S72' in SCHEMA
    for p,sha in LOCKS.items():assert digest(ROOT/p)==sha
    assert (CAL_SCENES,TEST_SCENES)==(32,24)
    prior={20266503,20266504,20266513,20266514,
           20266603,20266604,20266613,20266614,
           20266703,20266704,20266713,20266714,
           20266803,20266804,20266813,20266814,
           20266903,20266904,20266913,20266914}
    new={i for group in TEST_SEEDS.values() for i in group}
    assert not new & prior
    assert not new & set(CAL_SEEDS.values())


def test_corrected_only_during_audit_and_original_restored():
    historical=s68.audit_synthetic_scene
    with corrected_audit_only():
        assert s68.audit_synthetic_scene is audit_orientation_invariant_synthetic
    assert s68.audit_synthetic_scene is historical
    with pytest.raises(ValueError):
        with corrected_audit_only():
            raise ValueError('source failure')
    assert s68.audit_synthetic_scene is historical


def test_no_physical_auto_join_authority_in_run_source():
    code=(ROOT/'scripts/research/bubble_chamber/s72_fresh_corrected_branch_benchmark.py').read_text()
    assert 'physical_auto_join_authorized' in code
    assert "'physical_verified_H50Z_GT':0" in code
    assert 's68._photo_subfolder(tiles,target)' in code
    assert 'corrected_audit_only()' in code
