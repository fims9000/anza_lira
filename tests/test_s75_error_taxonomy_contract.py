"""Protect S75C from conflating native physics with synthetic error categories."""
from pathlib import Path
import pytest
from scripts.research.bubble_chamber.s75_posthoc_error_taxonomy import (
    SCHEMA,ERROR_KEYS,DETAIL_KEYS,run)


def test_s75c_label_and_completeness_gate():
    assert 'POSTHOC' in SCHEMA and 'NOT_INDEPENDENT' in SCHEMA
    assert len(ERROR_KEYS)==3
    assert all(k in DETAIL_KEYS for k in ERROR_KEYS)


def test_missing_data_rejected_before_replay():
    with pytest.raises((FileNotFoundError,ValueError)):
        run(Path('/not-present/source'),Path('/not-present/tiles'),
            Path('/not-present/s72'),Path('/not-present/s75cal'),Path('/not-present/s75test'))


def test_no_test_tuning_is_authorized():
    p=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s75_posthoc_error_taxonomy.py'
    t=p.read_text()
    assert "'new_training_or_threshold_selection':False" in t
    assert 'if actual!=original:' in t
    assert "'native_particle_auto_joins_authorized':False" in t
