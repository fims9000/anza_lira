"""S75B must verify original source/calibration and cannot tune frozen choices."""
from pathlib import Path
import pytest
from scripts.research.bubble_chamber.s75_posthoc_paired_uncertainty import run, SCHEMA


def test_posthoc_not_independent_and_refuses_unknown_photos():
    assert 'POSTHOC' in SCHEMA and 'NOT_NEW_TEST' in SCHEMA
    with pytest.raises(ValueError):run('bad',Path('missing'),Path('missing'),
                                      Path('missing'),Path('missing'),Path('missing'))


def test_source_declares_paired_historic_verification():
    p=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s75_posthoc_paired_uncertainty.py'
    txt=p.read_text()
    assert "if actual != original:" in txt
    assert "'no_test_set_reselection':True" in txt
    assert "'independent_new_test':False" in txt
