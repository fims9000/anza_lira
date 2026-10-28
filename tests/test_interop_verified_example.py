"""Runnable developer onboarding demo has no private coordinate leakage."""
import json
import subprocess
import sys
from pathlib import Path


def test_verified_external_demo_runs_from_repository_root():
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,'scripts/research/bubble_chamber/example_verified_external_detector.py'],
                      cwd=root,capture_output=True,text=True,timeout=15,check=True)
    j=json.loads(p.stdout)
    assert j['source_input_generated_not_H50Z'] is True
    assert j['this_mock_scorer_is_not_trained']
    assert j['private_review_positions_exported'] is False
    assert j['known_real_physics_tracks']==0
    assert not j['physical_auto_joins_authorized']
    assert 'lines_PRIVATE' not in p.stdout and 'source_image_id_PRIVATE' not in p.stdout
