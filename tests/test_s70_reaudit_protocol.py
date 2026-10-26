"""Source-only S70 retrospective contract: freeze original predictions and data roles."""
from pathlib import Path
from hashlib import sha256
import inspect
import numpy as np
import pytest

from scripts.research.bubble_chamber.s70_reaudit_s64_historic import (
    LOCKS,SCHEMA,digest,paired_bootstrap,
)
from bubble_chamber.s64_pixel_to_path import infer_from_pixels

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'original_benchmark':ROOT/'scripts/research/bubble_chamber/s64_benchmark_pixel_to_path.py',
 'old_evaluator':ROOT/'bubble_chamber/s64_pixel_to_path.py',
 'new_evaluator':ROOT/'bubble_chamber/s69_orientation_invariant_audit.py',
 'original_aggregate':ROOT/'docs/bubble_chamber/S64_PUBLIC_PIXEL_TO_GRAPH_RESULTS_2026_10_26.json',
}

def test_exact_frozen_historic_source_hashes():
    assert all(digest(p)==LOCKS[k] for k,p in FILES.items())
    assert 'RETROSPECTIVE' in SCHEMA and 'S64' in SCHEMA
    assert len({digest(p) for p in FILES.values()})==4


def test_original_pixel_predictor_stays_truth_free():
    params=set(inspect.signature(infer_from_pixels).parameters)
    assert params=={'gray','scorer','cutoff','arm','margin'}
    assert not params.intersection({'source','injected','reference','labels','source_frames'})


def test_bootstrap_resamples_scenes_not_pixels():
    old={'selected_synthetic_correct_links':0,'synthetic_noncorrect_selected':1,
         'selected_synthetic_wrong_endpoint':1,'complete_three_piece_paths_synthetic':0,
         'scene_any_noncorrect':1}
    new={'selected_synthetic_correct_links':1,'synthetic_noncorrect_selected':0,
         'selected_synthetic_wrong_endpoint':0,'complete_three_piece_paths_synthetic':1,
         'scene_any_noncorrect':0}
    rows=[(old,new) for _ in range(24)]
    a=paired_bootstrap(rows,seed=10)
    b=paired_bootstrap(rows,seed=10)
    assert a==b
    assert a['selected_synthetic_correct_links']['difference_corrected_minus_historic_total']==24
    assert a['selected_synthetic_correct_links']['paired_scene_95pct']==[1.,1.]
    assert a['synthetic_noncorrect_selected']['paired_scene_95pct']==[-1.,-1.]
