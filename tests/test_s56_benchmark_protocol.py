"""S56 frozen protocol and machine-only reporting guardrails."""
import json
from pathlib import Path
import numpy as np
import pytest

from bubble_chamber.s56_global_path_assembly import Edge
from scripts.research.bubble_chamber.s56_benchmark_global_paths import (
    SCHEMA,PROTOCOL,ARMS, evaluate,totals,paired_bootstrap,make_selections)


def test_protocol_lock_seed_and_scene_isolation():
    assert PROTOCOL['test_seeds']=={'005':(20265603,20265604), '006':(20265613,20265614)}
    assert PROTOCOL['cal_negative_fpr']==.05
    assert PROTOCOL['port_or_node_margin']==.04
    assert PROTOCOL['global_objective_stability_margin']==.04
    assert len(set(PROTOCOL['train_seeds'].values())|set(PROTOCOL['cal_seeds'].values())|
               set(PROTOCOL['test_seeds']['005'])|set(PROTOCOL['test_seeds']['006']))==8
    assert SCHEMA.endswith('public_v1')
    assert ARMS==('node_mutual','port_greedy','global_forest')


def test_scoring_abstains_on_infinite_calibrated_cutoff():
    e=[Edge(0,1,-1,0,.99)]
    got,qa=make_selections(e,float('inf'))
    assert all(not got[a] for a in ARMS)
    assert qa['native_physics_authorized_joins']==0


def test_scene_metric_counts_are_withheld_in_public_totals():
    records=[{'true':3,'wrong':0,'any_wrong':0,'suggestions':3,
              'full_trajectories':1,'exact_complete_scene':0,'components':4,'solver_success':1},
             {'true':2,'wrong':1,'any_wrong':1,'suggestions':3,
              'full_trajectories':0,'exact_complete_scene':0,'components':4,'solver_success':1}]
    data=totals(records)
    assert data['true']==5 and data['wrong']==1
    assert data['synthetic_true_links_total']==8
    assert data['synthetic_full_paths_total']==4
    assert data['physically_verified_native_joins']==0
    assert 'per_scene_counts_private' not in data
    assert 'frame_id' not in json.dumps(data)
    uncertainty=paired_bootstrap(records,records,seed=20265607,n_boot=40)
    assert uncertainty['wrong']['mean_delta_per_simulated_scene']==0
    assert uncertainty['true']['conditional_scene_bootstrap_95pct']==[0.,0.]


def test_full_scene_metric_rejects_incomplete_score_vector():
    row={'i':0,'j':1,'end_i':-1,'end_j':0,'label_SYNTHETIC_ONLY':1}
    with pytest.raises(ValueError):evaluate([{'rows':[row]}],[.5,.6],.4)


def test_original_ct28_models_and_native_physics_labels_outside_s56_source():
    # Guard against accidentally importing CT28 trained relation head or native
    # expert ground truth into this synthetic-only benchmark.
    root=Path(__file__).resolve().parents[1]
    code=(root/'scripts/research/bubble_chamber/s56_benchmark_global_paths.py').read_text()
    assert 'from models.azconv' not in code
    assert 'native_physical_auto_joins' in code
    assert 'truth_SYNTHETIC_ONLY' in (root/'bubble_chamber/s56_three_piece_scenes.py').read_text()
