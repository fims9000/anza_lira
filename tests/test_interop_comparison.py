"""Only the generated-data benchmark contract; not native H50Z accuracy."""
from __future__ import annotations
import json
import numpy as np
import pytest
from scripts.research.bubble_chamber import interop_comparison as q


def test_generator_is_deterministic_and_never_claims_real_images():
    templates=q.make_fake_templates()
    assert len(templates)==28
    assert all(t['frame']=='GENERATED_NOT_FILM070' for t in templates)
    assert q.SCENES==20 and q.SEED==20268588
    assert q.CONDITIONS==('clean','blur_sigma_11','scratch_grain')
    assert q.SOLVERS==('none','node_mutual','port_greedy','global_milp')


def test_toy_qa_score_is_bounded_not_physics_probability():
    x=[{'endpoint_gap_px':13.,'outgoing_tangent_A_to_gap':1.,
        'outgoing_tangent_B_to_gap':.9,'outgoing_tangent_mutual':.9,
        'chord_curvature_consistency':.85}]
    values=q.score_geometry_for_QA(x)
    assert values.shape==(1,) and 0 <= values[0]<=1
    assert q.score_geometry_for_QA([]).shape==(0,)


def test_full_generated_scene_uses_after_inference_gt_only():
    # The toy (2 scenes) is a code-path test, never the registered 20-scene study.
    result=q.run(2)
    assert result['mode']=='GENERATED_TOY_ONLY_NOT_H50Z'
    assert result['native_GT_count']==0 and not result['physical_auto_joins_authorized']
    assert not result['source_train_calibration_per_detector']
    assert result['image_backgrounds_not_H50Z'] and result['source_not_film070']
    assert len(result['component_source_sha256'])>=5
    for cond in q.CONDITIONS:
        for name in result['methods']:
            for solver in q.SOLVERS:
                stats=result['results'][cond][name][solver]
                assert stats['scenes']==2
                assert stats['selected_synthetic_correct_links']+stats['synthetic_noncorrect_selected'] <= 12
                assert 0 <= stats['complete_three_piece_paths_synthetic'] <= 4
                if solver=='none':
                    assert stats['selected_synthetic_correct_links']==0
                    assert stats['synthetic_noncorrect_selected']==0
