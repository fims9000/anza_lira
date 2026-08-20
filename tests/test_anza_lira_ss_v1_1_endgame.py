from __future__ import annotations

import numpy as np
import torch

from structural_stability_v1_1.endgame_evaluation import THRESHOLDS, _select_threshold
from cracks_experiment.partial_labels import average_annotator_loss
from structural_stability_v1_1.endgame_training import _bernoulli_js, _partial_supervision, _sample_tensor, _spd_log_2x2, _topology_consistency
from structural_stability_v1_1.matrix_log import spd_matrix_log
from structural_stability_v1.agreement import crowd_agreement
from structural_stability_v1_1.geometry_targets import geometry_target


def test_frozen_threshold_grid() -> None:
    assert len(THRESHOLDS) == 37
    assert THRESHOLDS[0] == 0.05 and THRESHOLDS[-1] == 0.95


def test_threshold_selection_precision_and_tie_break() -> None:
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"threshold": threshold, "cldice": 0.8 if threshold in {0.5, 0.525} else 0.1, "precision": 0.8, "recall": 0.7, "dice": 0.5})
    assert _select_threshold(rows)["selected_threshold"] == 0.525


def test_identity_alignment_and_consistency_losses() -> None:
    value = torch.rand(2, 1, 16, 16, requires_grad=True)
    displacement = torch.zeros(2, 2, 16, 16)
    sampled = _sample_tensor(value, displacement)
    assert torch.allclose(sampled, value, atol=1e-6)
    weight = torch.ones_like(value)
    assert float(_bernoulli_js(value.sigmoid(), value.sigmoid(), weight).detach()) < 1e-7
    topology = _topology_consistency(value.sigmoid(), value.sigmoid(), weight)
    assert torch.isfinite(topology) and 0 <= float(topology.detach()) <= 1


def test_axial_sampling_translation_direction() -> None:
    value = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    displacement = torch.zeros(1, 2, 5, 5)
    displacement[:, 1] = 1
    sampled = _sample_tensor(value, displacement)
    assert torch.allclose(sampled[0, 0, :, :-1], value[0, 0, :, 1:], atol=1e-5)


def test_analytic_spd_log_matches_frozen_eigh_reference() -> None:
    matrix = torch.tensor([[[[[2.0]], [[0.3]]], [[[0.3]], [[0.8]]]]], dtype=torch.float64)
    assert torch.allclose(_spd_log_2x2(matrix), spd_matrix_log(matrix), atol=1e-7, rtol=1e-7)


def test_vectorized_partial_supervision_matches_reference_loop() -> None:
    torch.manual_seed(4)
    base = torch.randn(1, 1, 24, 24, requires_grad=True)
    targets = (torch.rand(4, 1, 24, 24) > 0.75).float()
    weights = torch.where(torch.rand_like(targets) > 0.2, torch.ones_like(targets), torch.zeros_like(targets))
    reference, _ = average_annotator_loss(base, targets, weights, topology_weight=0.2, topology_num_iters=5)
    vectorized = _partial_supervision(base.repeat_interleave(4, 0), targets, weights)
    assert torch.allclose(vectorized, reference, atol=1e-6, rtol=1e-6)


def test_fast_agreement_path_matches_full_geometry_target() -> None:
    masks = []
    colors = np.asarray(((31, 119, 180), (44, 160, 44), (255, 127, 14), (255, 255, 255)), dtype=np.uint8)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        masks.append(colors[rng.integers(0, 4, size=(32, 40))])
    assert np.array_equal(crowd_agreement(masks)["agreement"], geometry_target(masks)["agreement"])
