from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anza_tracegraph.batch import RelationDataset
from anza_tracegraph.candidates import generate_candidates
from anza_tracegraph.corridor import extract_corridors, hyperbolic_distance
from anza_tracegraph.data import generate_scene
from anza_tracegraph.frozen_source import BASE_CHECKPOINT, DENSE_CHECKPOINT, FEATURE_NORM, predicted_relation_scene
from anza_tracegraph.protocol import PROTOCOL
from anza_tracegraph.tracelets import Endpoint, Tracelet, endpoints, extract_tracelets


def test_tracelet_ordering_endpoints_and_tangent_are_deterministic():
    mask = np.zeros((32, 32), bool); mask[16, 4:28] = True; probability = mask.astype(float); image = probability.copy()
    first = extract_tracelets(mask, probability, image); second = extract_tracelets(mask, probability, image)
    assert len(first) == 1 and np.array_equal(first[0].points_yx, second[0].points_yx)
    a, b = endpoints(first[0], 5); assert a.point_yx != b.point_yx
    assert abs(np.dot(a.outgoing_tangent_yx, b.outgoing_tangent_yx)) == pytest.approx(1.0)


def test_shared_candidates_have_no_self_link_and_recall_truth():
    for index in range(128):
        scene = generate_scene("development", index)
        assert all(candidate.endpoint.tracelet_id != scene["source_endpoint"].tracelet_id for candidate in scene["candidates"])
        if scene["has_valid_continuation"]: assert scene["candidate_recalled"] and scene["target_index"] >= 0


def test_candidate_order_is_deterministic():
    source = Endpoint(0, -1, (10.0, 10.0), (0.0, 1.0), 1.0)
    destinations = [Endpoint(i, 0, (10.0 + i, 30.0), (0.0, -1.0), 1.0) for i in range(1, 5)]
    assert generate_candidates(source, destinations) == generate_candidates(source, destinations)


def test_corridor_shape_markers_and_orientation_are_shared():
    batch = next(iter(torch.utils.data.DataLoader(RelationDataset("train", [0, 1]), batch_size=2)))
    corridors, grids = extract_corridors(batch["dense"], batch["source_point"], batch["destination_points"])
    assert corridors.shape == (16, 10, 32, 64) and grids.shape == (16, 32, 64, 2)
    assert corridors[:, -2].argmax(dim=-1).float().mean() < corridors.shape[-1] / 2
    assert corridors[:, -1].argmax(dim=-1).float().mean() > corridors.shape[-1] / 2


def test_hyperbolic_q_is_axial_isotropic_and_transverse_suppressing():
    tangent = torch.tensor([[0.0, 1.0]]); along = torch.tensor([[0.0, 1.0]]); transverse = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(hyperbolic_distance(along, tangent, 0.35), hyperbolic_distance(along, -tangent, 0.35))
    assert hyperbolic_distance(along, tangent, 0.35) < hyperbolic_distance(transverse, tangent, 0.35)
    assert torch.allclose(hyperbolic_distance(along, tangent, 0.0), hyperbolic_distance(transverse, tangent, 0.0))


def test_split_seeds_are_disjoint_and_confirm_inaccessible():
    train = generate_scene("train", 7); dev = generate_scene("development", 7)
    assert not np.array_equal(train["dense"], dev["dense"])
    with pytest.raises(PermissionError): generate_scene("confirm", 0)


def test_frozen_dense_source_provenance_is_explicit():
    import hashlib

    assert BASE_CHECKPOINT.is_file()
    assert DENSE_CHECKPOINT.is_file()
    assert FEATURE_NORM.is_file()
    assert hashlib.sha256(DENSE_CHECKPOINT.read_bytes()).hexdigest() == PROTOCOL["dense_source"]["checkpoint_sha256"]


def test_predicted_tracelet_adapter_uses_prediction_not_latent_gap():
    raw = generate_scene("development", 0)
    probability = raw["dense"][3].copy()
    bank = np.full((8, *probability.shape), 1.0 / 8.0, dtype=np.float32)
    adapted = predicted_relation_scene(raw, probability, bank)
    assert adapted["source_available"]
    assert adapted["dense"].shape == (8, 96, 96)
    assert all(not (35 <= point[1] < 50) for tracelet in adapted["tracelets"] for point in tracelet.points_yx)
