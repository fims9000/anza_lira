"""K10/15 contract tests: synthetic model stubs, no fabricated CT results."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
import run_ct28_graph_lira as graph
from ccta_3d_endpoints import Candidate, Endpoint, Scene
from ccta_3d_k10 import K10Inference, verified_model_pack
from build_ct28_relation_features import HISTORICAL_FEATURES_PER_TYPE, CROSS_FEATURES


class BinaryStub:
    def __init__(self, features):
        self.feature_names_in_ = np.array(features)
    def predict_proba(self, values):
        return np.tile([.03, .97], (len(values), 1))


class RelationStub:
    classes_ = np.array([0, 1, 2, 3])
    def predict_proba(self, values):
        return np.tile([.01, .97, .01, .01], (len(values), 1))


def expected_features(streams):
    output = []
    for stream in streams:
        for kind in ("PAIR", "JUNCTION"):
            output += [f"{stream}_{kind.lower()}_{name}"
                       for name in HISTORICAL_FEATURES_PER_TYPE]
        output += [f"{stream}_{name}" for name in CROSS_FEATURES]
    return output


def stub_pack():
    return {
        "local_models": {
            "PAIR": {
                "geometry": BinaryStub(["pair_g0"]),
                "ct": BinaryStub(["pair_ct0"]),
                "geometry_plus_ct": BinaryStub(["pair_g0", "pair_ct0"]),
            },
            "JUNCTION": {
                "geometry": BinaryStub(["junction_span_mm"]),
                "ct": BinaryStub(["junction_ct0"]),
                "geometry_plus_ct": BinaryStub(["junction_span_mm", "junction_ct0"]),
            },
        },
        "relation_heads": {
            "geometry_only": RelationStub(),
            "ct_conditioned": RelationStub(),
        },
        "geometry_relation_features": expected_features(("geometry_score",)),
        "ct_conditioned_relation_features": expected_features(
            ("geometry_score", "ct_score")
        ),
    }


def fixture_scene():
    endpoints = [
        Endpoint(f"E{i:04d}", i+1, (10+i, 15, 15),
                 (-float(10+i), -15., 15.), (1., 0., 0.))
        for i in range(5)
    ]
    scene = Scene("REAL3D:training:W0000", tuple(x.endpoint_id for x in endpoints),
                  (Candidate("PAIR:E0000+E0001", "PAIR", ("E0000", "E0001")),
                   Candidate("JUNCTION:E0002+E0003+E0004", "JUNCTION",
                             ("E0002", "E0003", "E0004"))))
    return scene, endpoints


def test_k10_frozen_relation_and_15_repeat_acceptance(monkeypatch):
    scene, endpoints = fixture_scene()
    def dummy_features(row, lookup, ct, inverse, arm_names):
        return {
            "scene_id": row.scene_id,
            "candidate_id": row.candidate_id,
            "candidate_type": row.candidate_type,
            "pair_g0": 2.0,
            "pair_ct0": 200.0,
            "junction_span_mm": 3.0,
            "junction_ct0": 200.0,
        }
    monkeypatch.setattr(graph, "perturbed_candidate_features", dummy_features)
    engine = K10Inference(stub_pack())
    prediction = engine.predict(scene, endpoints, np.ones((30,30,30)),
                                np.eye(4))
    assert prediction.raw_relation == "PAIR"
    assert prediction.proposed_relation == "PAIR"
    assert prediction.selected_relation == "PAIR"
    assert prediction.pair_candidate_id == "PAIR:E0000+E0001"
    assert prediction.junction_candidate_id is None
    assert prediction.confidence == pytest.approx(.97)
    assert prediction.consistency == 1.0
    assert prediction.accepted
    assert len(prediction.decision_keys_under_perturbation) == 15
    assert all(key == "PAIR|PAIR:E0000+E0001"
               for key in prediction.decision_keys_under_perturbation)


def test_model_pack_sha_is_strict_before_pickle_deserialization(tmp_path):
    path = tmp_path / "relation_models.joblib"
    path.write_bytes(b"not a trusted K10/15 pack")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verified_model_pack(path)


def test_invalid_or_incomplete_model_pack_rejected():
    with pytest.raises(ValueError, match="missing required"):
        K10Inference({})
    wrong = stub_pack()
    wrong["ct_conditioned_relation_features"] = ["wrong"]
    with pytest.raises(ValueError, match="84"):
        K10Inference(wrong)
