"""Full-volume runtime integration on synthetic broken tubular masks."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_k10 import ScenePrediction
from ccta_3d_runtime import RuntimeConfig, run_volume, frozen_path_protocol
from ct28_path_stage import PathConfig, PathResult, lps_to_voxel


def synthetic_case():
    mask = np.zeros((50, 65, 35), dtype=bool)
    for y in (10, 16, 28, 40, 52):
        mask[10:19, y, 18] = True
    ct = np.full(mask.shape, 250, dtype=np.float32)
    return ct, mask, np.eye(4)


class AcceptPair:
    pack_sha = "stubbed_model_for_synthetic_test_only"
    def predict(self, scene, endpoints, ct, affine):
        pairs = [candidate for candidate in scene.candidates if candidate.kind == "PAIR"]
        pair = pairs[0]
        return ScenePrediction(
            scene_id=scene.scene_id,
            raw_relation="PAIR", proposed_relation="PAIR",
            selected_relation="PAIR", confidence=.99, consistency=1,
            accepted=True, pair_candidate_id=pair.candidate_id,
            junction_candidate_id=None, identity_log_utility=-0.1,
            candidate_count_pair=len(pairs),
            candidate_count_junction=0, relation_distribution=(.01, .99, 0, 0),
            decision_keys_under_perturbation=("PAIR|" + pair.candidate_id,) * 15,
        )


def direct_line_solver(ct, affine, positions, tangents, config, slack, forbidden):
    voxels = np.rint(lps_to_voxel(positions, np.linalg.inv(affine))).astype(int)
    if len(voxels) != 2:
        return PathResult("unsupported_junction_stub", 0, 0, (), None, 0, 0, 0)
    first, last = voxels
    steps = max(abs(last - first))
    route = [tuple(int(x) for x in np.rint(first + (last-first) * (i/steps)))
             for i in range(steps+1)]
    return PathResult("ok", 1, float(np.linalg.norm(last-first)),
                      (tuple(route),), None, 250., 0., 250.)


def test_full_volume_predict_score_route_mask_and_global_conflict():
    ct, mask, affine = synthetic_case()
    before = mask.copy()
    output, report = run_volume(
        ct, mask, affine, patient_label="train_synthetic",
        engine=AcceptPair(),
        params=RuntimeConfig(radius_mm=0, max_new_voxels=50, min_component_voxels=5),
        path_config=PathConfig(), solver=direct_line_solver,
    )
    assert report["status"] == "experimental_inference"
    assert report["scene_count"] >= 1
    assert report["accepted_decisions"] >= 1
    assert report["added_voxels"] > 0
    assert output.sum() == before.sum() + report["added_voxels"]
    assert (output | before == output).all()
    assert np.array_equal(mask, before), "input mask must never be modified"
    assert report["output_components_26"] < report["extraction"]["components"]
    assert report["model_pack_sha256"] == AcceptPair.pack_sha
    assert any(d["reason"] == "globally_reused_endpoint"
               for d in report["decisions"])


def test_candidate_inspection_only_never_modifies_mask():
    ct, mask, affine = synthetic_case()
    out, report = run_volume(
        ct, mask, affine, patient_label="external_case_1",
        engine=None,
        params=RuntimeConfig(radius_mm=0, max_new_voxels=20, min_component_voxels=5),
        path_config=PathConfig(),
    )
    assert np.array_equal(out, mask)
    assert report["status"] == "candidate_inspection_only"
    assert report["accepted_decisions"] == 0
    assert report["scene_count"] > 0


def test_rejects_heldout_and_input_mismatch_without_opening_test():
    ct, mask, affine = synthetic_case()
    with pytest.raises(ValueError, match="Held-out"):
        run_volume(ct, mask, affine, patient_label="954", engine=None,
                   params=RuntimeConfig(radius_mm=0,max_new_voxels=20),
                   path_config=PathConfig())
    with pytest.raises(ValueError, match="same 3D voxel grid"):
        run_volume(ct[:-1], mask, affine, patient_label="external",
                   engine=None, params=RuntimeConfig(radius_mm=0,max_new_voxels=20),
                   path_config=PathConfig())


def test_frozen_path_v2_cannot_be_modified(tmp_path):
    proto = Path(__file__).resolve().parents[1] / (
        "artifacts/varvara/ct28_graph_lira/path/selected_path_protocol.json"
    )
    settings, slack = frozen_path_protocol(proto)
    assert settings.hu_difference_scale == 200.0
    assert settings.corridor_radius_mm == 2.0
    assert slack == .05
    import json
    data = json.loads(proto.read_text())
    data["config"]["corridor_radius_mm"] = 2.5
    wrong = tmp_path / "tampered.json"
    wrong.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="parameters changed"):
        frozen_path_protocol(wrong)
