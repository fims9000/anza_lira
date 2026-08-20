import json

from affinity_repair.matrix import affinity_matrix
from affinity_repair.reporting import build_affinity_report, build_zip


def _summary(candidate_id):
    affinity = candidate_id in {"C2", "C3"}
    metrics = {
        "visible_dice": 0.9, "visible_cldice": 0.9, "latent_skeleton_f1_2px": 0.9,
        "endpoint_f1": 0.9, "gap_recovery_rate": 0.9, "false_bridge_rate": 0.4,
        "hard_affinity_macro_ap": 0.9 if affinity else None,
        "matched_negative_gap_auroc": 0.9 if affinity else None,
        "learned_beta": 0.2 if affinity else 0.0,
        "true_minus_false_affinity_ci95": [0.2, 0.1, 0.3],
        "beta_on_minus_off_latent_skeleton_f1_ci95": [0.02, 0.01, 0.03],
        "per_stratum": {
            name: {"edge_count": 10, "positive_count": 5, "average_precision": 0.9, "auroc": 0.9}
            for name in (
                "acute_angle_crossing", "similar_tangent_crossing", "nontrivial_pairing",
                "crossing_near_junction", "near_parallel_close", "matched_negative_gap",
            )
        } if affinity else {},
    }
    return {"candidate_id": candidate_id, "selected_visible_threshold": 0.5, "metrics": metrics}


def test_report_and_zip_are_machine_linked(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    for spec in affinity_matrix():
        (validation / f"{spec.candidate_id}-{spec.run_hash}.json").write_text(json.dumps(_summary(spec.candidate_id)))
    gate = {
        "status": "AFFINITY_MECHANISM_FAIL", "selected_candidate": None,
        "confirm_authorized": False,
        "decisions": {
            name: {"checks": {"causal_topology_ci": False}, "all_gates_pass": False}
            for name in ("C2", "C3")
        },
    }
    (tmp_path / "mechanism_gate.json").write_text(json.dumps(gate))
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "benchmark_v4_config.json").write_text("{}")
    result = build_affinity_report(tmp_path)
    assert result["status"] == "AFFINITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"
    package = build_zip(tmp_path)
    assert package["crc"] == "PASS"
    numbers = json.loads((tmp_path / "final" / "THESIS_NUMBERS.json").read_text())
    assert numbers["expert_data_accessed"] is False
