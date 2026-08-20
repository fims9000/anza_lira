"""Phase 3C-A no-training forensic orchestrator."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from anza2.eval.low_fpr import low_fpr_metrics, sampled_operating_curve
from anza2.eval.mechanism_metrics import aggregate_mechanism, mechanism_observations
from anza2_experiment.learned_affinity import LearnedAffinityModel
from anza2_experiment.synthetic_mechanism import _branch_rows, _path_rows
from anza2_experiment.synthetic_replacement import REPLACEMENT_CONFIRM_SEED_BASE
from models.anza2.affinity import ANZA2StructuralAffinity
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v4 import benchmark_v4_config, generate_sample_v4

from .component_replacement import (
    COMPONENT_MATRIX, ORACLE_GEOMETRY_SEMANTICS, align_learned_field,
    component_replacements, oracle_field_from_sample,
)
from .field_fidelity import aggregate_fidelity, field_fidelity_row
from .fusion_audit import summarize_fusion
from .root_cause import classify_root_cause


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3c_a"
PHASE3B_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3b"
SEEDS = (41, 42, 43)
DEVELOPMENT_INDICES = tuple(range(512))
ABSOLUTE_MECHANISM_THRESHOLD = 0.04482836276292801


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    checkpoints = {str(seed): digest(PHASE3B_ROOT / "runs" / f"causal_s{seed}" / "checkpoint-last.pt") for seed in SEEDS}
    return {
        "version": "anza2_phase3c_a_learned_field_forensics_v1",
        "phase2b_protocol_sha256": "5b1789554722f91a28e32590f897d7c9f6c2642f5a83994c86ea43c352d4cd64",
        "phase3b_protocol_sha256": "552a696ba062ac0ac488ae5d3a5e8396b520e6f9cc4577aab5cbbfad2bf7d6c8",
        "frozen_checkpoints": checkpoints,
        "component_matrix": COMPONENT_MATRIX,
        "development_stream": "CrossingTraceBench-v4 validation[0:512]",
        "previously_scored_phase3b_indices": "validation[0:256]",
        "forensic_only_additional_indices": "validation[256:512] context strata; no fitting/model selection",
        "benchmark_v4_sha256": benchmark_v4_config()["sha256"],
        "seeds": list(SEEDS), "image_size": 64,
        "mechanism_absolute_threshold": ABSOLUTE_MECHANISM_THRESHOLD,
        "low_fpr_budget": 0.05,
        "oracle_geometry_semantics": ORACLE_GEOMETRY_SEMANTICS,
        "curved_gap_status": "not present in frozen v4 validation; Phase-2B reproduction plus curved-trace edge recall reported",
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
        "next_action": "stop after selecting one RC1-RC7; do not run repair",
    }


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reproduce_phase2b() -> dict[str, Any]:
    thresholds = json.loads((PROJECT_ROOT / "results/anza2/phase2/threshold_freeze.json").read_text())["thresholds"]
    paths = _path_rows("confirm", seed_base=REPLACEMENT_CONFIRM_SEED_BASE)
    branches = _branch_rows("confirm", seed_base=REPLACEMENT_CONFIRM_SEED_BASE)
    result = {}
    for method in ("anza2_absolute", "legacy_global_normalized"):
        path_labels = np.asarray([row["label"] for row in paths], dtype=bool)
        path_hits = np.asarray([row[method] >= thresholds[method] for row in paths], dtype=bool)
        branch_hits = np.asarray([row[method] >= thresholds[method] for row in branches], dtype=bool)
        by_case = {}
        for case in sorted({row["case"] for row in branches}):
            selected = np.asarray([row["case"] == case for row in branches])
            by_case[case] = float(branch_hits[selected].mean())
        result[method] = {
            "branch_recall": float(branch_hits.mean()), "branch_recall_by_case": by_case,
            "path_tpr": float(path_hits[path_labels].mean()), "false_bridge_fpr": float(path_hits[~path_labels].mean()),
        }
    frozen = json.loads((PROJECT_ROOT / "results/anza2/phase2b/metrics.json").read_text())
    reproduced = bool(
        result["anza2_absolute"]["branch_recall"] == frozen["branch_metrics"]["anza2_absolute"]["recall"]
        and result["legacy_global_normalized"]["branch_recall"] == frozen["branch_metrics"]["legacy_global_normalized"]["recall"]
        and result["anza2_absolute"]["path_tpr"] == frozen["path_metrics"]["anza2_absolute"]["tpr"]
        and result["anza2_absolute"]["false_bridge_fpr"] == frozen["path_metrics"]["anza2_absolute"]["fpr"]
    )
    return {"status": "PASS" if reproduced else "FAIL", "reproduced": reproduced, "methods": result}


def _load_model(seed: int, device: torch.device, protocol_hash: str) -> LearnedAffinityModel:
    checkpoint = PHASE3B_ROOT / "runs" / f"causal_s{seed}" / "checkpoint-last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("protocol_sha256") != protocol_hash:
        raise ValueError("Phase-3B checkpoint identity mismatch")
    if payload.get("cracks_data_accessed") is not False or payload.get("expert_data_accessed") is not False:
        raise ValueError("Phase-3B checkpoint data-lock violation")
    model = LearnedAffinityModel(initial_beta=0.05).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _per_case_rows(mechanism_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for config in COMPONENT_MATRIX:
        selected_config = [row for row in mechanism_rows if row["config"] == config]
        for case in sorted({row["case"] for row in selected_config}):
            group = [row for row in selected_config if row["case"] == case]
            metrics = aggregate_mechanism(group, threshold=ABSOLUTE_MECHANISM_THRESHOLD)
            output.append({"config": config, "case": case, **metrics})
    return output


def run_forensics(output_root: Path = OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(); protocol_hash = canonical_hash(protocol)
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("Phase-3C-A protocol drift")
    protocol_path.write_text(encoded); (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")

    phase2b = reproduce_phase2b()
    torch.set_num_threads(min(2, torch.get_num_threads())); device_obj = torch.device(device)
    affinity = ANZA2StructuralAffinity().to(device_obj)
    edge_scores = {name: {"positive": [], "negative": []} for name in COMPONENT_MATRIX}
    mechanism_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    fusion_records: list[dict[str, Any]] = []
    phase3b_protocol_hash = json.loads((PHASE3B_ROOT / "protocol.json").read_text())
    expected_phase3b_hash = canonical_hash(phase3b_protocol_hash)
    for seed in SEEDS:
        model = _load_model(seed, device_obj, expected_phase3b_hash)
        beta = float(model.combiner.beta.detach())
        with torch.inference_mode():
            for index in DEVELOPMENT_INDICES:
                sample = generate_sample_v4("validation", index, image_size=64)
                image = torch.as_tensor(sample["image"], dtype=torch.float32, device=device_obj).unsqueeze(0)
                output = model(image, use_anza=True)
                oracle, valid = oracle_field_from_sample(sample, device=device_obj)
                learned, _mapping = align_learned_field(output["field"], oracle, valid)
                replacements = component_replacements(oracle, learned)
                targets = build_affinity_targets(sample, tuple(affinity.offsets))
                positive = np.asarray(targets["affinity_positive"], dtype=bool)
                negative = np.asarray(targets["affinity_hard_negative"], dtype=bool)
                relations = {}
                for name, field in replacements.items():
                    relation = affinity(field)[0].cpu().numpy().astype(np.float32)
                    relations[name] = relation
                    if positive.any(): edge_scores[name]["positive"].append(relation[positive])
                    if negative.any(): edge_scores[name]["negative"].append(relation[negative])
                    observations = mechanism_observations(sample, relation)
                    for row in observations:
                        mechanism_rows.append({
                            "config": name, "seed": seed, "sample_index": index,
                            "case": sample["case"], **row,
                        })
                fidelity_rows.append(field_fidelity_row(
                    sample, learned, oracle, valid, seed=seed, sample_index=index
                ))
                valid_edge = positive | negative
                generic_logits = output["generic_logits"][0].cpu().numpy().astype(np.float32)
                raw_anza = relations["F1_full_learned"]
                fused = torch.sigmoid(output["logits"])[0].cpu().numpy().astype(np.float32)
                generic = torch.sigmoid(output["generic_logits"])[0].cpu().numpy().astype(np.float32)
                effective = beta * np.log(np.clip(raw_anza, 1e-6, 1 - 1e-6) / np.clip(1 - raw_anza, 1e-6, 1))
                fusion_records.append({
                    "seed": seed, "sample_index": index, "case": sample["case"],
                    "positive": positive[valid_edge], "negative": negative[valid_edge],
                    "raw_anza": raw_anza[valid_edge], "generic": generic[valid_edge],
                    "fused": fused[valid_edge], "effective_term": effective[valid_edge],
                    "generic_logits": generic_logits[valid_edge],
                })
                if index % 128 == 127:
                    print(f"phase=anza2_phase3c_a seed={seed} samples={index + 1}/512 training=NO confirm=CLOSED", flush=True)

    matrix: dict[str, dict[str, Any]] = {}
    component_rows = []
    curve_rows = []
    for name in COMPONENT_MATRIX:
        positive = np.concatenate(edge_scores[name]["positive"])
        negative = np.concatenate(edge_scores[name]["negative"])
        edge = low_fpr_metrics(positive, negative)
        mechanism = aggregate_mechanism(
            [row for row in mechanism_rows if row["config"] == name],
            threshold=ABSOLUTE_MECHANISM_THRESHOLD,
        )
        matrix[name] = {**mechanism, **edge}
        component_rows.append({"config": name, **matrix[name]})
        for row in sampled_operating_curve(positive, negative, max_points=201):
            curve_rows.append({"config": name, **row})
    fusion_rows, fusion = summarize_fusion(fusion_records)
    fidelity = aggregate_fidelity(fidelity_rows)
    root_cause = classify_root_cause(matrix, fusion, phase2b_reproduced=phase2b["reproduced"])

    (output_root / "metrics.json").write_text(json.dumps({
        "status": "PHASE3C_A_FORENSIC_PASS" if root_cause["repair_authorized"] else root_cause["root_cause"],
        "protocol_sha256": protocol_hash, "phase2b_reproduction": phase2b,
        "component_matrix": matrix, "fusion": fusion, "root_cause": root_cause,
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (output_root / "field_fidelity.json").write_text(json.dumps(fidelity, indent=2, sort_keys=True) + "\n")
    (output_root / "fusion_audit.json").write_text(json.dumps(fusion, indent=2, sort_keys=True) + "\n")
    (output_root / "root_cause.json").write_text(json.dumps(root_cause, indent=2, sort_keys=True) + "\n")
    _write_csv(output_root / "component_replacement.csv", component_rows)
    _write_csv(output_root / "fusion_audit.csv", fusion_rows)
    _write_csv(output_root / "per_case.csv", _per_case_rows(mechanism_rows))
    _write_csv(output_root / "operating_curve.csv", curve_rows)
    (output_root / "bootstrap.json").write_text(json.dumps({
        "status": "NOT_PRIMARY_INFERENCE_FORENSIC",
        "reason": "Phase 3C-A is deterministic component localization; the frozen Phase-3B sample bootstrap remains inferential evidence.",
        "phase3b_delta": 0.0002740574439678621,
        "phase3b_ci95": [0.00011431401376591499, 0.0004369868269516147],
        "unit": "synthetic sample after seed averaging", "resamples": 10000,
    }, indent=2, sort_keys=True) + "\n")
    (output_root / "split_manifest.json").write_text(json.dumps({
        "stream": "CrossingTraceBench-v4 validation", "indices": [0, 511], "count": 512,
        "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (output_root / "data_access_log.json").write_text(json.dumps({
        "synthetic": "Phase-2B already-opened oracle stream plus v4 validation development stream",
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (output_root / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
    }, indent=2, sort_keys=True) + "\n")
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    (output_root / "code_state.json").write_text(json.dumps({
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "git_status_lines": git_status, "commit_created": False,
    }, indent=2, sort_keys=True) + "\n")
    return json.loads((output_root / "metrics.json").read_text())
