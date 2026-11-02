#!/usr/bin/env python3
"""Evaluate local CT28 copy-count ablation packs with the frozen clean-validation gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
SCENES = REPO / "artifacts/varvara/ct28_graph_lira"
GRAPH = Path(__file__).with_name("run_ct28_graph_lira.py")
TEST_IDS = {954, 958, 972, 973, 980, 984}


def validate_cached_evaluation(evidence: Path) -> None:
    """Reject stale or partial results instead of silently mixing model versions."""
    out = evidence / "clean_eval"
    protocol_path = out / "graph_lira_protocol.json"
    if not protocol_path.exists():
        raise ValueError(f"cached evaluation has no protocol: {out}")
    protocol = json.loads(protocol_path.read_text())
    expected = {"tau": .85, "consistency_gate": .60, "perturbations": 15,
                "position_noise_mm": .8, "tangent_noise_deg": 20., "seed": 20260920}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"cached evaluation uses different gate/noise settings: {out}")
    identity = {"geometry_only": "geometry_score", "ct_conditioned": {
        "PAIR": "geometry_plus_ct_score", "JUNCTION": "geometry_score"}}
    if protocol.get("identity_score_by_model") != identity:
        raise ValueError(f"cached evaluation uses different identity selection: {out}")
    paths = {"scene_plan": SCENES / "scene_plan_train_val.csv",
             "scene_endpoints": SCENES / "scene_endpoints_train_val.csv",
             "scene_candidates": SCENES / "scene_candidates_train_val.csv",
             "local_scores": evidence / "relation_local_scores_train_val.csv",
             "relation_predictions": evidence / "relation_predictions_val.csv",
             "relation_models": evidence / "relation_models.joblib"}
    for name, path in paths.items():
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != protocol.get("source_sha256", {}).get(name):
            raise ValueError(f"cached evaluation source changed: {name} in {out}")
    outputs = protocol.get("output_sha256", {})
    required = {"predictions": "graph_lira_predictions_val.csv",
                "consistency": "perturbation_consistency_val.csv",
                "metrics": "graph_lira_structural_metrics_val.csv",
                "per_patient": "graph_lira_per_patient_val.csv"}
    for name, filename in required.items():
        path = out / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != outputs.get(name):
            raise ValueError(f"cached evaluation output missing or changed: {name} in {out}")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def evaluate(evidence: Path) -> str:
    out = evidence / "clean_eval"
    if (out / "graph_lira_predictions_val.csv").exists():
        validate_cached_evaluation(evidence)
        return evidence.name
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(GRAPH),
        "--scene-dir", str(SCENES),
        "--evidence-dir", str(evidence),
        "--out-dir", str(out),
        "--ct-conditioned-identity-score", "geometry_plus_ct_score",
        "--ct-conditioned-junction-identity-score", "geometry_score",
        "--bootstrap-repetitions", "100",
    ]
    with (out / "run.log").open("w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return evidence.name


def summarize(evidence: Path) -> tuple[dict, pd.DataFrame]:
    root = evidence / "clean_eval"
    frame = pd.read_csv(root / "graph_lira_predictions_val.csv")
    frame = frame[frame.model.eq("ct_conditioned")].copy()
    if len(frame) != 368 or set(frame.scan_id.astype(int)) & TEST_IDS or frame.scan_id.nunique() != 5:
        raise ValueError(f"unexpected validation cohort: {evidence}")
    repair = frame[frame.relation_class.ne("NONE")]
    accepted = frame[frame.accepted.eq(1)]
    consistency = pd.read_csv(root / "perturbation_consistency_val.csv")
    consistency = consistency[consistency.model.eq("ct_conditioned")]
    if len(consistency) != 368:
        raise ValueError(f"incomplete consistency rows: {evidence}")
    detail = frame.merge(consistency[["scene_id", "baseline_relation_key", "perturbation_keys"]],
                         on="scene_id", validate="one_to_one")
    components = []
    for row in detail.itertuples(index=False):
        keys = json.loads(row.perturbation_keys)
        if len(keys) != 15:
            raise ValueError(f"evaluation perturbation count changed: {row.scene_id}")
        base_type = row.baseline_relation_key.split("|")[0]
        type_matches = sum(key.split("|")[0] == base_type for key in keys)
        exact_matches = sum(key == row.baseline_relation_key for key in keys)
        components.append({
            "variant": evidence.name, "scene_id": row.scene_id,
            "scan_id": row.scan_id, "scene_family": row.scene_family,
            "relation_class": row.relation_class,
            "pre_consistency_exact": row.pre_consistency_exact,
            "accepted": row.accepted,
            "type_consistency": type_matches / 15,
            "identity_given_type": exact_matches / type_matches if type_matches else 0.0,
            "exact_key_repeat_fraction": exact_matches / 15,
            "combined_consistency": row.combined_consistency,
        })
    components_frame = pd.DataFrame(components)
    blocked = components_frame[(components_frame.relation_class.ne("NONE"))
                               & components_frame.pre_consistency_exact.eq(1)
                               & components_frame.accepted.eq(0)]
    result = {
        "variant": evidence.name,
        "pre_exact": int(repair.pre_consistency_exact.sum()),
        "pre_false": int(frame.pre_consistency_false.sum()),
        "post_exact": int(repair.exact.sum()),
        "post_false": int(frame.false_scene.sum()),
        "post_incomplete": int(accepted.incomplete.sum()),
        "accepted": len(accepted),
        "exact_pair_only": int(accepted.loc[accepted.scene_family.eq("pair_only"), "exact"].sum()),
        "exact_junction_only": int(accepted.loc[accepted.scene_family.eq("junction_only"), "exact"].sum()),
        "exact_mixed": int(accepted.loc[accepted.scene_family.eq("mixed"), "exact"].sum()),
        "blocked_pre_exact": len(blocked),
        "blocked_type_below_0_6": int(blocked.type_consistency.lt(0.6).sum()),
        "blocked_type_at_least_0_6": int(blocked.type_consistency.ge(0.6).sum()),
        "repair_type_consistency_mean": float(components_frame.loc[
            components_frame.relation_class.ne("NONE"), "type_consistency"].mean()),
        "repair_identity_given_type_mean": float(components_frame.loc[
            components_frame.relation_class.ne("NONE"), "identity_given_type"].mean()),
    }
    return result, components_frame


def main() -> None:
    config = args()
    root = config.root.resolve()
    names = config.variants or sorted(path.name for path in root.iterdir()
                                      if path.is_dir() and (path / "relation_models.joblib").exists())
    if not names:
        raise ValueError("no evidence variants found")
    evidence_dirs = [root / name for name in names]
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(evaluate, path): path.name for path in evidence_dirs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED {name}; inspect {root / name / 'clean_eval/run.log'}", flush=True)
                raise
            print(f"DONE {name}", flush=True)
    rows, details, patients = [], [], []
    for evidence in evidence_dirs:
        summary, components = summarize(evidence)
        rows.append(summary)
        details.append(components)
        patient_frame = pd.read_csv(evidence / "clean_eval/graph_lira_per_patient_val.csv")
        patient_frame = patient_frame[patient_frame.model.eq("ct_conditioned")].copy()
        patient_frame.insert(0, "variant", evidence.name)
        patients.append(patient_frame)
    pd.DataFrame(rows).to_csv(root / "comparison_clean_val.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(root / "consistency_components_val.csv", index=False)
    pd.concat(patients, ignore_index=True).to_csv(root / "per_patient_comparison_val.csv", index=False)


if __name__ == "__main__":
    main()
