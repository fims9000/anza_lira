#!/usr/bin/env python3
"""Build a four-cell PAIR-fusion × relation-copy validation comparison."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import shlex

import joblib
import numpy as np
import pandas as pd

import train_ct28_relation_head as training


REPO = training.REPO_ROOT
ROOT = REPO / "varvara/anza_lira_local_results/ct28_copy_count_ablations"
TEST_IDS = {954, 958, 972, 973, 980, 984}


def bind_cached_evaluation(evidence: Path, reused: Path) -> None:
    """Record equivalent-model cache reuse without retaining stale source hashes."""
    pack = joblib.load(evidence / "relation_models.joblib")
    origin = joblib.load(reused / "relation_models.joblib")
    for key in ("local_models", "relation_heads", "geometry_relation_features",
                "ct_conditioned_relation_features"):
        if joblib.hash(pack[key]) != joblib.hash(origin[key]):
            raise ValueError(f"cached models are not equivalent: {key}")
    scores = pd.read_csv(evidence / "relation_local_scores_train_val.csv")
    if not scores.equals(pd.read_csv(reused / "relation_local_scores_train_val.csv")):
        raise ValueError("cached local scores are not equivalent")
    current = pd.read_csv(evidence / "relation_predictions_val.csv")
    previous = pd.read_csv(reused / "relation_predictions_val.csv")
    for model in ("geometry_only", "ct_conditioned"):
        left = current[current.model.eq(model)].set_index("scene_id").sort_index()
        right = previous[previous.model.eq(model)].set_index("scene_id").sort_index()
        if left.empty or not left.equals(right):
            raise ValueError(f"cached relation predictions differ: {model}")
    source = reused / "clean_eval/graph_lira_protocol.json"
    protocol = json.loads(source.read_text())
    protocol["cached_evaluation_origin"] = {
        "evidence_dir": str(reused.resolve()), "protocol_sha256": training.sha256(source),
        "equivalence": "local models, relation heads, feature lists, parsed local scores and both evaluated probability streams match",
    }
    protocol["evidence_dir"] = str(evidence.resolve())
    for name, filename in (("relation_models", "relation_models.joblib"),
                           ("local_scores", "relation_local_scores_train_val.csv"),
                           ("relation_predictions", "relation_predictions_val.csv")):
        protocol["source_sha256"][name] = training.sha256(evidence / filename)
    protocol["exact_run_command"] = shlex.join([
        "python", "scripts/research/ccta_graph_lira_safe_repair/run_ct28_graph_lira.py",
        "--evidence-dir", str(evidence), "--out-dir", str(evidence / "clean_eval"),
        "--ct-conditioned-identity-score", "geometry_plus_ct_score",
        "--ct-conditioned-junction-identity-score", "geometry_score",
        "--position-noise-mm", str(protocol["position_noise_mm"]),
        "--tangent-noise-deg", str(protocol["tangent_noise_deg"]),
        "--seed", str(protocol["seed"]),
        "--bootstrap-repetitions", str(protocol["bootstrap"]["repetitions"]),
    ])
    (evidence / "clean_eval/graph_lira_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-k", type=int, required=True)
    parser.add_argument("--relation-k", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    config = arguments()
    for k in (config.fusion_k, config.relation_k):
        if k not in (0, 3, 5, 10, 15, 30):
            raise ValueError("K must be a predeclared sweep value")
    for fk, rk in ((5, 15), (config.fusion_k, 15), (5, config.relation_k),
                   (config.fusion_k, config.relation_k)):
        fusion_dir = config.root / "fusion" / f"k{fk:02d}"
        relation_dir = config.root / "relation" / f"k{rk:02d}"
        fusion = pd.read_csv(fusion_dir / "relation_local_scores_train_val.csv")
        relation = pd.read_csv(relation_dir / "relation_local_scores_train_val.csv")
        if set(relation.scan_id.astype(int)) & TEST_IDS:
            raise ValueError("held-out test entered sweep")
        if not fusion[["scene_id", "candidate_id"]].equals(
            relation[["scene_id", "candidate_id"]]):
            raise ValueError("candidate identity/order changed")
        if not np.array_equal(fusion[["geometry_score", "ct_score"]].to_numpy(),
                              relation[["geometry_score", "ct_score"]].to_numpy()):
            raise ValueError("relation evidence changed between sweeps")
        fusion_protocol = json.loads((fusion_dir / "relation_model_protocol.json").read_text())
        relation_protocol = json.loads((relation_dir / "relation_model_protocol.json").read_text())
        if fusion_protocol["source_sha256"] != relation_protocol["source_sha256"]:
            raise ValueError("scene source SHA changed")
        scores = relation.copy()
        mask = scores.candidate_type.eq("PAIR")
        scores.loc[mask, "geometry_plus_ct_score"] = fusion.loc[mask, "geometry_plus_ct_score"].to_numpy()
        relation_pack = joblib.load(relation_dir / "relation_models.joblib")
        fusion_pack = joblib.load(fusion_dir / "relation_models.joblib")
        pack = copy.deepcopy(relation_pack)
        pack["local_models"]["PAIR"]["geometry_plus_ct"] = fusion_pack["local_models"]["PAIR"]["geometry_plus_ct"]
        out = config.root / "joint_check" / f"fusion_{fk:02d}_relation_{rk:02d}"
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "relation_local_scores_train_val.csv", index=False)
        joblib.dump(pack, out / "relation_models.joblib")
        (out / "relation_predictions_val.csv").write_bytes(
            (relation_dir / "relation_predictions_val.csv").read_bytes())
        reused = None
        if fk == 5:
            reused = relation_dir
        elif rk == 15:
            reused = fusion_dir
        if reused is not None and (reused / "clean_eval/graph_lira_predictions_val.csv").exists():
            reused_scores = pd.read_csv(reused / "relation_local_scores_train_val.csv")
            reused_predictions = pd.read_csv(reused / "relation_predictions_val.csv")
            current_predictions = pd.read_csv(out / "relation_predictions_val.csv")
            reused_ct = reused_predictions[reused_predictions.model.eq("ct_conditioned")].reset_index(drop=True)
            current_ct = current_predictions[current_predictions.model.eq("ct_conditioned")].reset_index(drop=True)
            if not scores.equals(reused_scores) or not current_ct.equals(reused_ct):
                raise ValueError("cached evaluation does not match combined evidence")
            shutil.copytree(reused / "clean_eval", out / "clean_eval", dirs_exist_ok=True)
            bind_cached_evaluation(out, reused)
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_ablation_not_frozen", "held_out_test_accessed": False,
            "source_sha256": relation_protocol["source_sha256"],
            "pair_fusion_train_copy_count": fk, "relation_train_copy_count": rk,
            "reused_evaluation_from": str(reused) if reused is not None else None,
            "pair_identity": "geometry_plus_ct_score", "junction_identity": "geometry_score",
            "tau": 0.85, "consistency": 0.60, "evaluation_perturbations": 15,
        }, indent=2)+"\n")
        print(f"fusion K={fk}, relation K={rk} ready", flush=True)


if __name__ == "__main__":
    main()
