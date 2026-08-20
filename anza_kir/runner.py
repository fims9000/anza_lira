"""End-to-end bounded ANZA-KIR IR0--IR2 runner."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anza_ks_k2.evaluation import calibration_curve, pixel_summary, select_threshold

from .benchmark import NATURAL_SIZES, POOL_SIZES, SEEDS, TASKS, generate_sample, selected_hash
from .evaluation import apply_gates, pair_rows, pair_summary, paired_bootstrap
from .model import KIR_VARIANTS, build_base_model, build_kir_model
from .protocol import protocol, protocol_hash
from .training import compute_feature_norm, load_base_state, predict_records, train_ir1_base, train_ir2_variant


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/anza_kir/ir2"
FREEZE = RESULT / "freeze"
CHECKPOINTS = ROOT.parent / "_wip_backups/anza_lira/anza_kir_checkpoints"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, Any]:
    paths = sorted((ROOT / "anza_kir").glob("*.py")) + sorted((ROOT / "scripts").glob("*anza_kir*.py")) + sorted((ROOT / "tests").glob("test_anza_kir*.py"))
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]; digest = hashlib.sha256()
    for row in rows: digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": rows, "sha256": digest.hexdigest()}


def k2_source_sha() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "anza_ks_k2").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def write_ir0() -> dict[str, Any]:
    metrics = json.loads((ROOT / "results/anza_ks/k2/metrics.json").read_text())
    m0 = metrics["variants"]["M0_backbone"]; m1 = metrics["variants"]["M1_static"]; m4 = metrics["variants"]["M4_anza_ks"]
    value = {
        "parent_status": metrics["status"],
        "old_false_accept_floor": {"M0": m0["mechanism"]["false_positive_count"], "M1": m1["mechanism"]["false_positive_count"], "denominator": m0["mechanism"]["scene_count"]},
        "old_M4_minus_M1": {key: m4["natural_primary"][key] - m1["natural_primary"][key] for key in ("dice", "cldice", "fragmentation")},
        "old_structural_projection_directly_supervised": False,
        "old_k2_source_sha256": k2_source_sha(),
        "interpretation": "Motivation only; frozen K2 STOP is unchanged.",
    }
    _json(RESULT / "ir0_forensic.json", value)
    (RESULT / "ANZA_KIR_IR0_FORENSIC.md").write_text(
        "# ANZA-KIR IR0 Forensic\n\nThe frozen K2 status remains `STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED`. "
        f"M0/M1 had {value['old_false_accept_floor']['M0']}/{value['old_false_accept_floor']['denominator']} and {value['old_false_accept_floor']['M1']}/{value['old_false_accept_floor']['denominator']} mechanism false accepts. "
        "The old 1x1 structural projection was not directly supervised; old M4 natural deltas are diagnostics only.\n"
    )
    return value


def _score_pool(base: torch.nn.Module, stream: str, *, device: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []; base.eval()
    with torch.inference_mode():
        for start in range(0, POOL_SIZES[stream], 64):
            local = [generate_sample(stream, index, allow_confirm=True) for index in range(start, min(start + 64, POOL_SIZES[stream]))]
            images = torch.from_numpy(np.stack([sample["image"] for sample in local])).to(device)
            probabilities = torch.sigmoid(base(images)).cpu().numpy()[:, 0]
            for offset, (probability, sample) in enumerate(zip(probabilities, local, strict=True)):
                target = np.asarray(sample["target"], dtype=bool); distractor = np.asarray(sample["distractor"], dtype=bool)
                target_score = float(probability[target].mean()); distractor_score = float(probability[distractor].mean())
                rows.append({"index": start + offset, "task": sample["mechanism_task"], "target_score": target_score, "distractor_score": distractor_score, "margin": target_score - distractor_score})
            if start and start % 6400 == 0: print(f"phase=ANZA-KIR-MINE stream={stream} scored={start}/{POOL_SIZES[stream]}", flush=True)
    return rows


def _select_bottom(rows: list[dict[str, Any]]) -> list[int]:
    by_task = {task: sorted((row for row in rows if row["task"] == task), key=lambda row: (row["margin"], row["index"])) for task in TASKS}
    selected_by_task = {task: [row["index"] for row in values[: int(len(values) * 0.20)]] for task, values in by_task.items()}
    # Round-robin preserves balance even if a bounded training subset is taken.
    selected = []
    for offset in range(max(map(len, selected_by_task.values()))):
        for task in TASKS:
            if offset < len(selected_by_task[task]): selected.append(selected_by_task[task][offset])
    return selected


def mine_and_freeze(base_checkpoint: Path, protocol_sha256: str, *, device: str) -> dict[str, Any]:
    manifest_path = FREEZE / "benchmark_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("protocol_sha256") != protocol_sha256 or existing.get("base_checkpoint_sha256") != _sha(base_checkpoint): raise ValueError("frozen KIR benchmark provenance mismatch")
        return existing
    base = build_base_model().to(device); base.load_state_dict(load_base_state(base_checkpoint, device)); base.eval()
    pools = {}; selected = {}
    for stream in POOL_SIZES:
        rows = _score_pool(base, stream, device=device); indices = _select_bottom(rows); selected[stream] = indices
        local = {row["index"]: row for row in rows}; selected_rows = [local[index] for index in indices]
        pools[stream] = {"pool_size": len(rows), "selected_count": len(indices), "selected_fraction": len(indices) / len(rows), "selected_pair_error": float(np.mean([row["distractor_score"] >= row["target_score"] for row in selected_rows])), "selected_mean_margin": float(np.mean([row["margin"] for row in selected_rows])), "selected_indices": indices, "selected_content_sha256": selected_hash(stream, indices)}
        _json(FREEZE / f"{stream}_base_scores.json", {"rows": rows})
    dev_error = pools["mine-dev"]["selected_pair_error"]
    valid = 0.10 <= dev_error <= 0.40 and pools["mine-dev"]["selected_count"] >= 2000 and sum(POOL_SIZES.values()) >= 50_000
    natural_hashes = {stream: selected_hash(stream, list(range(size))) for stream, size in NATURAL_SIZES.items()}
    manifest = {"version": protocol()["version"], "protocol_sha256": protocol_sha256, "base_checkpoint_sha256": _sha(base_checkpoint), "selection": "bottom frozen 20% base margin within each task; no residual/static/Cat/Shear features", "candidate_total": sum(POOL_SIZES.values()), "pools": pools, "natural_hashes": natural_hashes, "seeds": SEEDS, "base_pair_error_valid": valid, "confirm_evaluated": False}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(); manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest(); _json(manifest_path, manifest)
    if not valid: raise RuntimeError(f"ANZA-KIR hard benchmark construction failed: dev base PairError={dev_error:.6f}")
    return manifest


def _load_model(variant: str, checkpoint: Path, base_checkpoint: Path, norm: dict[str, Any], device: str) -> torch.nn.Module:
    model = build_kir_model(variant, load_base_state(base_checkpoint, device), norm["methods"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); model.eval(); return model


def _report(metrics: dict[str, Any]) -> str:
    lines = ["# ANZA-KIR IR2 Report", "", f"Status: `{metrics['status']}`", "", "Frozen K2 remains a negative result. This is a separate evidence-anchored seed-41 residual experiment; confirm, CRACKS, expert, seeds 42/43, and controlled unfreezing stayed closed.", "", "| Variant | PairError | Mean margin | Pair AUROC | Dice | clDice | Fragmentation | Gamma |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in KIR_VARIANTS:
        row = metrics["variants"][variant]; gamma = row["run"]["history"][-1]["gamma"]
        lines.append(f"| {variant} | {row['hard']['pair_error']:.4f} | {row['hard']['mean_margin']:.4f} | {row['hard']['pair_auc']:.4f} | {row['natural']['dice']:.4f} | {row['natural']['cldice']:.4f} | {row['natural']['fragmentation']:.4f} | {gamma:.5f} |")
    lines += ["", "## Frozen gates", "", f"`{json.dumps(metrics['gates'], sort_keys=True)}`", "", "## Claim boundary", "", "A seed-41 synthetic result cannot establish CRACKS or real seismic improvement. An Anosov-specific statement is permitted only if the R3-vs-R1 gate passes.", ""]
    return "\n".join(lines)


def run(*, device: str = "cuda") -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True); FREEZE.mkdir(parents=True, exist_ok=True); write_ir0()
    value = protocol(); phash = protocol_hash(value); _json(FREEZE / "protocol.json", value)
    ir1 = train_ir1_base(value, phash, CHECKPOINTS, RESULT / "ir1", device=device); base_checkpoint = Path(ir1["checkpoint"])
    benchmark = mine_and_freeze(base_checkpoint, phash, device=device)
    norm_path = FREEZE / "feature_norm.json"
    if not norm_path.exists(): _json(norm_path, compute_feature_norm(base_checkpoint, [("residual-train-natural", i) for i in range(64)], device=device))
    norm = json.loads(norm_path.read_text()); _json(FREEZE / "source_manifest.json", source_manifest())
    freeze_receipt = {"status": "ANZA_KIR_IR2_AUTHORIZED", "protocol_sha256": phash, "source_sha256": source_manifest()["sha256"], "benchmark_manifest_sha256": benchmark["manifest_sha256"], "base_checkpoint_sha256": _sha(base_checkpoint), "feature_norm_sha256": _sha(norm_path), "old_k2_source_sha256": k2_source_sha(), "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    _json(FREEZE / "pre_ir2_receipt.json", freeze_receipt)
    runs = {}
    hard_train = benchmark["pools"]["mine-train"]["selected_indices"]
    for variant in KIR_VARIANTS:
        runs[variant] = train_ir2_variant(variant, value, phash, base_checkpoint, norm, hard_train, CHECKPOINTS, RESULT / "runs", device=device)
    counts = {run["trainable_parameter_count"] for run in runs.values()}
    if len(counts) != 1: raise ValueError(f"IR2 trainable parameter mismatch: {counts}")

    thresholds = {}; variants = {}; hard_rows_by_variant = {}; natural_rows_by_variant = {}; raw_rows = []
    for variant in KIR_VARIANTS:
        model = _load_model(variant, Path(runs[variant]["checkpoint"]), base_checkpoint, norm, device)
        cal_prob, cal_samples = predict_records(model, [("calibration-natural", i) for i in range(NATURAL_SIZES["calibration-natural"])], device=device)
        curve = calibration_curve(cal_prob, cal_samples); threshold = select_threshold(curve, "dice"); thresholds[variant] = threshold
        natural_prob, natural_samples = predict_records(model, [("dev-natural", i) for i in range(NATURAL_SIZES["dev-natural"])], device=device)
        natural, natural_rows = pixel_summary(natural_prob, natural_samples, threshold); natural_rows_by_variant[variant] = natural_rows
        hard_indices = benchmark["pools"]["mine-dev"]["selected_indices"]
        hard_prob, hard_samples = predict_records(model, [("mine-dev", i) for i in hard_indices], device=device)
        paired = pair_rows(hard_prob, hard_samples); hard_rows_by_variant[variant] = paired; hard = pair_summary(paired)
        variants[variant] = {"run": runs[variant], "threshold": threshold, "natural": natural, "hard": hard}
        for row in natural_rows: raw_rows.append({"variant": variant, "split": "dev-natural", **row, "target_score": "", "distractor_score": "", "margin": "", "pair_error": "", "mechanism_task": "", "index": ""})
        for row in paired: raw_rows.append({"variant": variant, "split": "dev-hard", "dice": "", "precision": "", "recall": "", "cldice": "", "fragmentation": "", "foreground_fraction": "", **row})

    bootstraps = {}
    for label, control in (("R3_vs_R0", "R0_static_residual"), ("R3_vs_R1", "R1_shear_ks_residual"), ("R3_vs_R2", "R2_cat_raw_residual")):
        control_hard = hard_rows_by_variant[control]; candidate_hard = hard_rows_by_variant["R3_anza_kir"]
        control_nat = natural_rows_by_variant[control]; candidate_nat = natural_rows_by_variant["R3_anza_kir"]
        bootstraps[label] = {
            "pair_error": paired_bootstrap(np.asarray([c["pair_error"] - a["pair_error"] for c, a in zip(control_hard, candidate_hard, strict=True)])),
            "margin": paired_bootstrap(np.asarray([a["margin"] - c["margin"] for c, a in zip(control_hard, candidate_hard, strict=True)])),
            "dice": paired_bootstrap(np.asarray([a["dice"] - c["dice"] for c, a in zip(control_nat, candidate_nat, strict=True)])),
            "cldice": paired_bootstrap(np.asarray([a["cldice"] - c["cldice"] for c, a in zip(control_nat, candidate_nat, strict=True)])),
            "fragmentation": paired_bootstrap(np.asarray([c["fragmentation"] - a["fragmentation"] for c, a in zip(control_nat, candidate_nat, strict=True)])),
        }
    status, gates = apply_gates(variants, bootstraps)
    metrics = {"status": status, "seed": 41, "parent_status": "STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED", "protocol_sha256": phash, "benchmark_manifest_sha256": benchmark["manifest_sha256"], "base_checkpoint_sha256": _sha(base_checkpoint), "feature_norm_sha256": _sha(norm_path), "variants": variants, "bootstraps": bootstraps, "gates": gates, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False, "seeds_42_43_opened": False, "controlled_unfreezing_opened": False}
    _json(RESULT / "metrics.json", metrics); _json(RESULT / "threshold_freeze.json", {"source": "calibration-natural only", "thresholds": thresholds, "confirm_opened": False})
    with (RESULT / "raw_per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0])); writer.writeheader(); writer.writerows(raw_rows)
    (RESULT / "ANZA_KIR_IR2_REPORT.md").write_text(_report(metrics)); _json(RESULT / "TASK_STATE.json", {"status": status, "next_action": "STOP after bounded seed-41 IR2; apply frozen outcome rule", "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False})
    return metrics
