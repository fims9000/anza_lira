"""Independent synthetic confirmation of pair-gated path completion.

The confirmation intentionally uses generator-visible endpoints so it isolates
the learned pair decision from upstream segmentation errors.  No latent
connectivity, gap mask, or instance lineage is available to inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from connectivity_repair.balanced_metrics import balanced_matched_pair_metrics
from path_completion.pair_classifier import (
    EndpointPairClassifier,
    PAIR_PROTOCOL,
    _canonical_hash,
    _single_pair,
    oriented_pair_crop,
)
from path_completion.widest_path import EndpointPair, rasterize_path
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v5 import benchmark_v5_config, generate_sample_v5
from synthetic.evaluation_corrected import evaluate_sample_corrected


CONFIRM_PROTOCOL = {
    "version": "anza_learned_pair_path_confirm_v1_frozen",
    "benchmark_sha256": benchmark_v5_config()["sha256"],
    "stream": "v5 confirm positive 0:128 plus matched negative 128:256",
    "pair_checkpoint": "hash-locked endpoint classifier; no retraining",
    "candidate_endpoints": "generator-visible binary support; no latent connectivity",
    "path": "deterministic straight shortest raster between accepted endpoints",
    "d_max": "train-frozen path oracle value",
    "raster_radius": "train-frozen path oracle value",
    "decision_threshold": "pair classifier train-frozen threshold",
    "gates": {
        "pair_auroc_min": 0.85,
        "positive_gap_recovery_min": 0.70,
        "false_bridge_max": 0.20,
        "visible_dice_loss_max": 0.005,
        "visible_cldice_non_decrease": True,
        "endpoint_f1_improvement_min": 0.02,
    },
    "test": "LOCKED_UNOPENED",
    "cracks_expert": "FORBIDDEN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def straight_shortest_path(pair: EndpointPair) -> tuple[tuple[int, int], ...]:
    steps = int(max(abs(pair.second[0] - pair.first[0]), abs(pair.second[1] - pair.first[1]))) + 1
    yy = np.rint(np.linspace(pair.first[0], pair.second[0], steps)).astype(int)
    xx = np.rint(np.linspace(pair.first[1], pair.second[1], steps)).astype(int)
    points: list[tuple[int, int]] = []
    for point in zip(yy.tolist(), xx.tolist()):
        if not points or points[-1] != point:
            points.append(point)
    return tuple(points)


def load_frozen_pair_classifier(
    project_root: Path,
    device: torch.device,
) -> tuple[EndpointPairClassifier, dict[str, Any]]:
    root = Path(project_root) / "results" / "path_completion" / "pair_classifier"
    result = json.loads((root / "result.json").read_text())
    checkpoint_path = root / "checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_protocol_hash = _canonical_hash(PAIR_PROTOCOL)
    if result.get("status") != "ENDPOINT_PAIR_CLASSIFIER_PASS":
        raise PermissionError("pair classifier did not pass its frozen validation gate")
    if result.get("protocol_sha256") != expected_protocol_hash or payload.get("protocol_sha256") != expected_protocol_hash:
        raise PermissionError("pair classifier protocol hash drift")
    if float(payload["threshold"]) != float(result["threshold_frozen_from_train"]):
        raise PermissionError("pair classifier threshold drift")
    model = EndpointPairClassifier()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol_sha256": expected_protocol_hash,
        "threshold": float(payload["threshold"]),
        "d_max_px": float(result["train_frozen_geometry"]["d_max_px"]),
        "path_radius_px": int(result["train_frozen_geometry"]["path_radius_px"]),
    }


def score_pair(
    model: EndpointPairClassifier,
    sample: dict[str, Any],
    pair: EndpointPair,
    device: torch.device,
) -> float:
    inputs = torch.from_numpy(oriented_pair_crop(sample, pair)[None]).to(device)
    with torch.inference_mode():
        return float(torch.sigmoid(model(inputs))[0].cpu())


def pair_gated_completion(
    sample: dict[str, Any],
    pair: EndpointPair,
    *,
    score: float,
    threshold: float,
    path_radius: int,
) -> np.ndarray:
    visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
    completion = visible.copy()
    if float(score) >= float(threshold):
        completion |= rasterize_path(straight_shortest_path(pair), visible.shape, radius=int(path_radius))
    return completion


def _mean(rows: list[dict[str, Any]], key: str, *, case: str | None = None) -> float:
    values = [float(row[key]) for row in rows if case is None or row["case"] == case]
    return float(np.mean(values))


def run_learned_confirm(project_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    torch_device = torch.device(device)
    model, frozen = load_frozen_pair_classifier(project_root, torch_device)
    rows: list[dict[str, Any]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    visual: dict[int, dict[str, Any]] = {}
    for index in range(2 * PAIRED_GAP_COUNT):
        sample = generate_sample_v5("confirm", index)
        pair = _single_pair(sample, frozen["d_max_px"])
        score = score_pair(model, sample, pair, torch_device)
        completion = pair_gated_completion(
            sample,
            pair,
            score=score,
            threshold=frozen["threshold"],
            path_radius=frozen["path_radius_px"],
        )
        visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
        base = evaluate_sample_corrected(visible, sample, predicted_completion_mask=visible)["family_a"]
        completed = evaluate_sample_corrected(visible, sample, predicted_completion_mask=completion)["family_a"]
        is_positive = sample["case"] == "fault_with_gap"
        (positive_scores if is_positive else negative_scores).append(score)
        negative_gap = np.asarray(sample["negative_gap_mask"], dtype=bool)
        rows.append({
            "index": index,
            "pair_id": int(sample["pair_id"]),
            "case": sample["case"],
            "pair_score": score,
            "accepted": score >= frozen["threshold"],
            "base_visible_dice": base["visible_dice"],
            "completion_visible_dice": completed["visible_dice"],
            "base_visible_cldice": base["visible_cldice"],
            "completion_visible_cldice": completed["visible_cldice"],
            "base_latent_cldice": base["latent_cldice"],
            "completion_latent_cldice": completed["latent_cldice"],
            "base_endpoint_f1": base["endpoint_f1"],
            "completion_endpoint_f1": completed["endpoint_f1"],
            "gap_recovery_rate": completed["gap_recovery_rate"],
            "false_bridge_rate": completed["false_bridge_rate"],
            "negative_gap_filled_pixel_fraction": float(completion[negative_gap].mean()) if negative_gap.any() else 0.0,
            "modified_pixels": int(np.logical_xor(completion, visible).sum()),
        })
        if index in {0, PAIRED_GAP_COUNT}:
            visual[index] = {
                "image": np.asarray(sample["image"]),
                "visible": visible,
                "completion": completion,
                "latent": np.asarray(sample["latent_fault_mask"], dtype=bool),
                "score": score,
            }
    pair_metrics = balanced_matched_pair_metrics(
        np.asarray(positive_scores), np.asarray(negative_scores), threshold=frozen["threshold"]
    )
    positive_case = "fault_with_gap"
    negative_case = "negative_gap"
    summary = {
        "positive_gap_recovery": _mean(rows, "gap_recovery_rate", case=positive_case),
        "false_bridge_rate": _mean(rows, "false_bridge_rate", case=negative_case),
        "negative_gap_filled_pixel_fraction": _mean(rows, "negative_gap_filled_pixel_fraction", case=negative_case),
        "base_visible_dice": _mean(rows, "base_visible_dice"),
        "completion_visible_dice": _mean(rows, "completion_visible_dice"),
        "base_visible_cldice": _mean(rows, "base_visible_cldice"),
        "completion_visible_cldice": _mean(rows, "completion_visible_cldice"),
        "base_latent_cldice": _mean(rows, "base_latent_cldice"),
        "completion_latent_cldice": _mean(rows, "completion_latent_cldice"),
        "base_endpoint_f1": _mean(rows, "base_endpoint_f1"),
        "completion_endpoint_f1": _mean(rows, "completion_endpoint_f1"),
        "modified_pixels_mean": _mean(rows, "modified_pixels"),
    }
    gates = CONFIRM_PROTOCOL["gates"]
    checks = {
        "pair_auroc": pair_metrics["auroc"] >= float(gates["pair_auroc_min"]),
        "positive_gap_recovery": summary["positive_gap_recovery"] >= float(gates["positive_gap_recovery_min"]),
        "false_bridge": summary["false_bridge_rate"] <= float(gates["false_bridge_max"]),
        "visible_dice_safety": summary["base_visible_dice"] - summary["completion_visible_dice"] <= float(gates["visible_dice_loss_max"]),
        "visible_cldice_non_decrease": summary["completion_visible_cldice"] >= summary["base_visible_cldice"],
        "endpoint_f1_improvement": summary["completion_endpoint_f1"] - summary["base_endpoint_f1"] >= float(gates["endpoint_f1_improvement_min"]),
    }
    return {
        "status": "LEARNED_PATH_SYNTHETIC_CONFIRM_PASS" if all(checks.values()) else "LEARNED_PATH_SYNTHETIC_CONFIRM_FAIL",
        "protocol": CONFIRM_PROTOCOL,
        "protocol_sha256": _canonical_hash(CONFIRM_PROTOCOL),
        "frozen_pair_classifier": frozen,
        "pair_metrics": pair_metrics,
        "summary": summary,
        "checks": checks,
        "rows": rows,
        "visual": visual,
        "confirm_samples_opened": 2 * PAIRED_GAP_COUNT,
        "test_v5_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
        "inference_uses_latent_connectivity": False,
        "inference_uses_gap_or_instance_truth": False,
        "candidate_endpoints_from_generator_visible_mask": True,
    }


def _write_figure(output_root: Path, visual: dict[int, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, index in enumerate((0, PAIRED_GAP_COUNT)):
        item = visual[index]
        panels = (
            (np.moveaxis(item["image"], 0, -1), "input"),
            (item["visible"], "visible endpoints"),
            (item["completion"], f"learned score={item['score']:.3f}"),
            (item["latent"], "latent evaluator GT"),
        )
        for axis, (image, title) in zip(axes[row], panels):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
    axes[0, 0].set_ylabel("positive")
    axes[1, 0].set_ylabel("matched negative")
    fig.tight_layout()
    fig.savefig(output_root / "fig_learned_path_confirm.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / "fig_learned_path_confirm.svg", bbox_inches="tight")
    plt.close(fig)


def write_learned_confirm(output_root: Path, *, project_root: Path, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.json"
    receipt_path = output_root / "open_receipt.json"
    current_checkpoint = _sha256(Path(project_root) / "results" / "path_completion" / "pair_classifier" / "checkpoint.pt")
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("frozen_pair_classifier", {}).get("checkpoint_sha256") != current_checkpoint:
            raise PermissionError("confirm result exists but pair checkpoint changed")
        return {**existing, "action": "SKIP_ALREADY_FROZEN"}
    if receipt_path.exists():
        raise PermissionError("confirm stream was opened without a completed immutable result")
    receipt = {
        "status": "OPENED_BEFORE_INFERENCE",
        "protocol_sha256": _canonical_hash(CONFIRM_PROTOCOL),
        "pair_checkpoint_sha256": current_checkpoint,
        "stream": CONFIRM_PROTOCOL["stream"],
        "test_v5": "LOCKED_UNOPENED",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    result = run_learned_confirm(project_root, device=device)
    rows = result.pop("rows")
    visual = result.pop("visual")
    with (output_root / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_figure(output_root, visual)
    result["rows_csv"] = str(output_root / "rows.csv")
    result["open_receipt"] = str(receipt_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Learned pair-gated path confirmation",
        "",
        f"Status: `{result['status']}`",
        "",
        "This confirmation isolates pair selection using generator-visible endpoints. It does not yet measure robustness to model-generated endpoints.",
        "",
        f"- pair AUROC: `{result['pair_metrics']['auroc']:.6f}`",
        f"- positive gap recovery: `{result['summary']['positive_gap_recovery']:.6f}`",
        f"- false bridge rate: `{result['summary']['false_bridge_rate']:.6f}`",
        f"- endpoint F1 change: `{result['summary']['completion_endpoint_f1'] - result['summary']['base_endpoint_f1']:.6f}`",
        "- latent connectivity used by inference: `false`",
        "- v5 test: `LOCKED_UNOPENED`",
        "- CRACKS expert: `FORBIDDEN_NOT_ACCESSED`",
    ]
    (output_root / "LEARNED_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n")
    return result

