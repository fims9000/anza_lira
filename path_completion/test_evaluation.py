"""One-time CrossingTraceBench-v5 test evaluation after calibration freeze."""

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
from path_completion.calibration import _canonical_hash
from path_completion.learned_confirm import (
    load_frozen_pair_classifier,
    pair_gated_completion,
    score_pair,
)
from path_completion.pair_classifier import _single_pair
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v5 import generate_authorized_test_sample_v5
from synthetic.evaluation_corrected import evaluate_sample_corrected


TEST_PROTOCOL = {
    "version": "anza_path_classifier_v5_test_v1_frozen",
    "calibration": "frozen validation temperature and threshold",
    "samples": "v5 test positive 0:128 plus matched negative 128:256",
    "candidate_endpoints": "controlled generator-visible support",
    "latent_connectivity_inference": False,
    "gap_truth_inference": False,
    "path": "same train-frozen deterministic endpoint rasterization",
    "gates": {
        "auroc_min": 0.95,
        "recovery_min": 0.75,
        "false_bridge_max": 0.02,
        "visible_dice_unchanged": True,
        "visible_cldice_unchanged": True,
        "endpoint_f1_improvement_min": 0.05,
    },
    "expert": "FORBIDDEN",
    "post_test_retuning": "FORBIDDEN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict[str, Any]], key: str, *, case: str | None = None) -> float:
    values = [float(row[key]) for row in rows if case is None or row["case"] == case]
    return float(np.mean(values))


def run_v5_test(project_root: Path, calibration: dict[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    root = Path(project_root)
    torch_device = torch.device(device)
    model, frozen = load_frozen_pair_classifier(root, torch_device)
    if frozen["checkpoint_sha256"] != calibration.get("classifier_checkpoint_sha256"):
        raise PermissionError("classifier checkpoint differs from calibration freeze")
    temperature = float(calibration["temperature"])
    threshold = float(calibration["selected_operating_point"]["threshold"])
    rows: list[dict[str, Any]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    visual: dict[int, dict[str, Any]] = {}
    for index in range(2 * PAIRED_GAP_COUNT):
        sample = generate_authorized_test_sample_v5(index, calibration_freeze=calibration)
        pair = _single_pair(sample, frozen["d_max_px"])
        raw_probability = score_pair(model, sample, pair, torch_device)
        raw_logit = float(np.log(np.clip(raw_probability, 1e-12, 1 - 1e-12) / np.clip(1 - raw_probability, 1e-12, 1)))
        calibrated_probability = float(1.0 / (1.0 + np.exp(-raw_logit / temperature)))
        completion = pair_gated_completion(
            sample,
            pair,
            score=calibrated_probability,
            threshold=threshold,
            path_radius=frozen["path_radius_px"],
        )
        visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
        base = evaluate_sample_corrected(visible, sample, predicted_completion_mask=visible)["family_a"]
        completed = evaluate_sample_corrected(visible, sample, predicted_completion_mask=completion)["family_a"]
        positive = sample["case"] == "fault_with_gap"
        (positive_scores if positive else negative_scores).append(calibrated_probability)
        rows.append({
            "index": index,
            "pair_id": int(sample["pair_id"]),
            "case": sample["case"],
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "accepted": calibrated_probability >= threshold,
            "gap_recovery_rate": completed["gap_recovery_rate"],
            "false_bridge_rate": completed["false_bridge_rate"],
            "base_visible_dice": base["visible_dice"],
            "completion_visible_dice": completed["visible_dice"],
            "base_visible_cldice": base["visible_cldice"],
            "completion_visible_cldice": completed["visible_cldice"],
            "base_latent_cldice": base["latent_cldice"],
            "completion_latent_cldice": completed["latent_cldice"],
            "base_endpoint_f1": base["endpoint_f1"],
            "completion_endpoint_f1": completed["endpoint_f1"],
            "modified_pixels": int(np.logical_xor(completion, visible).sum()),
        })
        if index in {0, PAIRED_GAP_COUNT}:
            visual[index] = {
                "image": np.asarray(sample["image"]),
                "visible": visible,
                "completion": completion,
                "latent": np.asarray(sample["latent_fault_mask"], dtype=bool),
                "score": calibrated_probability,
            }
    pair_metrics = balanced_matched_pair_metrics(
        np.asarray(positive_scores), np.asarray(negative_scores), threshold=threshold
    )
    summary = {
        "positive_gap_recovery": _mean(rows, "gap_recovery_rate", case="fault_with_gap"),
        "false_bridge_rate": _mean(rows, "false_bridge_rate", case="negative_gap"),
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
    gates = TEST_PROTOCOL["gates"]
    checks = {
        "auroc": pair_metrics["auroc"] >= float(gates["auroc_min"]),
        "recovery": summary["positive_gap_recovery"] >= float(gates["recovery_min"]),
        "false_bridge": summary["false_bridge_rate"] <= float(gates["false_bridge_max"]),
        "visible_dice_unchanged": summary["completion_visible_dice"] == summary["base_visible_dice"],
        "visible_cldice_unchanged": summary["completion_visible_cldice"] == summary["base_visible_cldice"],
        "endpoint_f1_improvement": summary["completion_endpoint_f1"] - summary["base_endpoint_f1"] >= float(gates["endpoint_f1_improvement_min"]),
    }
    return {
        "status": "PATH_CLASSIFIER_TEST_PASS" if all(checks.values()) else "PATH_CLASSIFIER_TEST_FAIL",
        "protocol": TEST_PROTOCOL,
        "protocol_sha256": _canonical_hash(TEST_PROTOCOL),
        "calibration_freeze_sha256": calibration["freeze_sha256"],
        "classifier_checkpoint_sha256": frozen["checkpoint_sha256"],
        "temperature": temperature,
        "threshold": threshold,
        "pair_metrics": pair_metrics,
        "summary": summary,
        "checks": checks,
        "rows": rows,
        "visual": visual,
        "v5_test_samples_opened": 2 * PAIRED_GAP_COUNT,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
        "post_test_retuning": "FORBIDDEN_NOT_PERFORMED",
    }


def _write_figure(output: Path, visual: dict[int, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for row, index in enumerate((0, PAIRED_GAP_COUNT)):
        item = visual[index]
        for axis, (image, title) in zip(axes[row], (
            (np.moveaxis(item["image"], 0, -1), "input"),
            (item["visible"], "visible endpoints"),
            (item["completion"], f"completion score={item['score']:.3f}"),
            (item["latent"], "latent evaluator GT"),
        )):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
    fig.savefig(output / "fig_v5_test_completion.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "fig_v5_test_completion.svg", bbox_inches="tight")
    plt.close(fig)


def write_v5_test(output_root: Path, *, project_root: Path, device: str = "cuda") -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "test_result.json"
    receipt_path = output / "test_open_receipt.json"
    calibration_path = Path(project_root) / "results/final_practical_cycle/path_calibration/calibration_freeze.json"
    calibration = json.loads(calibration_path.read_text())
    core = {key: value for key, value in calibration.items() if key != "freeze_sha256"}
    if calibration.get("freeze_sha256") != _canonical_hash(core):
        raise PermissionError("calibration freeze hash invalid before test open")
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("calibration_freeze_sha256") != calibration["freeze_sha256"]:
            raise PermissionError("test result calibration drift")
        return {**existing, "action": "SKIP_ALREADY_FROZEN"}
    if receipt_path.exists():
        raise PermissionError("v5 test was opened without a completed result")
    receipt = {
        "status": "V5_TEST_OPENED_BEFORE_INFERENCE",
        "calibration_freeze_sha256": calibration["freeze_sha256"],
        "classifier_checkpoint_sha256": calibration["classifier_checkpoint_sha256"],
        "protocol_sha256": _canonical_hash(TEST_PROTOCOL),
        "expert_data_accessed": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    result = run_v5_test(project_root, calibration, device=device)
    rows = result.pop("rows")
    visual = result.pop("visual")
    rows_path = output / "test_scores.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_figure(output, visual)
    result["test_scores_csv"] = str(rows_path)
    result["test_scores_sha256"] = _sha256(rows_path)
    result["test_open_receipt"] = str(receipt_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

