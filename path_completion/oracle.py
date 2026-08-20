"""Frozen O0/O1/O2 mathematical oracles for max-min path completion."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from connectivity_repair.balanced_metrics import balanced_matched_pair_metrics
from models.azconv_affinity import LOCAL8_OFFSETS, _shift_tensor
from path_completion.maxmin import maxmin_closure_torch
from path_completion.widest_path import candidate_endpoint_pairs, rasterize_path, widest_path
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v5 import benchmark_v5_config, generate_sample_v5
from synthetic.evaluation_corrected import evaluate_sample_corrected


PATH_PROTOCOL = {
    "version": "anza_maxmin_widest_path_oracle_v1",
    "benchmark": benchmark_v5_config()["sha256"],
    "train_geometry_indices": "train_v5[0:128] positive gaps only",
    "oracle_indices": "validation_v5[0:128] positive plus validation_v5[128:256] matched negative",
    "d_min_px": 3.0,
    "d_max_rule": "ceil train 95th percentile extracted positive gap endpoint distance",
    "path_radius_rule": "minimum radius 0:5 with every train positive gap coverage >=0.99",
    "path_threshold": 0.5,
    "old_restart_control": {"steps": 8, "alpha": 0.8, "threshold": 0.5},
    "o0_gate": {
        "positive_gap_recovery_min": 0.99,
        "negative_gap_filled_fraction_max": 0.01,
        "visible_foreground_preserved_exactly": True,
    },
    "o1_gate": {
        "pair_auroc": 1.0,
        "positive_gap_recovery_min": 0.95,
        "false_bridge_max": 0.02,
        "visible_dice_loss_max": 0.001,
    },
    "test_v5": "LOCKED_UNOPENED",
    "expert": "FORBIDDEN",
    "cracks": "NOT_ACCESSED_BY_ORACLE",
}


def _relation(sample: Mapping[str, Any]) -> np.ndarray:
    return build_affinity_targets(sample, LOCAL8_OFFSETS)["affinity_positive"].astype(np.float32)


def freeze_train_geometry() -> dict[str, Any]:
    endpoint_distances: list[float] = []
    train_records: list[tuple[dict[str, Any], np.ndarray, tuple[tuple[int, int], ...]]] = []
    for index in range(PAIRED_GAP_COUNT):
        sample = generate_sample_v5("train", index)
        pairs = candidate_endpoint_pairs(
            sample["visible_fault_mask"], d_min=float(PATH_PROTOCOL["d_min_px"]), d_max=128.0
        )
        gap_xy = np.asarray(sample["gaps"][0]["endpoint_xy"], dtype=float)[:, ::-1]
        ranked = sorted(
            pairs,
            key=lambda pair: min(
                math.dist(pair.first, gap_xy[0]) + math.dist(pair.second, gap_xy[1]),
                math.dist(pair.first, gap_xy[1]) + math.dist(pair.second, gap_xy[0]),
            ),
        )
        if not ranked:
            raise ValueError("train positive gap has no endpoint pair")
        pair = ranked[0]
        endpoint_distances.append(pair.distance)
        score, path = widest_path(_relation(sample), pair.first, pair.second)
        if score != 1.0 or not path:
            raise ValueError("perfect train relation did not connect a positive gap")
        train_records.append((sample, _relation(sample), path))
    d_max = int(math.ceil(float(np.quantile(endpoint_distances, 0.95))))
    selected_radius = None
    radius_min_coverages: dict[str, float] = {}
    for radius in range(6):
        coverages = []
        for sample, relation, path in train_records:
            node_support = relation.any(axis=0)
            path_mask = rasterize_path(path, sample["visible_fault_mask"].shape, radius=radius) & node_support
            gap = np.asarray(sample["positive_gap_mask"], dtype=bool)
            coverages.append(float(path_mask[gap].mean()))
        radius_min_coverages[str(radius)] = float(min(coverages))
        if min(coverages) >= 0.99:
            selected_radius = radius
            break
    if selected_radius is None:
        raise ValueError("no train-frozen path raster radius reaches 0.99 gap coverage")
    return {
        "d_max_px": d_max,
        "train_endpoint_distance_q95": float(np.quantile(endpoint_distances, 0.95)),
        "train_endpoint_distance_min": float(min(endpoint_distances)),
        "train_endpoint_distance_max": float(max(endpoint_distances)),
        "path_radius_px": int(selected_radius),
        "radius_min_positive_gap_coverage": radius_min_coverages,
        "train_sample_count": PAIRED_GAP_COUNT,
        "validation_accessed_for_freeze": False,
    }


def restarted_average(
    seed: torch.Tensor,
    relation: torch.Tensor,
    *,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    """Old algebraic control with the same seed and perfect relation."""

    if seed.ndim == 3:
        seed = seed.unsqueeze(1)
    denominator = relation.sum(dim=1, keepdim=True)
    transition = torch.where(denominator > 0, relation / denominator.clamp_min(1.0), torch.zeros_like(relation))
    state = seed.clone()
    for _ in range(int(steps)):
        propagated = torch.zeros_like(state)
        for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
            neighbor, _ = _shift_tensor(state, dx, dy)
            propagated = propagated + transition[:, channel : channel + 1] * neighbor
        propagated = torch.where(denominator > 0, propagated, state)
        state = (1.0 - float(alpha)) * seed + float(alpha) * propagated
    return state


def _family_metrics(sample: Mapping[str, Any], completion: np.ndarray) -> dict[str, Any]:
    visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
    return evaluate_sample_corrected(
        visible,
        sample,
        predicted_completion_mask=np.asarray(completion, dtype=bool),
    )["family_a"]


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    positive = [row for row in selected if row["case"] == "fault_with_gap"]
    negative = [row for row in selected if row["case"] == "negative_gap"]
    return {
        "method": method,
        "sample_count": len(selected),
        "positive_gap_recovery": float(np.mean([row["gap_recovery_rate"] for row in positive])),
        "negative_gap_filled_pixel_fraction": float(np.mean([row["negative_gap_filled_pixel_fraction"] for row in negative])),
        "false_bridge_rate": float(np.mean([row["false_bridge_rate"] for row in negative])),
        "visible_dice": float(np.mean([row["visible_dice"] for row in selected])),
        "visible_cldice": float(np.mean([row["visible_cldice"] for row in selected])),
        "modified_pixels_mean": float(np.mean([row["modified_pixels"] for row in selected])),
    }


def run_oracles(*, device: str = "cuda") -> dict[str, Any]:
    frozen = freeze_train_geometry()
    samples = [generate_sample_v5("validation", index) for index in range(2 * PAIRED_GAP_COUNT)]
    rows: list[dict[str, Any]] = []
    pair_scores_positive: list[float] = []
    pair_scores_negative: list[float] = []
    visual: dict[int, dict[str, np.ndarray | float]] = {}
    torch_device = torch.device(device)
    batch_size = 16
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        seeds = torch.stack([
            torch.as_tensor(sample["visible_fault_mask"], dtype=torch.float32)
            for sample in batch_samples
        ]).to(torch_device)
        relations = torch.stack([torch.as_tensor(_relation(sample)) for sample in batch_samples]).to(torch_device)
        maxmin, iterations = maxmin_closure_torch(seeds, relations)
        old = restarted_average(
            seeds,
            relations,
            steps=int(PATH_PROTOCOL["old_restart_control"]["steps"]),
            alpha=float(PATH_PROTOCOL["old_restart_control"]["alpha"]),
        )
        for local, sample in enumerate(batch_samples):
            index = start + local
            base = np.asarray(sample["visible_fault_mask"], dtype=bool)
            maxmin_mask = maxmin[local, 0].cpu().numpy() >= 0.5
            old_mask = old[local, 0].cpu().numpy() >= float(PATH_PROTOCOL["old_restart_control"]["threshold"])
            pairs = candidate_endpoint_pairs(
                base,
                d_min=float(PATH_PROTOCOL["d_min_px"]),
                d_max=float(frozen["d_max_px"]),
            )
            if len(pairs) != 1:
                raise ValueError(f"oracle sample {index} has {len(pairs)} candidate endpoint pairs, expected one")
            pair = pairs[0]
            relation = relations[local].cpu().numpy()
            score, path = widest_path(relation, pair.first, pair.second)
            is_positive = sample["case"] == "fault_with_gap"
            (pair_scores_positive if is_positive else pair_scores_negative).append(score)
            path_completion = base.copy()
            if score >= float(PATH_PROTOCOL["path_threshold"]):
                node_support = relation.any(axis=0)
                path_completion |= rasterize_path(
                    path, base.shape, radius=int(frozen["path_radius_px"])
                ) & node_support
            methods = {
                "restart_average": old_mask,
                "maxmin_closure": maxmin_mask,
                "widest_path": path_completion,
            }
            for method, completion in methods.items():
                metrics = _family_metrics(sample, completion)
                negative_gap = np.asarray(sample["negative_gap_mask"], dtype=bool)
                rows.append({
                    "index": index,
                    "pair_id": sample["pair_id"],
                    "case": sample["case"],
                    "method": method,
                    "visible_dice": metrics["visible_dice"],
                    "visible_cldice": metrics["visible_cldice"],
                    "gap_recovery_rate": metrics["gap_recovery_rate"],
                    "false_bridge_rate": metrics["false_bridge_rate"],
                    "negative_gap_filled_pixel_fraction": float(completion[negative_gap].mean()) if negative_gap.any() else 0.0,
                    "visible_foreground_preserved": bool(np.all(completion[base])),
                    "modified_pixels": int(np.logical_xor(completion, base).sum()),
                    "maxmin_iterations": iterations if method == "maxmin_closure" else None,
                    "path_score": score if method == "widest_path" else None,
                })
            if index in {0, PAIRED_GAP_COUNT}:
                visual[index] = {
                    "image": np.asarray(sample["image"]),
                    "visible": base,
                    "latent": np.asarray(sample["latent_fault_mask"], dtype=bool),
                    "restart": old_mask,
                    "maxmin": maxmin_mask,
                    "path": path_completion,
                    "path_score": float(score),
                }
    comparison = [_aggregate(rows, method) for method in ("restart_average", "maxmin_closure", "widest_path")]
    pair_metrics = balanced_matched_pair_metrics(
        np.asarray(pair_scores_positive), np.asarray(pair_scores_negative), threshold=float(PATH_PROTOCOL["path_threshold"])
    )
    mm = next(row for row in comparison if row["method"] == "maxmin_closure")
    path = next(row for row in comparison if row["method"] == "widest_path")
    o0_checks = {
        "positive_gap_recovery": mm["positive_gap_recovery"] >= 0.99,
        "negative_gap_filled_fraction": mm["negative_gap_filled_pixel_fraction"] <= 0.01,
        "visible_preserved_exactly": all(row["visible_foreground_preserved"] for row in rows if row["method"] == "maxmin_closure"),
    }
    o1_checks = {
        "pair_auroc": pair_metrics["auroc"] == 1.0,
        "positive_gap_recovery": path["positive_gap_recovery"] >= 0.95,
        "false_bridge": path["false_bridge_rate"] <= 0.02,
        "visible_dice_safety": 1.0 - path["visible_dice"] <= 0.001,
    }
    return {
        "status": "MAXMIN_PATH_ORACLE_PASS" if all(o0_checks.values()) and all(o1_checks.values()) else "MAXMIN_PATH_NEGATIVE_WITH_ROOT_CAUSE",
        "protocol": PATH_PROTOCOL,
        "train_frozen_geometry": frozen,
        "o0_checks": o0_checks,
        "o1_checks": o1_checks,
        "pair_metrics": pair_metrics,
        "comparison": comparison,
        "rows": rows,
        "visual": visual,
        "test_v5_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }


def _write_figures(output_root: Path, visual: dict[int, dict[str, Any]]) -> None:
    positive = visual[0]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    panels = [
        (positive["visible"], "Visible seed"),
        (positive["restart"], "Restart average"),
        (positive["maxmin"], "Max-min closure"),
        (positive["path"], "Widest path"),
        (positive["latent"], "Latent oracle GT"),
    ]
    for axis, (array, title) in zip(axes, panels):
        axis.imshow(array, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_root / "fig_restart_vs_maxmin.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / "fig_restart_vs_maxmin.svg", bbox_inches="tight")
    plt.close(fig)

    negative = visual[PAIRED_GAP_COUNT]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, item, label in ((0, positive, "positive"), (1, negative, "matched negative")):
        image = np.moveaxis(np.asarray(item["image"]), 0, -1)
        axes[row, 0].imshow(np.clip(image, 0, 1))
        axes[row, 0].set_title(f"{label} input")
        axes[row, 1].imshow(item["visible"], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("visible seed")
        axes[row, 2].imshow(item["path"], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"path score={item['path_score']:.1f}")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_root / "fig_positive_negative_pair.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / "fig_positive_negative_pair.svg", bbox_inches="tight")
    plt.close(fig)


def write_oracles(output_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_oracles(device=device)
    serializable = {key: value for key, value in result.items() if key not in {"rows", "visual"}}
    (output_root / "oracle_summary.json").write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    (output_root / "ORACLE_PROTOCOL.json").write_text(json.dumps(PATH_PROTOCOL, indent=2, sort_keys=True) + "\n")
    with (output_root / "oracle_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    with (output_root / "oracle_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["comparison"][0]))
        writer.writeheader()
        writer.writerows(result["comparison"])
    _write_figures(output_root, result["visual"])
    report = [
        "# Max-min and widest-path mathematical oracle",
        "",
        f"Status: `{result['status']}`",
        "",
        "All rows use the same 128 positive and 128 matched-negative validation samples. Max-min and widest-path receive perfect latent-lineage connectivity; this is a feasibility result, not a learned-model result.",
        "",
        "| Method | gap recovery | negative filled fraction | false bridge | visible Dice | visible clDice | modified pixels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparison"]:
        report.append(
            f"| {row['method']} | {row['positive_gap_recovery']:.4f} | "
            f"{row['negative_gap_filled_pixel_fraction']:.4f} | {row['false_bridge_rate']:.4f} | "
            f"{row['visible_dice']:.4f} | {row['visible_cldice']:.4f} | {row['modified_pixels_mean']:.2f} |"
        )
    report.extend([
        "",
        f"Endpoint-pair AUROC: `{result['pair_metrics']['auroc']:.4f}`; balanced AUPRC: `{result['pair_metrics']['balanced_auprc']:.4f}`.",
        f"Train-only freeze: d_max=`{result['train_frozen_geometry']['d_max_px']}` px, path radius=`{result['train_frozen_geometry']['path_radius_px']}` px.",
    ])
    (output_root / "ORACLE_REPORT.md").write_text("\n".join(report) + "\n")
    return serializable


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = write_oracles(
        root / "results" / "path_completion" / "oracle",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps(result, indent=2, sort_keys=True))

