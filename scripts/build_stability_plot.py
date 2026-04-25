#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_label(variant: str) -> str:
    if variant == "baseline":
        return "Baseline U-Net"
    if variant == "attention_unet":
        return "Attention U-Net"
    if variant == "az_thesis":
        return "Proposed AZ-based method"
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stability plot (Dice distribution + quality/latency scatter).")
    parser.add_argument("--session-dir", required=True, type=str, help="Directory with all_metrics.json")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    session_dir = (PROJECT_ROOT / args.session_dir).resolve()
    metrics_path = session_dir / "all_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing all_metrics.json: {metrics_path}")
    rows = _load_json(metrics_path)
    if not isinstance(rows, list) or not rows:
        raise SystemExit("all_metrics.json is empty or invalid.")

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant", "unknown"))
        by_variant[variant].append(row)

    variants = sorted(by_variant.keys(), key=lambda x: (x != "baseline", x))
    labels = [_variant_label(v) for v in variants]
    dice_sets = [[float(item.get("test_dice", 0.0)) for item in by_variant[v]] for v in variants]
    iou_means = [float(np.mean([float(item.get("test_iou", 0.0)) for item in by_variant[v]])) for v in variants]
    dice_means = [float(np.mean([float(item.get("test_dice", 0.0)) for item in by_variant[v]])) for v in variants]
    fwd_means = [float(np.mean([float(item.get("seconds_per_forward_batch", 0.0)) for item in by_variant[v]])) for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax0, ax1 = axes

    bp = ax0.boxplot(dice_sets, labels=labels, patch_artist=True, showmeans=True)
    palette = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A", "#6DC8EC", "#9270CA", "#FF9D4D"]
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.65)
    ax0.set_title("Dice Stability Across Seeds")
    ax0.set_ylabel("Dice")
    ax0.grid(axis="y", alpha=0.25)
    ax0.set_ylim(bottom=max(0.0, min(min(vals) for vals in dice_sets) - 0.05), top=min(1.0, max(max(vals) for vals in dice_sets) + 0.05))
    ax0.tick_params(axis="x", rotation=15)

    x = np.asarray(fwd_means, dtype=np.float32)
    y = np.asarray(dice_means, dtype=np.float32)
    sizes = 240.0 + 800.0 * np.asarray(iou_means, dtype=np.float32)
    ax1.scatter(x, y, s=sizes, c=np.arange(len(variants)), cmap="tab10", alpha=0.85, edgecolors="black", linewidths=0.6)
    for i, label in enumerate(labels):
        ax1.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax1.set_title("Quality vs Inference Cost")
    ax1.set_xlabel("Seconds per forward batch (mean)")
    ax1.set_ylabel("Dice (mean)")
    ax1.grid(alpha=0.25)

    fig.suptitle(f"Stability Summary: {session_dir.name}", fontsize=12)
    fig.tight_layout()

    output = (PROJECT_ROOT / args.output).resolve() if args.output else (session_dir / "stability_plot.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()

