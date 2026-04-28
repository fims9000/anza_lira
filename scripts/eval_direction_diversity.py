#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.export_geometry_attention_story as story


def _sample_stats(direction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    theta = direction[mask]
    if theta.size == 0:
        return {
            "num_points": 0.0,
            "direction_resultant_r": 1.0,
            "direction_diversity": 0.0,
            "orientation_resultant_r": 1.0,
            "orientation_diversity": 0.0,
            "hist_nonzero_bins_12": 0.0,
        }
    r_dir = float(np.abs(np.mean(np.exp(1j * theta))))
    r_ori = float(np.abs(np.mean(np.exp(1j * (2.0 * theta)))))
    r_dir = float(min(max(r_dir, 0.0), 1.0))
    r_ori = float(min(max(r_ori, 0.0), 1.0))
    hist, _ = np.histogram(theta, bins=np.linspace(-math.pi, math.pi, 13))
    return {
        "num_points": float(theta.size),
        "direction_resultant_r": r_dir,
        "direction_diversity": float(1.0 - r_dir),
        "orientation_resultant_r": r_ori,
        "orientation_diversity": float(1.0 - r_ori),
        "hist_nonzero_bins_12": float((hist > 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model-native direction diversity from AZ theta_map.")
    p.add_argument("--run", required=True, help="Run directory relative to --results-dir (e.g. results/my_run).")
    p.add_argument("--results-dir", default=".")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    root = Path(args.results_dir).resolve()
    device = torch.device(args.device)

    run_dir = story._find_run(root, args.run, None)
    model, cfg, metrics, _ = story._load_model(run_dir, device)
    dataset, dataset_name = story._build_eval_dataset(cfg)
    thr = float(metrics.get("selected_threshold", cfg.get("seg_threshold", 0.5)))

    rows: list[dict[str, float]] = []
    for idx in range(len(dataset)):
        x_norm, y, valid = dataset[idx]
        x = x_norm.unsqueeze(0).to(device)
        prob = story._predict_prob(model, x)
        pred = (prob >= thr).astype(np.float32)
        _ = model(x)

        az_layers = []
        for name, module in model.named_modules():
            if isinstance(module, story.AZConv2d):
                snap = module.interpretation_snapshot()
                if snap:
                    az_layers.append((name, snap))
        if not az_layers:
            continue
        _, snap = az_layers[-1]
        valid_np = valid[0].numpy().astype(np.float32)
        gt_np = y[0].numpy().astype(np.float32)
        direction, _signed_gain, conf = story._direction_gain_confidence_maps(snap, valid_np)
        mask = ((pred > 0.5) | (gt_np > 0.5)) & (valid_np > 0.5) & (conf > 0.12)
        row = _sample_stats(direction, mask)
        row["sample_index"] = float(idx)
        rows.append(row)

    if not rows:
        raise SystemExit("No valid rows computed.")

    keys = [
        "num_points",
        "direction_resultant_r",
        "direction_diversity",
        "orientation_resultant_r",
        "orientation_diversity",
        "hist_nonzero_bins_12",
    ]
    summary = {
        "run": str(run_dir),
        "dataset": dataset_name,
        "num_samples": len(rows),
        "means": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "mins": {k: float(np.min([r[k] for r in rows])) for k in keys},
        "maxs": {k: float(np.max([r[k] for r in rows])) for k in keys},
    }
    out_path = Path(args.output_json) if args.output_json else (run_dir / "direction_diversity_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
