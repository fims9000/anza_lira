#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.export_geometry_attention_story as story


def _score_delta(
    dataset: Any,
    az_model: torch.nn.Module,
    az_thr: float,
    baseline_model: torch.nn.Module,
    baseline_thr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx in range(len(dataset)):
        x_norm, y, valid = dataset[idx]
        x = x_norm.unsqueeze(0).to(device)
        gt = y[0].numpy().astype(np.float32)
        v = valid[0].numpy().astype(np.float32)
        base_prob = story._predict_prob(baseline_model, x)
        az_prob = story._predict_prob(az_model, x)
        base_pred = (base_prob >= baseline_thr).astype(np.float32)
        az_pred = (az_prob >= az_thr).astype(np.float32)
        base_dice = float(story._dice(base_pred, gt, v))
        az_dice = float(story._dice(az_pred, gt, v))
        rows.append(
            {
                "index": float(idx),
                "baseline_dice": base_dice,
                "az_dice": az_dice,
                "delta_dice": az_dice - base_dice,
            }
        )
    return rows


def _pick_triplet(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("No rows to rank.")
    by_delta = sorted(rows, key=lambda r: float(r["delta_dice"]))
    fail_row = by_delta[0]
    win_row = by_delta[-1]

    # Parity sample: closest to zero delta but not exactly the same index as fail/win if possible.
    candidates = sorted(rows, key=lambda r: (abs(float(r["delta_dice"])), -float(r["az_dice"])))
    parity_row = candidates[0]
    used = {int(fail_row["index"]), int(win_row["index"])}
    for cand in candidates:
        if int(cand["index"]) not in used:
            parity_row = cand
            break

    return {"win": win_row, "parity": parity_row, "fail": fail_row}


def _run_story_export(
    results_dir: Path,
    az_run_name: str,
    baseline_run_name: str,
    sample_index: int,
    max_layers: int,
    output_dir: Path,
    device: str,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_geometry_attention_story.py"),
        "--results-dir",
        str(results_dir),
        "--run",
        az_run_name,
        "--baseline-run",
        baseline_run_name,
        "--sample-index",
        str(int(sample_index)),
        "--max-layers",
        str(int(max_layers)),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-export win/parity/fail geometry stories against baseline.")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--run", required=True, type=str, help="AZ run directory name.")
    parser.add_argument("--baseline-run", required=True, type=str, help="Baseline run directory name.")
    parser.add_argument("--output-dir", type=str, default="article_assets/final_figures/drive_triplet_story")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    az_run_dir = story._find_run(results_dir, args.run, None)
    baseline_run_dir = story._find_run(results_dir, args.baseline_run, None)

    az_model, az_cfg, az_metrics, _az_variant = story._load_model(az_run_dir, device)
    baseline_model, _base_cfg, baseline_metrics, _ = story._load_model(baseline_run_dir, device)
    dataset, _dataset_name = story._build_eval_dataset(az_cfg)

    az_thr = float(az_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))
    baseline_thr = float(baseline_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))

    rows = _score_delta(dataset, az_model, az_thr, baseline_model, baseline_thr, device)
    triplet = _pick_triplet(rows)

    for label in ("win", "parity", "fail"):
        idx = int(triplet[label]["index"])
        _run_story_export(
            results_dir=results_dir,
            az_run_name=az_run_dir.name,
            baseline_run_name=baseline_run_dir.name,
            sample_index=idx,
            max_layers=int(args.max_layers),
            output_dir=output_dir / label,
            device=args.device,
        )

    summary = {
        "results_dir": str(results_dir),
        "az_run": az_run_dir.name,
        "baseline_run": baseline_run_dir.name,
        "az_threshold": az_thr,
        "baseline_threshold": baseline_thr,
        "selected_samples": triplet,
    }
    (output_dir / "triplet_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

