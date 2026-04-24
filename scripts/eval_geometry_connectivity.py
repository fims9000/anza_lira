#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from utils import build_model


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload at {path} must be a mapping.")
    return payload


def _evaluate_run(run_dir: Path, device: torch.device, topology_iters: int = 10) -> Dict[str, float]:
    checkpoint_path = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    if not checkpoint_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Run directory is missing checkpoint or metrics: {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = _load_checkpoint(checkpoint_path)
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain config payload.")

    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    model = build_model(
        payload.get("variant", metrics.get("variant", "baseline")),
        num_outputs=num_outputs,
        in_channels=in_channels,
        num_rules=int(cfg.get("num_rules", 4)),
        task=utils.task_for_dataset(cfg["dataset"]),
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    _train_loader, _val_loader, test_loader, _in_c, _num_out, task = utils.build_dataloaders(cfg)
    if task != "segmentation":
        raise ValueError(f"Run is not segmentation: {run_dir}")

    threshold = float(metrics.get("selected_threshold", cfg.get("seg_threshold", 0.5)))
    topo_iters = int(cfg.get("topology_num_iters", topology_iters))
    eps = 1e-8

    skel_pred_target_overlap = 0.0
    skel_pred_sum = 0.0
    skel_target_pred_overlap = 0.0
    skel_target_sum = 0.0

    with torch.no_grad():
        for x, y, valid_mask in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            output = model(x)
            logits, _, _ = utils.unpack_segmentation_outputs(output)
            pred = (torch.sigmoid(logits) >= threshold).float() * valid_mask
            target = y * valid_mask

            skel_pred = utils._soft_skeletonize(pred, num_iters=topo_iters)
            skel_target = utils._soft_skeletonize(target, num_iters=topo_iters)

            skel_pred_target_overlap += float((skel_pred * target).sum().item())
            skel_pred_sum += float(skel_pred.sum().item())
            skel_target_pred_overlap += float((skel_target * pred).sum().item())
            skel_target_sum += float(skel_target.sum().item())

    skeleton_precision = skel_pred_target_overlap / (skel_pred_sum + eps)
    skeleton_recall = skel_target_pred_overlap / (skel_target_sum + eps)
    cldice = (2.0 * skeleton_precision * skeleton_recall + eps) / (skeleton_precision + skeleton_recall + eps)

    return {
        "test_cldice": float(cldice),
        "test_skeleton_precision": float(skeleton_precision),
        "test_skeleton_recall": float(skeleton_recall),
        "test_geometry_connectivity_mean": float((cldice + skeleton_precision + skeleton_recall) / 3.0),
        "connectivity_threshold": float(threshold),
    }


def _format_mean_std(values: List[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f} +- 0.0000"
    return f"{statistics.mean(values):.4f} +- {statistics.stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute geometry-aware connectivity metrics (clDice family) for segmentation runs.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with run subdirectories containing checkpoint_best.pt and metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter, e.g. arcade_syntax.")
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    device = torch.device(args.device)
    dataset_filter = utils.canonical_dataset_name(args.dataset) if args.dataset else None

    per_run_rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "checkpoint_best.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("task") != "segmentation":
            continue
        dataset_name = utils.canonical_dataset_name(str(metrics.get("dataset", "")))
        if dataset_filter and dataset_name != dataset_filter:
            continue

        connectivity_metrics = _evaluate_run(run_dir, device=device)
        metrics.update(connectivity_metrics)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        per_run_rows.append(
            {
                "run_name": run_dir.name,
                "variant": str(metrics.get("variant", "unknown")),
                "seed": metrics.get("seed"),
                **connectivity_metrics,
            }
        )
        print(
            f"[{run_dir.name}] clDice={connectivity_metrics['test_cldice']:.4f} "
            f"sPrec={connectivity_metrics['test_skeleton_precision']:.4f} "
            f"sRec={connectivity_metrics['test_skeleton_recall']:.4f}"
        )

    if not per_run_rows:
        raise SystemExit("No eligible runs found for connectivity evaluation.")

    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_run_rows:
        by_variant.setdefault(row["variant"], []).append(row)

    lines = [
        "| Variant | Runs | clDice mean+-std | Skeleton Precision mean+-std | Skeleton Recall mean+-std | Geometry Connectivity mean+-std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted(by_variant):
        rows = by_variant[variant]
        cl = _format_mean_std([float(r["test_cldice"]) for r in rows])
        sp = _format_mean_std([float(r["test_skeleton_precision"]) for r in rows])
        sr = _format_mean_std([float(r["test_skeleton_recall"]) for r in rows])
        gc = _format_mean_std([float(r["test_geometry_connectivity_mean"]) for r in rows])
        lines.append(f"| {variant} | {len(rows)} | {cl} | {sp} | {sr} | {gc} |")

    summary_path = results_dir / "geometry_connectivity_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote geometry-aware connectivity summary: {summary_path}")


if __name__ == "__main__":
    main()
