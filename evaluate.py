#!/usr/bin/env python3
"""Load a checkpoint and report metrics + approximate inference time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from utils import (
    binary_confusion_counts,
    build_dataloaders,
    build_model,
    count_parameters,
    estimate_drive_pos_weight,
    load_config,
    measure_inference_time,
    segmentation_objective,
    segmentation_metrics_from_counts,
    set_seed,
    spatial_shape_for_dataset,
)


@torch.no_grad()
def _classification_eval(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    n = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return {
        "test_loss": loss_sum / max(n, 1),
        "test_accuracy": 100.0 * correct / max(n, 1),
    }


@torch.no_grad()
def _segmentation_eval(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    threshold: float,
    pos_weight: torch.Tensor | None,
) -> Dict[str, float]:
    loss_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    n = 0
    tp = fp = tn = fn = 0.0

    for x, y, valid_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        logits = model(x)
        loss, aux, main_logits = segmentation_objective(
            logits,
            y,
            valid_mask,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            aux_weight=0.0,
            boundary_weight=0.0,
        )
        loss_sum += loss.item() * x.size(0)
        bce_sum += aux["bce_loss"] * x.size(0)
        dice_loss_sum += aux["dice_loss"] * x.size(0)
        c_tp, c_fp, c_tn, c_fn = binary_confusion_counts(main_logits, y, valid_mask, threshold=threshold)
        tp += c_tp
        fp += c_fp
        tn += c_tn
        fn += c_fn
        n += x.size(0)

    metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
    return {
        "test_loss": loss_sum / max(n, 1),
        "test_bce_loss": bce_sum / max(n, 1),
        "test_dice_loss": dice_loss_sum / max(n, 1),
        "test_dice": metrics["dice"],
        "test_iou": metrics["iou"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_specificity": metrics["specificity"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
    }


@torch.no_grad()
def run_eval(checkpoint_path: Path, cfg_path: str | None, device: torch.device | None = None) -> dict:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    variant = ckpt.get("variant", "baseline")
    cfg = ckpt.get("cfg")
    if cfg is None and cfg_path:
        cfg = load_config(cfg_path)
    if cfg is None:
        raise ValueError("Checkpoint has no cfg embedded; pass --config")

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, test_loader, in_c, num_outputs, task = build_dataloaders(cfg)
    model = build_model(
        variant,
        num_outputs=num_outputs,
        in_channels=in_c,
        num_rules=int(cfg.get("num_rules", 4)),
        task=task,
    )
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()

    t0 = time.perf_counter()
    if task == "classification":
        metrics = _classification_eval(model, test_loader, dev)
    else:
        raw_pos_weight = cfg.get("bce_pos_weight", "auto")
        if raw_pos_weight == "auto":
            pos_weight_value = estimate_drive_pos_weight(
                train_loader.dataset,
                min_weight=float(cfg.get("bce_pos_weight_min", 1.0)),
                max_weight=float(cfg.get("bce_pos_weight_max", 25.0)),
            )
        elif raw_pos_weight is None:
            pos_weight_value = 1.0
        else:
            pos_weight_value = float(raw_pos_weight)
        metrics = _segmentation_eval(
            model,
            test_loader,
            dev,
            bce_weight=float(cfg.get("bce_weight", 1.0)),
            dice_weight=float(cfg.get("dice_weight", 1.0)),
            threshold=float(cfg.get("seg_threshold", 0.5)),
            pos_weight=torch.tensor(pos_weight_value, device=dev),
        )
        metrics["bce_pos_weight"] = pos_weight_value
    wall = time.perf_counter() - t0

    sec_batch = measure_inference_time(
        model,
        dev,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial_shape=spatial_shape_for_dataset(cfg["dataset"]),
    )

    metrics.update(
        {
            "variant": variant,
            "task": task,
            "test_wall_seconds": wall,
            "seconds_per_forward_batch": sec_batch,
            "num_parameters": count_parameters(model),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Fallback if checkpoint has no cfg")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    dev = torch.device(args.device) if args.device else None
    metrics = run_eval(path, args.config, device=dev)
    print(json.dumps(metrics, indent=2))

    out_path = path.parent / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
