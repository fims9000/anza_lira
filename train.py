#!/usr/bin/env python3
"""Train baseline or AZ variants on classification or DRIVE segmentation tasks."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from models.azconv import AZConv2d
from utils import (
    az_regularization_weights,
    binary_confusion_counts,
    build_threshold_grid,
    build_drive_threshold_search_report,
    build_dataloaders,
    build_model,
    compare_drive_metrics_to_baseline,
    ensure_dir,
    estimate_drive_pos_weight,
    format_drive_threshold_search_report,
    format_drive_superiority_report,
    load_config,
    measure_inference_time,
    save_json,
    select_best_threshold,
    select_best_drive_record,
    segmentation_metrics_from_counts,
    segmentation_objective,
    set_seed,
    skeleton_confusion_counts,
    skeleton_metrics_from_counts,
    spatial_shape_for_dataset,
    threshold_metric_value,
    sweep_segmentation_thresholds,
    update_drive_comparison_summary,
)

REG_LOG_KEYS = [
    "membership_entropy",
    "membership_entropy_deficit",
    "membership_smoothness",
    "geometry_smoothness",
    "hyperbolicity_penalty",
    "anisotropy_gap",
    "direction_collapse",
    "hybrid_mix_target",
]


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _scheduled_retinal_bias(
    cfg: Dict[str, Any],
    epoch: int,
    epochs: int,
    start_key: str,
    end_key: str,
    schedule_key: str,
    fallback_start: float,
) -> float:
    schedule_name = str(cfg.get(schedule_key, "constant")).lower().strip()
    start = float(cfg.get(start_key, fallback_start))
    end = float(cfg.get(end_key, start))
    if schedule_name in {"", "constant", "none"}:
        return start
    if schedule_name == "linear":
        progress = 1.0 if epochs <= 1 else float(epoch - 1) / float(max(epochs - 1, 1))
        return start + (end - start) * progress
    raise ValueError(f"Unknown {schedule_key}: {schedule_name}")


def apply_retinal_foreground_bias_schedule(
    train_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    epoch: int,
    epochs: int,
) -> float | None:
    dataset_name = utils.canonical_dataset_name(str(cfg.get("dataset", "")))
    if dataset_name in utils.GIS_SEG_DATASETS:
        start_key = "gis_foreground_bias"
        end_key = "gis_foreground_bias_end"
        schedule_key = "gis_foreground_bias_schedule"
        fallback_start = float(cfg.get("road_foreground_bias", 0.0))
    else:
        start_key = "retinal_foreground_bias"
        end_key = "retinal_foreground_bias_end"
        schedule_key = "retinal_foreground_bias_schedule"
        fallback_start = float(cfg.get("drive_foreground_bias", 0.0))

    target = _scheduled_retinal_bias(
        cfg,
        epoch,
        epochs,
        start_key=start_key,
        end_key=end_key,
        schedule_key=schedule_key,
        fallback_start=fallback_start,
    )
    base_dataset = _unwrap_dataset(train_loader.dataset)
    if hasattr(base_dataset, "set_foreground_bias"):
        base_dataset.set_foreground_bias(target)
        return float(base_dataset.foreground_bias)
    if hasattr(base_dataset, "foreground_bias"):
        base_dataset.foreground_bias = float(target)
        return float(base_dataset.foreground_bias)
    return None


def apply_retinal_thin_vessel_bias_schedule(
    train_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    epoch: int,
    epochs: int,
) -> float | None:
    target = _scheduled_retinal_bias(
        cfg,
        epoch,
        epochs,
        start_key="retinal_thin_vessel_bias",
        end_key="retinal_thin_vessel_bias_end",
        schedule_key="retinal_thin_vessel_bias_schedule",
        fallback_start=0.0,
    )
    base_dataset = _unwrap_dataset(train_loader.dataset)
    if hasattr(base_dataset, "set_thin_vessel_bias"):
        base_dataset.set_thin_vessel_bias(target)
        return float(base_dataset.thin_vessel_bias)
    if hasattr(base_dataset, "thin_vessel_bias"):
        base_dataset.thin_vessel_bias = float(target)
        return float(base_dataset.thin_vessel_bias)
    return None


def apply_retinal_hard_mining_bias_schedule(
    train_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    epoch: int,
    epochs: int,
) -> float | None:
    target = _scheduled_retinal_bias(
        cfg,
        epoch,
        epochs,
        start_key="retinal_hard_mining_bias",
        end_key="retinal_hard_mining_bias_end",
        schedule_key="retinal_hard_mining_bias_schedule",
        fallback_start=0.0,
    )
    base_dataset = _unwrap_dataset(train_loader.dataset)
    if hasattr(base_dataset, "set_hard_mining_bias"):
        base_dataset.set_hard_mining_bias(target)
        return float(base_dataset.hard_mining_bias)
    if hasattr(base_dataset, "hard_mining_bias"):
        base_dataset.hard_mining_bias = float(target)
        return float(base_dataset.hard_mining_bias)
    return None


def metric_name_for_task(task: str) -> str:
    return "accuracy" if task == "classification" else "dice"


def selection_key_for_task(task: str) -> str:
    return "val_acc" if task == "classification" else "val_dice"


def summarize_score(task: str, split: str, metrics: Dict[str, float]) -> str:
    if task == "classification":
        return f"{split}_acc={metrics[f'{split}_acc']:.2f}%"
    return f"{split}_dice={metrics[f'{split}_dice']:.4f} {split}_iou={metrics[f'{split}_iou']:.4f}"


def spatial_shape_for_run(cfg: Dict[str, Any], task: str) -> tuple[int, int]:
    if task == "segmentation":
        dataset_name = utils.canonical_dataset_name(str(cfg.get("dataset", "")))
        if dataset_name in utils.GIS_SEG_DATASETS:
            patch_size = cfg.get("gis_patch_size", cfg.get("road_patch_size"))
            if patch_size:
                patch = int(patch_size)
                return (patch, patch)
            image_size = cfg.get("gis_image_size", cfg.get("road_image_size"))
            if image_size:
                parsed = utils._parse_optional_hw(image_size)
                if parsed is not None:
                    return parsed
        patch_size = cfg.get("retinal_patch_size", cfg.get("drive_patch_size"))
        if patch_size:
            patch = int(patch_size)
            return (patch, patch)
    return spatial_shape_for_dataset(cfg["dataset"])


def resolve_loss_cfg(
    cfg: Dict[str, Any],
    task: str,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    loss_cfg: Dict[str, Any] = {
        "bce_weight": float(cfg.get("bce_weight", 1.0)),
        "dice_weight": float(cfg.get("dice_weight", 1.0)),
        "overlap_mode": str(cfg.get("overlap_mode", "dice")),
        "tversky_alpha": float(cfg.get("tversky_alpha", 0.5)),
        "tversky_beta": float(cfg.get("tversky_beta", 0.5)),
        "threshold": float(cfg.get("seg_threshold", 0.5)),
        "aux_weight": float(cfg.get("aux_loss_weight", 0.2)),
        "boundary_weight": float(cfg.get("boundary_loss_weight", 0.1)),
        "boundary_dice_weight": float(cfg.get("boundary_dice_weight", 0.0)),
        "boundary_pos_weight": cfg.get("boundary_pos_weight"),
        "boundary_pos_weight_min": float(cfg.get("boundary_pos_weight_min", 1.0)),
        "boundary_pos_weight_max": float(cfg.get("boundary_pos_weight_max", 25.0)),
        "topology_weight": float(cfg.get("topology_loss_weight", 0.0)),
        "topology_num_iters": int(cfg.get("topology_num_iters", 10)),
        "axis_alignment_weight": float(cfg.get("axis_alignment_weight", 0.0)),
        "axis_alignment_num_iters": int(cfg.get("axis_alignment_num_iters", 8)),
        "pos_weight": None,
        "pos_weight_value": 1.0,
    }
    if task != "segmentation":
        return loss_cfg

    raw_pos_weight = cfg.get("bce_pos_weight", "auto")
    if raw_pos_weight == "auto":
        pos_weight_dataset = getattr(train_loader.dataset, "pos_weight_reference", train_loader.dataset)
        pos_weight_value = estimate_drive_pos_weight(
            pos_weight_dataset,
            min_weight=float(cfg.get("bce_pos_weight_min", 1.0)),
            max_weight=float(cfg.get("bce_pos_weight_max", 25.0)),
        )
    elif raw_pos_weight is None:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(raw_pos_weight)

    loss_cfg["pos_weight_value"] = pos_weight_value
    loss_cfg["pos_weight"] = torch.tensor(pos_weight_value, device=device)
    return loss_cfg


def load_checkpoint_payload(path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"model": payload}


def initialize_model_from_checkpoint(model: nn.Module, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    checkpoint_path = cfg.get("init_checkpoint")
    if not checkpoint_path:
        return None

    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("model", payload)
    strict = bool(cfg.get("init_checkpoint_strict", True))
    skipped_shape_keys: List[str] = []
    skipped_missing_keys: List[str] = []
    remapped_keys: Dict[str, str] = {}
    loaded_key_count = len(state_dict)
    if not strict:
        current_state = model.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            target_key = key
            if target_key not in current_state:
                remap_candidates = []
                if key.startswith("enc1.block."):
                    remap_candidates.append("enc1.conv_body." + key[len("enc1.block.") :])
                if key.startswith("enc2.block."):
                    remap_candidates.append("enc2.conv_body." + key[len("enc2.block.") :])
                if key.startswith("enc3.block."):
                    remap_candidates.append("enc3.body." + key[len("enc3.block.") :])
                for prefix in ("up3", "up2", "up1"):
                    source_prefix = f"{prefix}.fuse.block."
                    if key.startswith(source_prefix):
                        remap_candidates.append(f"{prefix}.fuse.body." + key[len(source_prefix) :])
                if key.startswith("head."):
                    remap_candidates.append("main_head." + key[len("head.") :])
                target_key = next((candidate for candidate in remap_candidates if candidate in current_state), key)
            if target_key not in current_state:
                skipped_missing_keys.append(key)
                continue
            if tuple(current_state[target_key].shape) != tuple(value.shape):
                skipped_shape_keys.append(key)
                continue
            compatible_state[target_key] = value
            if target_key != key:
                remapped_keys[key] = target_key
        state_dict = compatible_state
        loaded_key_count = len(compatible_state)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    return {
        "path": str(Path(checkpoint_path)),
        "strict": strict,
        "variant": payload.get("variant"),
        "loaded_key_count": loaded_key_count,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "skipped_missing_keys": skipped_missing_keys,
        "skipped_shape_keys": skipped_shape_keys,
        "remapped_keys": remapped_keys,
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epochs: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str | None]:
    scheduler_name = str(cfg.get("lr_scheduler", "none")).lower().strip()
    min_lr = float(cfg.get("lr_min", 0.0))
    if scheduler_name in {"", "none", "off"}:
        return None, None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg.get("lr_scheduler_t_max", epochs))),
            eta_min=min_lr,
        )
        return scheduler, "epoch"
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("lr_plateau_factor", 0.5)),
            patience=max(0, int(cfg.get("lr_plateau_patience", 8))),
            threshold=float(cfg.get("lr_plateau_threshold", 1e-4)),
            min_lr=min_lr,
        )
        return scheduler, "metric"
    raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")


def build_eval_model(model: nn.Module, task: str, cfg: Dict[str, Any]) -> nn.Module:
    if task != "segmentation":
        return model
    eval_model: nn.Module = model
    tile_size = cfg.get("eval_tile_size", cfg.get("gis_eval_tile_size"))
    if tile_size:
        eval_model = utils.SlidingWindowSegmentationWrapper(
            eval_model,
            tile_size=int(tile_size),
            overlap=int(cfg.get("eval_tile_overlap", cfg.get("gis_eval_tile_overlap", 0))),
        )
    tta_mode = str(cfg.get("eval_tta", "none")).lower().strip()
    if tta_mode in {"", "none", "off"}:
        return eval_model
    return utils.SegmentationTTAWrapper(eval_model, mode=tta_mode)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    task: str,
    loss_cfg: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    if task == "classification":
        assert criterion is not None
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
            "val_loss": loss_sum / max(n, 1),
            "val_acc": 100.0 * correct / max(n, 1),
        }

    loss_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    topology_sum = 0.0
    axis_sum = 0.0
    n = 0
    tp = fp = tn = fn = 0.0
    skel_pred_on_target = skel_pred_total = skel_target_on_pred = skel_target_total = 0.0
    for x, y, valid_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        logits = model(x)
        loss, aux, main_logits = segmentation_objective(
            logits,
            y,
            valid_mask,
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
            pos_weight=loss_cfg["pos_weight"],
            overlap_mode=loss_cfg["overlap_mode"],
            tversky_alpha=loss_cfg["tversky_alpha"],
            tversky_beta=loss_cfg["tversky_beta"],
            aux_weight=loss_cfg["aux_weight"],
            boundary_weight=loss_cfg["boundary_weight"],
            boundary_dice_weight=loss_cfg["boundary_dice_weight"],
            boundary_pos_weight=loss_cfg["boundary_pos_weight"],
            boundary_pos_weight_min=loss_cfg["boundary_pos_weight_min"],
            boundary_pos_weight_max=loss_cfg["boundary_pos_weight_max"],
            topology_weight=loss_cfg["topology_weight"],
            topology_num_iters=loss_cfg["topology_num_iters"],
        )
        axis_loss_value = main_logits.new_zeros(())
        axis_weight = float(loss_cfg.get("axis_alignment_weight", 0.0))
        if axis_weight > 0.0 and hasattr(model, "axis_alignment_loss"):
            axis_loss_value = model.axis_alignment_loss(
                y,
                valid_mask,
                num_iters=int(loss_cfg.get("axis_alignment_num_iters", 8)),
            )
            loss = loss + axis_weight * axis_loss_value
        loss_sum += loss.item() * x.size(0)
        bce_sum += aux["bce_loss"] * x.size(0)
        dice_loss_sum += aux["dice_loss"] * x.size(0)
        topology_sum += aux["topology_loss"] * x.size(0)
        axis_sum += axis_loss_value.item() * x.size(0)
        c_tp, c_fp, c_tn, c_fn = binary_confusion_counts(
            main_logits,
            y,
            valid_mask,
            threshold=loss_cfg["threshold"],
        )
        tp += c_tp
        fp += c_fp
        tn += c_tn
        fn += c_fn
        s_pot, s_pred, s_top, s_target = skeleton_confusion_counts(
            main_logits,
            y,
            valid_mask,
            threshold=loss_cfg["threshold"],
            num_iters=int(loss_cfg["topology_num_iters"]),
        )
        skel_pred_on_target += s_pot
        skel_pred_total += s_pred
        skel_target_on_pred += s_top
        skel_target_total += s_target
        n += x.size(0)

    metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
    skeleton_metrics = skeleton_metrics_from_counts(
        skel_pred_on_target,
        skel_pred_total,
        skel_target_on_pred,
        skel_target_total,
    )
    return {
        "val_loss": loss_sum / max(n, 1),
        "val_bce_loss": bce_sum / max(n, 1),
        "val_dice_loss": dice_loss_sum / max(n, 1),
        "val_topology_loss": topology_sum / max(n, 1),
        "val_axis_loss": axis_sum / max(n, 1),
        "val_dice": metrics["dice"],
        "val_iou": metrics["iou"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_specificity": metrics["specificity"],
        "val_accuracy": metrics["accuracy"],
        "val_balanced_accuracy": metrics["balanced_accuracy"],
        "val_cldice": skeleton_metrics["cldice"],
        "val_skeleton_precision": skeleton_metrics["skeleton_precision"],
        "val_skeleton_recall": skeleton_metrics["skeleton_recall"],
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reg_weights: Dict[str, float],
    task: str,
    loss_cfg: Dict[str, float],
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    objective_sum = 0.0
    n = 0
    reg_sums = {key: 0.0 for key in REG_LOG_KEYS}

    if task == "classification":
        assert criterion is not None
        correct = 0
        for x, y in tqdm(loader, desc="train", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            base_loss = criterion(logits, y)
            loss = base_loss

            if hasattr(model, "regularization_terms"):
                reg_terms = model.regularization_terms()
                for key in REG_LOG_KEYS:
                    if key in reg_terms:
                        reg_sums[key] += reg_terms[key].item() * x.size(0)
                for key, weight in reg_weights.items():
                    if weight > 0.0 and key in reg_terms:
                        loss = loss + weight * reg_terms[key]

            loss.backward()
            optimizer.step()

            loss_sum += base_loss.item() * x.size(0)
            objective_sum += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)

        metrics = {
            "train_loss": loss_sum / max(n, 1),
            "train_objective": objective_sum / max(n, 1),
            "train_acc": 100.0 * correct / max(n, 1),
        }
    else:
        bce_sum = 0.0
        dice_loss_sum = 0.0
        aux_sum = 0.0
        boundary_sum = 0.0
        topology_sum = 0.0
        axis_sum = 0.0
        tp = fp = tn = fn = 0.0
        skel_pred_on_target = skel_pred_total = skel_target_on_pred = skel_target_total = 0.0
        for x, y, valid_mask in tqdm(loader, desc="train", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            base_loss, aux, main_logits = segmentation_objective(
                logits,
                y,
                valid_mask,
                bce_weight=loss_cfg["bce_weight"],
                dice_weight=loss_cfg["dice_weight"],
                pos_weight=loss_cfg["pos_weight"],
                overlap_mode=loss_cfg["overlap_mode"],
                tversky_alpha=loss_cfg["tversky_alpha"],
                tversky_beta=loss_cfg["tversky_beta"],
                aux_weight=loss_cfg["aux_weight"],
                boundary_weight=loss_cfg["boundary_weight"],
                boundary_dice_weight=loss_cfg["boundary_dice_weight"],
                boundary_pos_weight=loss_cfg["boundary_pos_weight"],
                boundary_pos_weight_min=loss_cfg["boundary_pos_weight_min"],
                boundary_pos_weight_max=loss_cfg["boundary_pos_weight_max"],
                topology_weight=loss_cfg["topology_weight"],
                topology_num_iters=loss_cfg["topology_num_iters"],
            )
            loss = base_loss
            axis_loss_value = main_logits.new_zeros(())
            axis_weight = float(loss_cfg.get("axis_alignment_weight", 0.0))
            if axis_weight > 0.0 and hasattr(model, "axis_alignment_loss"):
                axis_loss_value = model.axis_alignment_loss(
                    y,
                    valid_mask,
                    num_iters=int(loss_cfg.get("axis_alignment_num_iters", 8)),
                )
                loss = loss + axis_weight * axis_loss_value

            if hasattr(model, "regularization_terms"):
                reg_terms = model.regularization_terms()
                for key in REG_LOG_KEYS:
                    if key in reg_terms:
                        reg_sums[key] += reg_terms[key].item() * x.size(0)
                for key, weight in reg_weights.items():
                    if weight > 0.0 and key in reg_terms:
                        loss = loss + weight * reg_terms[key]

            loss.backward()
            optimizer.step()

            loss_sum += base_loss.item() * x.size(0)
            objective_sum += loss.item() * x.size(0)
            bce_sum += aux["bce_loss"] * x.size(0)
            dice_loss_sum += aux["dice_loss"] * x.size(0)
            aux_sum += aux["aux_loss"] * x.size(0)
            boundary_sum += aux["boundary_loss"] * x.size(0)
            topology_sum += aux["topology_loss"] * x.size(0)
            axis_sum += axis_loss_value.item() * x.size(0)
            c_tp, c_fp, c_tn, c_fn = binary_confusion_counts(
                main_logits.detach(),
                y,
                valid_mask,
                threshold=loss_cfg["threshold"],
            )
            tp += c_tp
            fp += c_fp
            tn += c_tn
            fn += c_fn
            s_pot, s_pred, s_top, s_target = skeleton_confusion_counts(
                main_logits.detach(),
                y,
                valid_mask,
                threshold=loss_cfg["threshold"],
                num_iters=int(loss_cfg["topology_num_iters"]),
            )
            skel_pred_on_target += s_pot
            skel_pred_total += s_pred
            skel_target_on_pred += s_top
            skel_target_total += s_target
            n += x.size(0)

        seg_metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
        skeleton_metrics = skeleton_metrics_from_counts(
            skel_pred_on_target,
            skel_pred_total,
            skel_target_on_pred,
            skel_target_total,
        )
        metrics = {
            "train_loss": loss_sum / max(n, 1),
            "train_objective": objective_sum / max(n, 1),
            "train_bce_loss": bce_sum / max(n, 1),
            "train_dice_loss": dice_loss_sum / max(n, 1),
            "train_aux_loss": aux_sum / max(n, 1),
            "train_boundary_loss": boundary_sum / max(n, 1),
            "train_topology_loss": topology_sum / max(n, 1),
            "train_axis_loss": axis_sum / max(n, 1),
            "train_dice": seg_metrics["dice"],
            "train_iou": seg_metrics["iou"],
            "train_precision": seg_metrics["precision"],
            "train_recall": seg_metrics["recall"],
            "train_specificity": seg_metrics["specificity"],
            "train_accuracy": seg_metrics["accuracy"],
            "train_balanced_accuracy": seg_metrics["balanced_accuracy"],
            "train_cldice": skeleton_metrics["cldice"],
            "train_skeleton_precision": skeleton_metrics["skeleton_precision"],
            "train_skeleton_recall": skeleton_metrics["skeleton_recall"],
        }

    for key in REG_LOG_KEYS:
        metrics[f"reg_{key}"] = reg_sums[key] / max(n, 1)
    return metrics


def plot_single_run(history: List[Dict[str, Any]], out_dir: Path, variant: str, task: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {variant}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    if task == "classification":
        train_key = "train_acc"
        val_key = "val_acc"
        ylabel = "Accuracy (%)"
        title = f"Accuracy - {variant}"
        filename = "accuracy_curves.png"
    else:
        train_key = "train_dice"
        val_key = "val_dice"
        ylabel = "Dice"
        title = f"Dice - {variant}"
        filename = "dice_curves.png"

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r[train_key] for r in history], label=train_key)
    plt.plot(epochs, [r[val_key] for r in history], label=val_key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


def collect_architecture_state(model: nn.Module) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    mix_rows: List[Dict[str, float]] = []
    residual_rows: List[Dict[str, float]] = []
    az_rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        mix_logit = getattr(module, "mix_logit", None)
        if isinstance(mix_logit, torch.nn.Parameter):
            mix_alpha = float(torch.sigmoid(mix_logit.detach()).cpu())
            conv_alpha = float(max(1.0 - mix_alpha, 1e-8))
            mix_rows.append(
                {
                    "name": name,
                    "value": mix_alpha,
                    "az_to_conv_ratio": float(mix_alpha / conv_alpha),
                }
            )
        residual_logit = getattr(module, "residual_logit", None)
        if isinstance(residual_logit, torch.nn.Parameter):
            residual_alpha = float(torch.sigmoid(residual_logit.detach()).cpu())
            residual_rows.append({"name": name, "value": residual_alpha})
        if isinstance(module, AZConv2d):
            row: Dict[str, Any] = {"name": name, "rules": int(module.R), "kernel_size": int(module.k)}
            row.update(module.metric_tensor_summary())
            snapshot = module.interpretation_snapshot()
            mu_rule_mean = snapshot.get("mu_rule_mean")
            if isinstance(mu_rule_mean, torch.Tensor):
                probs = mu_rule_mean.float().clamp_min(1e-8)
                probs = probs / probs.sum().clamp_min(1e-8)
                row["rule_usage_entropy_norm"] = float((-(probs * probs.log()).sum() / math.log(max(int(module.R), 2))).cpu())
            compat_map = snapshot.get("compat_map")
            if isinstance(compat_map, torch.Tensor):
                row["compat_mass"] = float(compat_map.float().sum().cpu())

            # Model-native direction diversity diagnostics from raw theta_map.
            theta_map = snapshot.get("theta_map")
            mu_map = snapshot.get("mu_map")
            if isinstance(theta_map, torch.Tensor) and isinstance(mu_map, torch.Tensor):
                theta = theta_map.float()
                mu = mu_map.float()
                if theta.ndim == 3 and mu.ndim == 3 and theta.shape == mu.shape:
                    dominant = mu.argmax(dim=0)
                    yy, xx = torch.meshgrid(
                        torch.arange(theta.shape[1], device=theta.device),
                        torch.arange(theta.shape[2], device=theta.device),
                        indexing="ij",
                    )
                    theta_dom = theta[dominant, yy, xx].reshape(-1)
                    if theta_dom.numel() > 0:
                        complex_dir = torch.polar(torch.ones_like(theta_dom), theta_dom)
                        resultant_dir = complex_dir.mean()
                        r_dir = torch.abs(resultant_dir).clamp(0.0, 1.0)
                        row["direction_resultant_r"] = float(r_dir.detach().cpu())
                        row["direction_diversity"] = float((1.0 - r_dir).detach().cpu())

                        # Orientation periodicity pi (theta and theta+pi equivalent).
                        theta2 = 2.0 * theta_dom
                        complex_ori = torch.polar(torch.ones_like(theta2), theta2)
                        resultant_ori = complex_ori.mean()
                        r_ori = torch.abs(resultant_ori).clamp(0.0, 1.0)
                        row["orientation_resultant_r"] = float(r_ori.detach().cpu())
                        row["orientation_diversity"] = float((1.0 - r_ori).detach().cpu())

                        bins = 12
                        hist = torch.histc(theta_dom, bins=bins, min=-math.pi, max=math.pi)
                        nonzero = (hist > 0).float().mean()
                        row["direction_hist_nonzero_frac"] = float(nonzero.detach().cpu())
            az_rows.append(row)

    if mix_rows:
        mix_vals = [row["value"] for row in mix_rows]
        mix_ratio_vals = [row["az_to_conv_ratio"] for row in mix_rows]
        state["hybrid_mix_alpha"] = mix_rows
        state["hybrid_mix_alpha_mean"] = float(sum(mix_vals) / len(mix_vals))
        state["hybrid_az_to_conv_ratio_mean"] = float(sum(mix_ratio_vals) / len(mix_ratio_vals))
        state["hybrid_az_to_conv_ratio_min"] = float(min(mix_ratio_vals))
        state["hybrid_az_to_conv_ratio_max"] = float(max(mix_ratio_vals))
    if residual_rows:
        residual_vals = [row["value"] for row in residual_rows]
        state["az_input_residual_alpha"] = residual_rows
        state["az_input_residual_alpha_mean"] = float(sum(residual_vals) / len(residual_vals))
    if az_rows:
        state["az_layer_count"] = len(az_rows)
        state["az_layers"] = az_rows
        mode_counts: Dict[str, int] = {}
        normalize_counts: Dict[str, int] = {}
        for row in az_rows:
            mode = str(row.get("geometry_mode", "unknown"))
            norm = str(row.get("normalize_mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            normalize_counts[norm] = normalize_counts.get(norm, 0) + 1
        state["az_geometry_mode_counts"] = mode_counts
        state["az_normalize_mode_counts"] = normalize_counts
        for key in (
            "metric_min_eig",
            "metric_condition",
            "anisotropy_gap",
            "rule_usage_entropy_norm",
            "compat_mass",
            "direction_resultant_r",
            "direction_diversity",
            "orientation_resultant_r",
            "orientation_diversity",
            "direction_hist_nonzero_frac",
        ):
            vals = [float(row[key]) for row in az_rows if key in row]
            if vals:
                state[f"az_{key}_mean"] = float(sum(vals) / len(vals))
                state[f"az_{key}_min"] = float(min(vals))
                state[f"az_{key}_max"] = float(max(vals))
    return state


def run_training(cfg: Dict[str, Any], variant: str, run_dir: Path) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    device_str = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    reg_weights = az_regularization_weights(cfg)
    if variant.startswith("az_"):
        utils.sanity_check_azconv_forward(device)

    train_loader, val_loader, test_loader, in_c, num_outputs, task = build_dataloaders(cfg)
    model = build_model(
        variant,
        num_outputs=num_outputs,
        in_channels=in_c,
        num_rules=int(cfg.get("num_rules", 4)),
        task=task,
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    init_info = initialize_model_from_checkpoint(model, cfg)
    eval_model = build_eval_model(model, task, cfg)
    loss_cfg = resolve_loss_cfg(cfg, task, train_loader, device)

    n_params = utils.count_parameters(model)
    complexity = utils.estimate_model_complexity(
        model,
        device=device,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial_shape=spatial_shape_for_run(cfg, task),
    )
    criterion = nn.CrossEntropyLoss() if task == "classification" else None
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    epochs = int(cfg["epochs"])
    scheduler, scheduler_step_mode = build_lr_scheduler(optimizer, cfg, epochs)
    history: List[Dict[str, Any]] = []
    selection_key = selection_key_for_task(task)
    best_val = float("-inf")
    best_state = None
    best_epoch = 0
    threshold_selection_metric = str(cfg.get("eval_threshold_metric", "dice"))
    threshold_selection_reference_cfg = cfg.get("eval_threshold_reference")
    threshold_selection_score_tolerance = float(cfg.get("eval_threshold_score_tolerance", 0.0))
    threshold_selection_max = cfg.get("eval_threshold_max")
    threshold_selection_max = float(threshold_selection_max) if threshold_selection_max is not None else None
    threshold_selection_min_recall = cfg.get("eval_threshold_min_recall")
    threshold_selection_min_recall = (
        float(threshold_selection_min_recall) if threshold_selection_min_recall is not None else None
    )
    threshold_grid_for_selection: List[float] = []
    if task == "segmentation" and bool(cfg.get("eval_threshold_sweep", True)):
        threshold_grid_for_selection = build_threshold_grid(
            float(cfg.get("eval_threshold_start", 0.3)),
            float(cfg.get("eval_threshold_end", 0.8)),
            float(cfg.get("eval_threshold_step", 0.05)),
        )

    for epoch in range(1, epochs + 1):
        current_foreground_bias = None
        current_thin_vessel_bias = None
        current_hard_mining_bias = None
        if task == "segmentation":
            current_foreground_bias = apply_retinal_foreground_bias_schedule(train_loader, cfg, epoch, epochs)
            current_thin_vessel_bias = apply_retinal_thin_vessel_bias_schedule(train_loader, cfg, epoch, epochs)
            current_hard_mining_bias = apply_retinal_hard_mining_bias_schedule(train_loader, cfg, epoch, epochs)
        t0 = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, reg_weights, task, loss_cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t0

        val_metrics = evaluate_epoch(eval_model, val_loader, criterion, device, task, loss_cfg)
        val_selection_score = float(val_metrics[selection_key])
        val_selection_threshold = float(loss_cfg["threshold"])
        if threshold_grid_for_selection:
            val_sweep_rows = sweep_segmentation_thresholds(
                eval_model,
                val_loader,
                device,
                threshold_grid_for_selection,
                skeleton_num_iters=int(loss_cfg["topology_num_iters"]),
            )
            best_val_row = select_best_threshold(
                val_sweep_rows,
                metric=threshold_selection_metric,
                reference_threshold=(
                    float(threshold_selection_reference_cfg)
                    if threshold_selection_reference_cfg is not None
                    else float(loss_cfg["threshold"])
                ),
                score_tolerance=threshold_selection_score_tolerance,
                max_threshold=threshold_selection_max,
                min_recall=threshold_selection_min_recall,
            )
            val_selection_score = float(threshold_metric_value(best_val_row, threshold_selection_metric))
            val_selection_threshold = float(best_val_row["threshold"])

        row = {"epoch": epoch, "seconds_train_epoch": t_epoch, "lr": current_lr}
        if current_foreground_bias is not None:
            row["retinal_foreground_bias_epoch"] = float(current_foreground_bias)
        if current_thin_vessel_bias is not None:
            row["retinal_thin_vessel_bias_epoch"] = float(current_thin_vessel_bias)
        if current_hard_mining_bias is not None:
            row["retinal_hard_mining_bias_epoch"] = float(current_hard_mining_bias)
        row.update(train_metrics)
        row.update(val_metrics)
        if threshold_grid_for_selection:
            row["val_selection_metric"] = threshold_selection_metric
            row["val_selection_score"] = val_selection_score
            row["val_selection_threshold"] = val_selection_threshold

        if scheduler is not None:
            if scheduler_step_mode == "metric":
                scheduler.step(val_selection_score)
            else:
                scheduler.step()
            row["lr_next"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        msg = (
            f"[{variant}] epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_obj={train_metrics['train_objective']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"{summarize_score(task, 'val', val_metrics)} "
            f"time={t_epoch:.1f}s"
        )
        if threshold_grid_for_selection:
            msg += (
                f" sel_{threshold_selection_metric}={val_selection_score:.4f}"
                f" sel_thr={val_selection_threshold:.3f}"
            )
        if current_foreground_bias is not None:
            msg += f" fg_bias={current_foreground_bias:.2f}"
        if current_thin_vessel_bias is not None:
            msg += f" thin_bias={current_thin_vessel_bias:.2f}"
        if current_hard_mining_bias is not None:
            msg += f" hard_bias={current_hard_mining_bias:.2f}"
        if variant.startswith("az_"):
            msg += f" anis_gap={train_metrics['reg_anisotropy_gap']:.4f}"
        print(msg)

        current_val = val_selection_score
        if current_val > best_val:
            best_val = current_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "variant": variant,
                    "cfg": cfg,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_val": best_val,
                },
                run_dir / "checkpoint_best.pt",
            )

        with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)

    selected_threshold = float(loss_cfg["threshold"])
    threshold_selection_mode = "fixed"
    threshold_selection_metric_name = None
    threshold_sweep_rows: List[Dict[str, float]] = []
    val_selected_metrics: Dict[str, float] | None = None
    threshold_grid: List[float] = []

    if task == "segmentation" and bool(cfg.get("eval_threshold_sweep", True)):
        threshold_metric = threshold_selection_metric
        threshold_start = float(cfg.get("eval_threshold_start", 0.3))
        threshold_end = float(cfg.get("eval_threshold_end", 0.8))
        threshold_step = float(cfg.get("eval_threshold_step", 0.05))
        threshold_grid = build_threshold_grid(threshold_start, threshold_end, threshold_step)
        threshold_sweep_rows = sweep_segmentation_thresholds(
            eval_model,
            val_loader,
            device,
            threshold_grid,
            skeleton_num_iters=int(loss_cfg["topology_num_iters"]),
        )
        best_threshold_row = select_best_threshold(
            threshold_sweep_rows,
            metric=threshold_metric,
            reference_threshold=(
                float(threshold_selection_reference_cfg)
                if threshold_selection_reference_cfg is not None
                else float(selected_threshold)
            ),
            score_tolerance=threshold_selection_score_tolerance,
            max_threshold=threshold_selection_max,
            min_recall=threshold_selection_min_recall,
        )
        selected_threshold = float(best_threshold_row["threshold"])
        loss_cfg["threshold"] = selected_threshold
        threshold_selection_mode = "val_sweep"
        threshold_selection_metric_name = threshold_metric
        val_selected_metrics = evaluate_epoch(eval_model, val_loader, criterion, device, task, loss_cfg)

    test_metrics = evaluate_epoch(eval_model, test_loader, criterion, device, task, loss_cfg)

    sec_batch = measure_inference_time(
        eval_model,
        device,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial_shape=spatial_shape_for_run(cfg, task),
        warmup=int(cfg.get("timing_warmup", 3)),
        iters=int(cfg.get("timing_iters", 20)),
    )
    mean_train_epoch = float(sum(h["seconds_train_epoch"] for h in history) / max(len(history), 1))

    metrics: Dict[str, Any] = {
        "run_name": run_dir.name,
        "variant": variant,
        "article_model_name": cfg.get("article_model_name"),
        "article_internal_recipe": cfg.get("article_internal_recipe"),
        "dataset": cfg["dataset"],
        "task": task,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "seed": int(cfg["seed"]),
        "best_epoch": best_epoch,
        "num_rules": int(cfg.get("num_rules", 4)),
        "model_widths": list(utils.parse_model_widths(cfg.get("model_widths")) or []),
        "encoder_az_stages": cfg.get("encoder_az_stages"),
        "encoder_block_mode": cfg.get("encoder_block_mode"),
        "hybrid_mix_init": cfg.get("hybrid_mix_init"),
        "hybrid_mix_target": cfg.get("hybrid_mix_target"),
        "bottleneck_mode": cfg.get("bottleneck_mode"),
        "decoder_mode": cfg.get("decoder_mode"),
        "boundary_mode": cfg.get("boundary_mode"),
        "az_geometry_mode": str(cfg.get("az_geometry_mode", "variant_default")),
        "az_learn_directions": cfg.get("az_learn_directions"),
        "az_use_fuzzy": cfg.get("az_use_fuzzy"),
        "az_use_anisotropy": cfg.get("az_use_anisotropy"),
        "az_min_hyperbolicity": cfg.get("az_min_hyperbolicity"),
        "az_fuzzy_temperature": cfg.get("az_fuzzy_temperature"),
        "az_normalize_mode": str(cfg.get("az_normalize_mode", "variant_default")),
        "az_compatibility_floor": cfg.get("az_compatibility_floor"),
        "az_geometry_kernel_size": cfg.get("az_geometry_kernel_size"),
        "az_init_anisotropy_gap": cfg.get("az_init_anisotropy_gap"),
        "az_max_hyperbolicity": cfg.get("az_max_hyperbolicity"),
        "az_min_membership_entropy": cfg.get("az_min_membership_entropy"),
        "az_use_input_residual": cfg.get("az_use_input_residual"),
        "az_residual_init": cfg.get("az_residual_init"),
        "lr_scheduler": str(cfg.get("lr_scheduler", "none")),
        "lr_min": float(cfg.get("lr_min", 0.0)),
        "eval_tta": str(cfg.get("eval_tta", "none")),
        "eval_tta_views": utils.segmentation_tta_num_views(cfg.get("eval_tta", "none")) if task == "segmentation" else 1,
        "retinal_input_mode": str(cfg.get("retinal_input_mode", "rgb")),
        "retinal_green_blend_alpha": float(cfg.get("retinal_green_blend_alpha", 0.35)),
        "retinal_foreground_bias": float(cfg.get("retinal_foreground_bias", cfg.get("drive_foreground_bias", 0.0))),
        "retinal_foreground_bias_end": float(cfg.get("retinal_foreground_bias_end", cfg.get("retinal_foreground_bias", cfg.get("drive_foreground_bias", 0.0)))),
        "retinal_foreground_bias_schedule": str(cfg.get("retinal_foreground_bias_schedule", "constant")),
        "retinal_thin_vessel_bias": float(cfg.get("retinal_thin_vessel_bias", 0.0)),
        "retinal_thin_vessel_bias_end": float(cfg.get("retinal_thin_vessel_bias_end", cfg.get("retinal_thin_vessel_bias", 0.0))),
        "retinal_thin_vessel_bias_schedule": str(cfg.get("retinal_thin_vessel_bias_schedule", "constant")),
        "retinal_thin_vessel_neighbor_threshold": int(cfg.get("retinal_thin_vessel_neighbor_threshold", 4)),
        "retinal_hard_mining_dir": str(cfg.get("retinal_hard_mining_dir", "")),
        "retinal_hard_mining_bias": float(cfg.get("retinal_hard_mining_bias", 0.0)),
        "retinal_hard_mining_bias_end": float(cfg.get("retinal_hard_mining_bias_end", cfg.get("retinal_hard_mining_bias", 0.0))),
        "retinal_hard_mining_bias_schedule": str(cfg.get("retinal_hard_mining_bias_schedule", "constant")),
        "resolved_test_split": str(cfg.get("_resolved_test_split", "test")),
        "num_parameters": n_params,
        **complexity,
        "test_loss": test_metrics["val_loss"],
        "seconds_per_train_epoch_mean": mean_train_epoch,
        "seconds_per_forward_batch": sec_batch,
        "regularization_weights": reg_weights,
    }
    if history:
        metrics["final_lr"] = float(history[-1].get("lr_next", history[-1].get("lr", cfg["lr"])))
    if init_info is not None:
        metrics["init_checkpoint"] = init_info
    if task == "segmentation":
        tta_views = int(metrics["eval_tta_views"])
        metrics["approx_eval_gmacs_per_forward"] = float(metrics["approx_gmacs_per_forward"]) * tta_views
        metrics["approx_eval_gflops_per_forward"] = float(metrics["approx_gflops_per_forward"]) * tta_views
        metrics["approx_eval_az_extra_gmacs_per_forward"] = float(metrics["approx_az_extra_gmacs_per_forward"]) * tta_views
    if task == "classification":
        metrics["best_val_accuracy"] = best_val
        metrics["test_accuracy"] = test_metrics["val_acc"]
    else:
        metrics["bce_loss_weight"] = float(cfg.get("bce_weight", 1.0))
        metrics["dice_loss_weight"] = float(cfg.get("dice_weight", 1.0))
        metrics["overlap_mode"] = str(cfg.get("overlap_mode", "dice"))
        metrics["tversky_alpha"] = float(cfg.get("tversky_alpha", 0.5))
        metrics["tversky_beta"] = float(cfg.get("tversky_beta", 0.5))
        metrics["aux_loss_weight"] = float(cfg.get("aux_loss_weight", 0.2))
        metrics["boundary_loss_weight"] = float(cfg.get("boundary_loss_weight", 0.1))
        metrics["boundary_dice_weight"] = float(cfg.get("boundary_dice_weight", 0.0))
        metrics["boundary_pos_weight"] = cfg.get("boundary_pos_weight")
        metrics["boundary_pos_weight_min"] = float(cfg.get("boundary_pos_weight_min", 1.0))
        metrics["boundary_pos_weight_max"] = float(cfg.get("boundary_pos_weight_max", 25.0))
        metrics["bce_pos_weight"] = loss_cfg["pos_weight_value"]
        metrics["best_val_selection_score"] = best_val
        metrics["best_val_selection_metric"] = threshold_selection_metric
        if threshold_selection_metric == "dice":
            metrics["best_val_dice"] = best_val
        metrics["seg_threshold_base"] = float(cfg.get("seg_threshold", 0.5))
        metrics["selected_threshold"] = selected_threshold
        metrics["threshold_selection_mode"] = threshold_selection_mode
        metrics["topology_loss_weight"] = float(cfg.get("topology_loss_weight", 0.0))
        metrics["topology_num_iters"] = int(cfg.get("topology_num_iters", 10))
        if threshold_selection_metric_name is not None:
            metrics["threshold_selection_metric"] = threshold_selection_metric_name
            metrics["threshold_selection_reference"] = (
                float(threshold_selection_reference_cfg)
                if threshold_selection_reference_cfg is not None
                else None
            )
            metrics["threshold_selection_score_tolerance"] = threshold_selection_score_tolerance
            metrics["threshold_selection_max"] = threshold_selection_max
            metrics["threshold_selection_min_recall"] = threshold_selection_min_recall
            metrics["val_threshold_sweep"] = threshold_sweep_rows
        if val_selected_metrics is not None:
            metrics["val_selected_threshold_metrics"] = val_selected_metrics
        metrics["test_bce_loss"] = test_metrics["val_bce_loss"]
        metrics["test_dice_loss"] = test_metrics["val_dice_loss"]
        metrics["test_topology_loss"] = test_metrics["val_topology_loss"]
        metrics["test_dice"] = test_metrics["val_dice"]
        metrics["test_iou"] = test_metrics["val_iou"]
        metrics["test_precision"] = test_metrics["val_precision"]
        metrics["test_recall"] = test_metrics["val_recall"]
        metrics["test_specificity"] = test_metrics["val_specificity"]
        metrics["test_accuracy"] = test_metrics["val_accuracy"]
        metrics["test_balanced_accuracy"] = test_metrics["val_balanced_accuracy"]
        metrics["test_cldice"] = test_metrics["val_cldice"]
        metrics["test_skeleton_precision"] = test_metrics["val_skeleton_precision"]
        metrics["test_skeleton_recall"] = test_metrics["val_skeleton_recall"]
        if bool(cfg.get("search_beats_baseline", False)):
            search_metric = str(cfg.get("search_selection_metric", "dice"))
            search_grid = threshold_grid or build_threshold_grid(
                float(cfg.get("eval_threshold_start", 0.3)),
                float(cfg.get("eval_threshold_end", 0.8)),
                float(cfg.get("eval_threshold_step", 0.05)),
            )
            search_rows = sweep_segmentation_thresholds(
                model,
                test_loader,
                device,
                search_grid,
                skeleton_num_iters=int(loss_cfg["topology_num_iters"]),
            )
            baseline_metrics = select_best_drive_record(utils.collect_drive_metrics_records(run_dir.parent), variant="baseline")
            search_report = build_drive_threshold_search_report(
                sweep_rows=search_rows,
                baseline_metrics=baseline_metrics,
                metric_names=list(utils.DRIVE_SUPERIORITY_METRICS),
                min_delta=float(cfg.get("comparison_min_delta", 0.0)),
                selection_metric=search_metric,
            )
            metrics["test_threshold_dominance_search"] = search_report
    if history:
        metrics["final_anisotropy_gap"] = history[-1].get("reg_anisotropy_gap", 0.0)
    arch_state = collect_architecture_state(model)
    if arch_state:
        metrics["architecture_state"] = arch_state
    save_json(run_dir / "metrics.json", metrics)
    if task == "segmentation" and str(cfg["dataset"]).lower().replace("-", "_") == "drive":
        update_drive_comparison_summary(run_dir.parent)
    plot_single_run(history, run_dir, variant, task)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline or AZ variants on classification or DRIVE segmentation.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated list of variants to run sequentially.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run name or multi-run prefix.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic PyTorch/CuDNN execution.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay from config.")
    parser.add_argument("--model-widths", type=str, default=None, help="Override segmentation widths, e.g. 32,64,128,192.")
    parser.add_argument("--num-rules", type=int, default=None, help="Override ANZA rule count from config.")
    parser.add_argument("--overlap-mode", type=str, default=None, help="Segmentation overlap term: dice or tversky.")
    parser.add_argument("--tversky-alpha", type=float, default=None, help="False-positive weight for Tversky overlap.")
    parser.add_argument("--tversky-beta", type=float, default=None, help="False-negative weight for Tversky overlap.")
    parser.add_argument("--aux-loss-weight", type=float, default=None, help="Override auxiliary loss weight from config.")
    parser.add_argument("--boundary-loss-weight", type=float, default=None, help="Override boundary loss weight from config.")
    parser.add_argument("--topology-loss-weight", type=float, default=None, help="Override topology loss weight from config.")
    parser.add_argument("--topology-num-iters", type=int, default=None, help="Override topology skeletonization iterations.")
    parser.add_argument("--bce-pos-weight", type=float, default=None, help="Override BCE positive-class weight. Use a smaller value to reduce false positives, or a larger value to favor recall.")
    parser.add_argument("--drive-foreground-bias", type=float, default=None, help="Override DRIVE patch foreground bias from config.")
    parser.add_argument("--retinal-patch-size", type=int, default=None, help="Override retinal training crop size for vessel datasets.")
    parser.add_argument("--retinal-foreground-bias", type=float, default=None, help="Override retinal foreground-biased crop probability.")
    parser.add_argument("--encoder-az-stages", type=int, default=None, help="Number of encoder stages that use AZ blocks in az_thesis/az_sota.")
    parser.add_argument("--encoder-block-mode", type=str, default=None, help="Encoder AZ block type for az_thesis/az_sota: az or hybrid.")
    parser.add_argument("--hybrid-mix-init", type=float, default=None, help="Initial AZ branch mixture for hybrid encoder blocks in (0, 1).")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional checkpoint used to initialize model weights before training.")
    parser.add_argument("--init-checkpoint-strict", action="store_true", help="Load init checkpoint with strict key matching.")
    parser.add_argument("--lr-scheduler", type=str, default=None, help="Optional learning-rate scheduler: none, cosine, or plateau.")
    parser.add_argument("--lr-min", type=float, default=None, help="Minimum learning rate for cosine/plateau schedulers.")
    parser.add_argument("--lr-scheduler-t-max", type=int, default=None, help="Optional cosine scheduler T_max override.")
    parser.add_argument("--lr-plateau-factor", type=float, default=None, help="Factor for ReduceLROnPlateau.")
    parser.add_argument("--lr-plateau-patience", type=int, default=None, help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--lr-plateau-threshold", type=float, default=None, help="Improvement threshold for ReduceLROnPlateau.")
    parser.add_argument("--eval-tta", type=str, default=None, help="Optional segmentation-time augmentation for validation/test: none, flips, d4.")
    parser.add_argument("--eval-threshold-start", type=float, default=None, help="Override threshold-sweep start.")
    parser.add_argument("--eval-threshold-end", type=float, default=None, help="Override threshold-sweep end.")
    parser.add_argument("--eval-threshold-step", type=float, default=None, help="Override threshold-sweep step.")
    parser.add_argument(
        "--eval-threshold-metric",
        type=str,
        default=None,
        help="Metric used to choose the validation threshold. Supports raw metrics like dice/precision and derived metrics like core_mean.",
    )
    parser.add_argument(
        "--require-beats-baseline",
        action="store_true",
        help="After each DRIVE segmentation run, compare the current variant against the best stored baseline run and fail if any core metric is not higher.",
    )
    parser.add_argument(
        "--search-beats-baseline",
        action="store_true",
        help="Exploratory mode: search the declared threshold grid on the test set and report whether any operating point beats baseline on all gated metrics.",
    )
    parser.add_argument(
        "--search-selection-metric",
        type=str,
        default="dice",
        help="Metric used to choose the best threshold among those that beat baseline in exploratory search.",
    )
    parser.add_argument(
        "--comparison-metrics",
        type=str,
        default=None,
        help="Comma-separated test metrics for the superiority gate. Default: Dice, IoU, Precision, Recall, Specificity, Balanced Accuracy.",
    )
    parser.add_argument(
        "--comparison-min-delta",
        type=float,
        default=0.0,
        help="Required positive delta over baseline for each gated metric.",
    )
    parser.add_argument(
        "--canonical-drive",
        action="store_true",
        help="Run the canonical DRIVE comparison set: baseline, az_no_fuzzy, az_no_aniso, az_cat, az_thesis.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.deterministic:
        cfg["deterministic"] = True
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["weight_decay"] = float(args.weight_decay)
    if args.model_widths is not None:
        cfg["model_widths"] = [int(item.strip()) for item in str(args.model_widths).split(",") if item.strip()]
    if args.num_rules is not None:
        cfg["num_rules"] = int(args.num_rules)
    if args.overlap_mode is not None:
        cfg["overlap_mode"] = str(args.overlap_mode)
    if args.tversky_alpha is not None:
        cfg["tversky_alpha"] = float(args.tversky_alpha)
    if args.tversky_beta is not None:
        cfg["tversky_beta"] = float(args.tversky_beta)
    if args.aux_loss_weight is not None:
        cfg["aux_loss_weight"] = float(args.aux_loss_weight)
    if args.boundary_loss_weight is not None:
        cfg["boundary_loss_weight"] = float(args.boundary_loss_weight)
    if args.topology_loss_weight is not None:
        cfg["topology_loss_weight"] = float(args.topology_loss_weight)
    if args.topology_num_iters is not None:
        cfg["topology_num_iters"] = int(args.topology_num_iters)
    if args.bce_pos_weight is not None:
        cfg["bce_pos_weight"] = float(args.bce_pos_weight)
    if args.drive_foreground_bias is not None:
        cfg["drive_foreground_bias"] = float(args.drive_foreground_bias)
    if args.retinal_patch_size is not None:
        cfg["retinal_patch_size"] = int(args.retinal_patch_size)
    if args.retinal_foreground_bias is not None:
        cfg["retinal_foreground_bias"] = float(args.retinal_foreground_bias)
    if args.encoder_az_stages is not None:
        cfg["encoder_az_stages"] = int(args.encoder_az_stages)
    if args.encoder_block_mode is not None:
        cfg["encoder_block_mode"] = str(args.encoder_block_mode)
    if args.hybrid_mix_init is not None:
        cfg["hybrid_mix_init"] = float(args.hybrid_mix_init)
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = str(args.init_checkpoint)
    if args.init_checkpoint_strict:
        cfg["init_checkpoint_strict"] = True
    if args.lr_scheduler is not None:
        cfg["lr_scheduler"] = str(args.lr_scheduler)
    if args.lr_min is not None:
        cfg["lr_min"] = float(args.lr_min)
    if args.lr_scheduler_t_max is not None:
        cfg["lr_scheduler_t_max"] = int(args.lr_scheduler_t_max)
    if args.lr_plateau_factor is not None:
        cfg["lr_plateau_factor"] = float(args.lr_plateau_factor)
    if args.lr_plateau_patience is not None:
        cfg["lr_plateau_patience"] = int(args.lr_plateau_patience)
    if args.lr_plateau_threshold is not None:
        cfg["lr_plateau_threshold"] = float(args.lr_plateau_threshold)
    if args.eval_tta is not None:
        cfg["eval_tta"] = str(args.eval_tta)
    if args.eval_threshold_start is not None:
        cfg["eval_threshold_start"] = float(args.eval_threshold_start)
    if args.eval_threshold_end is not None:
        cfg["eval_threshold_end"] = float(args.eval_threshold_end)
    if args.eval_threshold_step is not None:
        cfg["eval_threshold_step"] = float(args.eval_threshold_step)
    if args.eval_threshold_metric is not None:
        cfg["eval_threshold_metric"] = str(args.eval_threshold_metric)
    cfg["search_beats_baseline"] = bool(args.search_beats_baseline)
    cfg["search_selection_metric"] = str(args.search_selection_metric)
    cfg["comparison_min_delta"] = float(args.comparison_min_delta)

    results_dir = ensure_dir(cfg.get("results_dir", "./results"))
    comparison_metrics = (
        [item.strip() for item in args.comparison_metrics.split(",") if item.strip()]
        if args.comparison_metrics
        else list(utils.DRIVE_SUPERIORITY_METRICS)
    )
    variants: List[str]
    if args.canonical_drive:
        variants = list(utils.CANONICAL_DRIVE_VARIANTS)
    elif args.variants:
        variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    else:
        variants = [args.variant or cfg.get("variant", "baseline")]

    invalid = [variant for variant in variants if variant not in utils.VARIANTS]
    if invalid:
        raise SystemExit(f"Unknown variant(s) {invalid}. Choose from {utils.VARIANTS}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = cfg.get("run_name")
    if len(variants) > 1 and not run_prefix:
        run_prefix = f"multi_{timestamp}"

    all_metrics: List[Dict[str, Any]] = []
    failed_superiority_variants: List[str] = []
    for idx, variant in enumerate(variants):
        if len(variants) == 1:
            run_name = run_prefix
            if not run_name:
                run_name = f"{variant}_{timestamp}"
        else:
            run_name = f"{run_prefix}_{variant}"

        run_dir = ensure_dir(results_dir / run_name)
        run_cfg = dict(cfg)
        run_cfg["variant"] = variant

        print(f"[run {idx + 1}/{len(variants)}] variant={variant} -> {run_dir.name}")
        metrics = run_training(run_cfg, variant, run_dir)
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

        if metrics.get("test_threshold_dominance_search") is not None:
            print(format_drive_threshold_search_report(metrics["test_threshold_dominance_search"]))

        if (
            args.require_beats_baseline
            and metrics.get("task") == "segmentation"
            and str(metrics.get("dataset", "")).lower().replace("-", "_") == "drive"
            and variant != "baseline"
        ):
            try:
                report = compare_drive_metrics_to_baseline(
                    results_dir=results_dir,
                    candidate_metrics=metrics,
                    metric_names=comparison_metrics,
                    min_delta=float(args.comparison_min_delta),
                )
            except ValueError as exc:
                raise SystemExit(f"Cannot evaluate baseline superiority gate for {variant}: {exc}") from exc
            print(format_drive_superiority_report(report))
            if not report["all_passed"]:
                failed_superiority_variants.append(variant)

    if len(all_metrics) > 1:
        print("\nMulti-run summary:")
        for metrics in all_metrics:
            if metrics["task"] == "segmentation":
                print(
                    f"- {metrics['variant']}: "
                    f"dice={metrics.get('test_dice', 0.0):.4f}, "
                    f"iou={metrics.get('test_iou', 0.0):.4f}, "
                    f"thr={metrics.get('selected_threshold', metrics.get('seg_threshold_base', 'n/a'))}"
                )
            else:
                print(f"- {metrics['variant']}: acc={metrics.get('test_accuracy', 0.0):.2f}%")

    if failed_superiority_variants:
        failed = ", ".join(failed_superiority_variants)
        raise SystemExit(f"Baseline superiority gate failed for: {failed}")


if __name__ == "__main__":
    main()
