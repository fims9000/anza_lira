#!/usr/bin/env python3
"""Train baseline or AZ variants on classification or DRIVE segmentation tasks."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
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
    spatial_shape_for_dataset,
    sweep_segmentation_thresholds,
    update_drive_comparison_summary,
)

REG_LOG_KEYS = [
    "membership_entropy",
    "membership_smoothness",
    "geometry_smoothness",
    "hyperbolicity_penalty",
    "anisotropy_gap",
]


def metric_name_for_task(task: str) -> str:
    return "accuracy" if task == "classification" else "dice"


def selection_key_for_task(task: str) -> str:
    return "val_acc" if task == "classification" else "val_dice"


def summarize_score(task: str, split: str, metrics: Dict[str, float]) -> str:
    if task == "classification":
        return f"{split}_acc={metrics[f'{split}_acc']:.2f}%"
    return f"{split}_dice={metrics[f'{split}_dice']:.4f} {split}_iou={metrics[f'{split}_iou']:.4f}"


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
        "topology_weight": float(cfg.get("topology_loss_weight", 0.0)),
        "topology_num_iters": int(cfg.get("topology_num_iters", 10)),
        "pos_weight": None,
        "pos_weight_value": 1.0,
    }
    if task != "segmentation":
        return loss_cfg

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

    loss_cfg["pos_weight_value"] = pos_weight_value
    loss_cfg["pos_weight"] = torch.tensor(pos_weight_value, device=device)
    return loss_cfg


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
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
            pos_weight=loss_cfg["pos_weight"],
            overlap_mode=loss_cfg["overlap_mode"],
            tversky_alpha=loss_cfg["tversky_alpha"],
            tversky_beta=loss_cfg["tversky_beta"],
            aux_weight=loss_cfg["aux_weight"],
            boundary_weight=loss_cfg["boundary_weight"],
            topology_weight=loss_cfg["topology_weight"],
            topology_num_iters=loss_cfg["topology_num_iters"],
        )
        loss_sum += loss.item() * x.size(0)
        bce_sum += aux["bce_loss"] * x.size(0)
        dice_loss_sum += aux["dice_loss"] * x.size(0)
        topology_sum += aux["topology_loss"] * x.size(0)
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
        n += x.size(0)

    metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
    return {
        "val_loss": loss_sum / max(n, 1),
        "val_bce_loss": bce_sum / max(n, 1),
        "val_dice_loss": dice_loss_sum / max(n, 1),
        "val_topology_loss": topology_sum / max(n, 1),
        "val_dice": metrics["dice"],
        "val_iou": metrics["iou"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_specificity": metrics["specificity"],
        "val_accuracy": metrics["accuracy"],
        "val_balanced_accuracy": metrics["balanced_accuracy"],
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
        tp = fp = tn = fn = 0.0
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
                topology_weight=loss_cfg["topology_weight"],
                topology_num_iters=loss_cfg["topology_num_iters"],
            )
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
            bce_sum += aux["bce_loss"] * x.size(0)
            dice_loss_sum += aux["dice_loss"] * x.size(0)
            aux_sum += aux["aux_loss"] * x.size(0)
            boundary_sum += aux["boundary_loss"] * x.size(0)
            topology_sum += aux["topology_loss"] * x.size(0)
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
            n += x.size(0)

        seg_metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
        metrics = {
            "train_loss": loss_sum / max(n, 1),
            "train_objective": objective_sum / max(n, 1),
            "train_bce_loss": bce_sum / max(n, 1),
            "train_dice_loss": dice_loss_sum / max(n, 1),
            "train_aux_loss": aux_sum / max(n, 1),
            "train_boundary_loss": boundary_sum / max(n, 1),
            "train_topology_loss": topology_sum / max(n, 1),
            "train_dice": seg_metrics["dice"],
            "train_iou": seg_metrics["iou"],
            "train_precision": seg_metrics["precision"],
            "train_recall": seg_metrics["recall"],
            "train_specificity": seg_metrics["specificity"],
            "train_accuracy": seg_metrics["accuracy"],
            "train_balanced_accuracy": seg_metrics["balanced_accuracy"],
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
    ).to(device)
    loss_cfg = resolve_loss_cfg(cfg, task, train_loader, device)

    n_params = utils.count_parameters(model)
    criterion = nn.CrossEntropyLoss() if task == "classification" else None
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    epochs = int(cfg["epochs"])
    history: List[Dict[str, Any]] = []
    selection_key = selection_key_for_task(task)
    best_val = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, reg_weights, task, loss_cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t0

        val_metrics = evaluate_epoch(model, val_loader, criterion, device, task, loss_cfg)

        row = {"epoch": epoch, "seconds_train_epoch": t_epoch}
        row.update(train_metrics)
        row.update(val_metrics)
        history.append(row)

        msg = (
            f"[{variant}] epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_obj={train_metrics['train_objective']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"{summarize_score(task, 'val', val_metrics)} "
            f"time={t_epoch:.1f}s"
        )
        if variant.startswith("az_"):
            msg += f" anis_gap={train_metrics['reg_anisotropy_gap']:.4f}"
        print(msg)

        current_val = float(val_metrics[selection_key])
        if current_val > best_val:
            best_val = current_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "variant": variant, "cfg": cfg}, run_dir / "checkpoint_best.pt")

        with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)

    selected_threshold = float(loss_cfg["threshold"])
    threshold_selection_mode = "fixed"
    threshold_selection_metric = None
    threshold_sweep_rows: List[Dict[str, float]] = []
    val_selected_metrics: Dict[str, float] | None = None
    threshold_grid: List[float] = []

    if task == "segmentation" and bool(cfg.get("eval_threshold_sweep", True)):
        threshold_metric = str(cfg.get("eval_threshold_metric", "dice"))
        threshold_start = float(cfg.get("eval_threshold_start", 0.3))
        threshold_end = float(cfg.get("eval_threshold_end", 0.8))
        threshold_step = float(cfg.get("eval_threshold_step", 0.05))
        threshold_grid = build_threshold_grid(threshold_start, threshold_end, threshold_step)
        threshold_sweep_rows = sweep_segmentation_thresholds(model, val_loader, device, threshold_grid)
        best_threshold_row = select_best_threshold(
            threshold_sweep_rows,
            metric=threshold_metric,
            reference_threshold=selected_threshold,
        )
        selected_threshold = float(best_threshold_row["threshold"])
        loss_cfg["threshold"] = selected_threshold
        threshold_selection_mode = "val_sweep"
        threshold_selection_metric = threshold_metric
        val_selected_metrics = evaluate_epoch(model, val_loader, criterion, device, task, loss_cfg)

    test_metrics = evaluate_epoch(model, test_loader, criterion, device, task, loss_cfg)

    sec_batch = measure_inference_time(
        model,
        device,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial_shape=spatial_shape_for_dataset(cfg["dataset"]),
        warmup=int(cfg.get("timing_warmup", 3)),
        iters=int(cfg.get("timing_iters", 20)),
    )
    mean_train_epoch = float(sum(h["seconds_train_epoch"] for h in history) / max(len(history), 1))

    metrics: Dict[str, Any] = {
        "run_name": run_dir.name,
        "variant": variant,
        "dataset": cfg["dataset"],
        "task": task,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "seed": int(cfg["seed"]),
        "num_rules": int(cfg.get("num_rules", 4)),
        "num_parameters": n_params,
        "test_loss": test_metrics["val_loss"],
        "seconds_per_train_epoch_mean": mean_train_epoch,
        "seconds_per_forward_batch": sec_batch,
        "regularization_weights": reg_weights,
    }
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
        metrics["bce_pos_weight"] = loss_cfg["pos_weight_value"]
        metrics["best_val_dice"] = best_val
        metrics["seg_threshold_base"] = float(cfg.get("seg_threshold", 0.5))
        metrics["selected_threshold"] = selected_threshold
        metrics["threshold_selection_mode"] = threshold_selection_mode
        metrics["topology_loss_weight"] = float(cfg.get("topology_loss_weight", 0.0))
        metrics["topology_num_iters"] = int(cfg.get("topology_num_iters", 10))
        if threshold_selection_metric is not None:
            metrics["threshold_selection_metric"] = threshold_selection_metric
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
        if bool(cfg.get("search_beats_baseline", False)):
            search_metric = str(cfg.get("search_selection_metric", "dice"))
            search_grid = threshold_grid or build_threshold_grid(
                float(cfg.get("eval_threshold_start", 0.3)),
                float(cfg.get("eval_threshold_end", 0.8)),
                float(cfg.get("eval_threshold_step", 0.05)),
            )
            search_rows = sweep_segmentation_thresholds(model, test_loader, device, search_grid)
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
