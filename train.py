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
    build_dataloaders,
    build_model,
    estimate_drive_pos_weight,
    ensure_dir,
    load_config,
    measure_inference_time,
    save_json,
    segmentation_objective,
    segmentation_metrics_from_counts,
    set_seed,
    spatial_shape_for_dataset,
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
    return (
        f"{split}_dice={metrics[f'{split}_dice']:.4f} "
        f"{split}_iou={metrics[f'{split}_iou']:.4f}"
    )


def resolve_loss_cfg(
    cfg: Dict[str, Any],
    task: str,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    loss_cfg: Dict[str, Any] = {
        "bce_weight": float(cfg.get("bce_weight", 1.0)),
        "dice_weight": float(cfg.get("dice_weight", 1.0)),
        "threshold": float(cfg.get("seg_threshold", 0.5)),
        "aux_weight": float(cfg.get("aux_loss_weight", 0.2)),
        "boundary_weight": float(cfg.get("boundary_loss_weight", 0.1)),
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
            aux_weight=loss_cfg["aux_weight"],
            boundary_weight=loss_cfg["boundary_weight"],
        )
        loss_sum += loss.item() * x.size(0)
        bce_sum += aux["bce_loss"] * x.size(0)
        dice_loss_sum += aux["dice_loss"] * x.size(0)
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
                aux_weight=loss_cfg["aux_weight"],
                boundary_weight=loss_cfg["boundary_weight"],
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


def plot_histories(histories: Dict[str, List[Dict[str, Any]]], out_dir: Path, task: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, rows in histories.items():
        epochs = [r["epoch"] for r in rows]
        plt.plot(epochs, [r["train_loss"] for r in rows], label=f"{name} train", linestyle="--", alpha=0.8)
        plt.plot(epochs, [r["val_loss"] for r in rows], label=f"{name} val", alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / validation loss")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_loss.png", dpi=150)
    plt.close()

    metric_key = "val_acc" if task == "classification" else "val_dice"
    metric_label = "Accuracy (%)" if task == "classification" else "Dice"
    metric_title = "Validation accuracy" if task == "classification" else "Validation Dice"
    metric_file = "compare_val_acc.png" if task == "classification" else "compare_val_dice.png"

    plt.figure(figsize=(8, 5))
    for name, rows in histories.items():
        epochs = [r["epoch"] for r in rows]
        plt.plot(epochs, [r[metric_key] for r in rows], label=f"{name} {metric_key}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.title(metric_title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / metric_file, dpi=150)
    plt.close()


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


def summary_table_md(rows: List[Dict[str, Any]], path: Path, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if task == "classification":
        header = "| Variant | Params | Test acc (%) | Val acc best (%) | Train epoch (s) | Fwd batch (s) |\n"
        sep = "|---|---:|---:|---:|---:|---:|\n"
        lines = [header, sep]
        for row in sorted(rows, key=lambda x: x.get("variant", "")):
            lines.append(
                f"| {row.get('variant', '')} | {row.get('num_parameters', 0)} | "
                f"{row.get('test_accuracy', 0):.2f} | {row.get('best_val_accuracy', 0):.2f} | "
                f"{row.get('seconds_per_train_epoch_mean', 0):.3f} | {row.get('seconds_per_forward_batch', 0):.5f} |\n"
            )
    else:
        header = (
            "| Variant | Params | Test Dice | Test IoU | Val Dice best | "
            "Train epoch (s) | Fwd batch (s) |\n"
        )
        sep = "|---|---:|---:|---:|---:|---:|---:|\n"
        lines = [header, sep]
        for row in sorted(rows, key=lambda x: x.get("variant", "")):
            lines.append(
                f"| {row.get('variant', '')} | {row.get('num_parameters', 0)} | "
                f"{row.get('test_dice', 0):.4f} | {row.get('test_iou', 0):.4f} | "
                f"{row.get('best_val_dice', 0):.4f} | {row.get('seconds_per_train_epoch_mean', 0):.3f} | "
                f"{row.get('seconds_per_forward_batch', 0):.5f} |\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


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
        "variant": variant,
        "dataset": cfg["dataset"],
        "task": task,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
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
        metrics["bce_pos_weight"] = loss_cfg["pos_weight_value"]
        metrics["best_val_dice"] = best_val
        metrics["test_bce_loss"] = test_metrics["val_bce_loss"]
        metrics["test_dice_loss"] = test_metrics["val_dice_loss"]
        metrics["test_dice"] = test_metrics["val_dice"]
        metrics["test_iou"] = test_metrics["val_iou"]
        metrics["test_precision"] = test_metrics["val_precision"]
        metrics["test_recall"] = test_metrics["val_recall"]
        metrics["test_specificity"] = test_metrics["val_specificity"]
        metrics["test_accuracy"] = test_metrics["val_accuracy"]
        metrics["test_balanced_accuracy"] = test_metrics["val_balanced_accuracy"]
    if history:
        metrics["final_anisotropy_gap"] = history[-1].get("reg_anisotropy_gap", 0.0)
    save_json(run_dir / "metrics.json", metrics)
    plot_single_run(history, run_dir, variant, task)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline or AZ variants on classification or DRIVE segmentation.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--variant", type=str, default=None, help="Override config variant")
    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="Sequentially train all variants and write comparison plots + table",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_results = ensure_dir(cfg.get("results_dir", "./results"))
    task = utils.task_for_dataset(cfg["dataset"])

    if args.all_variants:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_dir = base_results / f"multi_{stamp}"
        group_dir.mkdir(parents=True, exist_ok=True)

        all_metrics: List[Dict[str, Any]] = []
        histories: Dict[str, List[Dict[str, Any]]] = {}
        for variant in utils.VARIANTS:
            run_dir = group_dir / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = run_training(cfg, variant, run_dir)
            all_metrics.append(metrics)
            with open(run_dir / "history.json", "r", encoding="utf-8") as handle:
                histories[variant] = json.load(handle)

            score_name = metric_name_for_task(task)
            if task == "classification":
                score_value = metrics["test_accuracy"]
                print(f"Done {variant}: test_{score_name}={score_value:.2f}%")
            else:
                score_value = metrics["test_dice"]
                print(f"Done {variant}: test_{score_name}={score_value:.4f}")

        save_json(group_dir / "all_metrics.json", all_metrics)
        table_name = "accuracy_table.md" if task == "classification" else "segmentation_table.md"
        summary_table_md(all_metrics, group_dir / table_name, task)
        plot_histories(histories, group_dir, task)
        utils.write_interpretation(group_dir / "interpretation.txt", all_metrics)
        print(f"All variants finished. Outputs in {group_dir}")
        return

    variant = args.variant or cfg.get("variant", "baseline")
    if variant not in utils.VARIANTS:
        raise SystemExit(f"Unknown variant {variant}. Choose from {utils.VARIANTS}")

    run_name = cfg.get("run_name") or f"{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_results / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_training(cfg, variant, run_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
