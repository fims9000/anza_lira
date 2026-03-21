#!/usr/bin/env python3
"""Train baseline or AZConv variant; save checkpoints, logs, plots, and metrics."""

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
from utils import build_dataloaders, build_model, ensure_dir, load_config, measure_inference_time, save_json, set_seed


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
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
    return loss_sum / max(n, 1), 100.0 * correct / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    n = 0
    correct = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return loss_sum / max(n, 1), 100.0 * correct / max(n, 1)


def spatial_size_for_dataset(name: str) -> int:
    n = name.lower().replace("-", "_")
    return 28 if n in ("fashion_mnist", "fashionmnist") else 32


def plot_histories(histories: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, rows in histories.items():
        e = [r["epoch"] for r in rows]
        plt.plot(e, [r["train_loss"] for r in rows], label=f"{name} train", linestyle="--", alpha=0.8)
        plt.plot(e, [r["val_loss"] for r in rows], label=f"{name} val", alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / validation loss")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, rows in histories.items():
        e = [r["epoch"] for r in rows]
        plt.plot(e, [r["val_acc"] for r in rows], label=f"{name} val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_val_acc.png", dpi=150)
    plt.close()


def plot_single_run(history: List[Dict[str, Any]], out_dir: Path, variant: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    e = [r["epoch"] for r in history]

    plt.figure(figsize=(7, 4))
    plt.plot(e, [r["train_loss"] for r in history], label="train")
    plt.plot(e, [r["val_loss"] for r in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss — {variant}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(e, [r["train_acc"] for r in history], label="train acc")
    plt.plot(e, [r["val_acc"] for r in history], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy — {variant}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=150)
    plt.close()


def run_training(cfg: Dict[str, Any], variant: str, run_dir: Path) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device_str = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if variant.startswith("az_"):
        utils.sanity_check_azconv_forward(device)

    train_loader, val_loader, test_loader, in_c, num_classes = build_dataloaders(cfg)
    model = build_model(
        variant,
        num_classes=num_classes,
        in_channels=in_c,
        num_rules=int(cfg.get("num_rules", 4)),
    ).to(device)

    n_params = utils.count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    epochs = int(cfg["epochs"])
    history: List[Dict[str, Any]] = []
    best_val = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t0

        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds_train_epoch": t_epoch,
        }
        history.append(row)
        print(
            f"[{variant}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"time={t_epoch:.1f}s"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "variant": variant, "cfg": cfg}, run_dir / "checkpoint_best.pt")

        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)

    spatial = spatial_size_for_dataset(cfg["dataset"])
    sec_batch = measure_inference_time(
        model,
        device,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial=spatial,
        warmup=int(cfg.get("timing_warmup", 3)),
        iters=int(cfg.get("timing_iters", 20)),
    )
    mean_train_epoch = float(sum(h["seconds_train_epoch"] for h in history) / max(len(history), 1))

    metrics = {
        "variant": variant,
        "dataset": cfg["dataset"],
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "num_parameters": n_params,
        "best_val_accuracy": best_val,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "seconds_per_train_epoch_mean": mean_train_epoch,
        "seconds_per_forward_batch": sec_batch,
    }
    save_json(run_dir / "metrics.json", metrics)
    plot_single_run(history, run_dir, variant)
    return metrics


def accuracy_table_md(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Variant | Params | Test acc (%) | Val acc best (%) | Train epoch (s) | Fwd batch (s) |\\n"
    sep = "|---|---:|---:|---:|---:|---:|\\n"
    lines = [header, sep]
    for r in sorted(rows, key=lambda x: x.get("variant", "")):
        lines.append(
            f"| {r.get('variant','')} | {r.get('num_parameters',0)} | "
            f"{r.get('test_accuracy',0):.2f} | {r.get('best_val_accuracy',0):.2f} | "
            f"{r.get('seconds_per_train_epoch_mean',0):.3f} | {r.get('seconds_per_forward_batch',0):.5f} |\\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AZConv / baseline on CIFAR-10 or Fashion-MNIST")
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

    if args.all_variants:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_dir = base_results / f"multi_{stamp}"
        group_dir.mkdir(parents=True, exist_ok=True)

        all_metrics: List[Dict[str, Any]] = []
        histories: Dict[str, List[Dict[str, Any]]] = {}
        for v in utils.VARIANTS:
            run_dir = group_dir / v
            run_dir.mkdir(parents=True, exist_ok=True)
            m = run_training(cfg, v, run_dir)
            all_metrics.append(m)
            with open(run_dir / "history.json", "r", encoding="utf-8") as f:
                histories[v] = json.load(f)
            print(f"Done {v}: test_acc={m['test_accuracy']:.2f}%")

        save_json(group_dir / "all_metrics.json", all_metrics)
        accuracy_table_md(all_metrics, group_dir / "accuracy_table.md")
        plot_histories(histories, group_dir)
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

