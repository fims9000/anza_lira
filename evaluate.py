#!/usr/bin/env python3
"""Load a checkpoint and report test metrics + approximate inference time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import utils
from utils import build_dataloaders, build_model, count_parameters, load_config, measure_inference_time, set_seed


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

    set_seed(int(cfg.get("seed", 42)))
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, in_c, num_classes = build_dataloaders(cfg)
    model = build_model(
        variant,
        num_classes=num_classes,
        in_channels=in_c,
        num_rules=int(cfg.get("num_rules", 4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    loss_sum = 0.0
    n = 0
    correct = 0
    for x, y in test_loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    wall = time.perf_counter() - t0

    spatial = 28 if str(cfg["dataset"]).lower().replace("-", "_") in ("fashion_mnist", "fashionmnist") else 32
    sec_batch = measure_inference_time(
        model,
        dev,
        batch_size=int(cfg["batch_size"]),
        in_channels=in_c,
        spatial=spatial,
    )

    return {
        "variant": variant,
        "test_loss": loss_sum / max(n, 1),
        "test_accuracy": 100.0 * correct / max(n, 1),
        "test_wall_seconds": wall,
        "seconds_per_forward_batch": sec_batch,
        "num_parameters": count_parameters(model),
    }


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

