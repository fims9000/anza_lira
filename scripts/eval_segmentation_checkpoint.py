#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils
from drive_viewer import recommended_threshold_for_run


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing segmentation checkpoint with optional TTA, without retraining.")
    parser.add_argument("--run-dir", required=True, type=str, help="Path to an existing run directory containing checkpoint_best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config path fallback if cfg is absent in the checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=("training", "test"))
    parser.add_argument("--eval-tta", type=str, default="none", help="none, flips, or d4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path for evaluated metrics.")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    payload = train.load_checkpoint_payload(checkpoint_path)
    cfg = payload.get("cfg")
    if cfg is None:
        if args.config is None:
            raise SystemExit("Checkpoint does not contain cfg and --config was not provided.")
        cfg = utils.load_config(args.config)
    cfg = dict(cfg)
    cfg["device"] = args.device
    cfg["eval_tta"] = str(args.eval_tta)

    utils.set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device(args.device)
    train_loader, _val_loader, test_loader, in_c, num_outputs, task = utils.build_dataloaders(cfg)
    if task != "segmentation":
        raise SystemExit(f"This helper only supports segmentation tasks, got: {task}")

    variant = str(payload.get("variant") or cfg["variant"])
    model = utils.build_model(
        variant,
        num_outputs=num_outputs,
        in_channels=in_c,
        num_rules=int(cfg.get("num_rules", 4)),
        task=task,
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict)
    eval_model = train.build_eval_model(model, task, cfg)
    loss_cfg = train.resolve_loss_cfg(cfg, task, train_loader, device)

    run_info = type(
        "RunInfo",
        (),
        {
            "run_dir": run_dir,
            "metrics_path": run_dir / "metrics.json",
        },
    )()
    threshold = float(args.threshold) if args.threshold is not None else recommended_threshold_for_run(run_info, default_threshold=float(cfg.get("seg_threshold", 0.6)))
    loss_cfg["threshold"] = threshold

    raw_metrics = train.evaluate_epoch(eval_model, test_loader, None, device, task, loss_cfg)
    metrics: dict[str, Any] = {}
    split_prefix = f"{args.split}_"
    for key, value in raw_metrics.items():
        if key.startswith("val_"):
            metrics[f"{split_prefix}{key[4:]}"] = value
        else:
            metrics[key] = value
    metrics["dice"] = metrics.get(f"{split_prefix}dice")
    metrics["iou"] = metrics.get(f"{split_prefix}iou")
    metrics["precision"] = metrics.get(f"{split_prefix}precision")
    metrics["recall"] = metrics.get(f"{split_prefix}recall")
    metrics["specificity"] = metrics.get(f"{split_prefix}specificity")
    metrics["balanced_accuracy"] = metrics.get(f"{split_prefix}balanced_accuracy")
    metrics.update(
        {
            "run_dir": str(run_dir),
            "variant": variant,
            "dataset": cfg["dataset"],
            "split": args.split,
            "eval_tta": str(args.eval_tta),
            "selected_threshold": threshold,
        }
    )

    output_path = (PROJECT_ROOT / args.output).resolve() if args.output else run_dir / f"metrics_{args.split}_{str(args.eval_tta).lower().strip() or 'none'}.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
