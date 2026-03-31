#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils
from utils import ensure_dir, load_config, save_json, update_segmentation_multiseed_summary


def _parse_seeds(text: str) -> List[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def _run_one_stage(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    variant = str(cfg["variant"])
    return train.run_training(cfg, variant, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FIVES -> CHASE transfer experiments across multiple seeds.")
    parser.add_argument("--pretrain-config", type=str, default="configs/fives_pretrain_probe16_cont.yaml")
    parser.add_argument("--finetune-config", type=str, default="configs/chase_db1_from_fives16_continue_dice_ft20.yaml")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    args = parser.parse_args()

    pretrain_cfg = load_config(args.pretrain_config)
    finetune_cfg = load_config(args.finetune_config)
    seeds = _parse_seeds(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.run_name or f"retinal_transfer_{timestamp}"
    session_dir = ensure_dir(Path(args.results_dir) / session_name)
    pretrain_dir = ensure_dir(session_dir / "fives_pretrain")
    finetune_dir = ensure_dir(session_dir / "chase_finetune")

    all_metrics: List[Dict[str, Any]] = []

    print(f"Session directory: {session_dir}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    for seed in seeds:
        pre_cfg = dict(pretrain_cfg)
        pre_run_name = f"fives_seed{seed}"
        pre_run_dir = ensure_dir(pretrain_dir / pre_run_name)
        pre_cfg["seed"] = int(seed)
        pre_cfg["run_name"] = pre_run_name
        pre_cfg["results_dir"] = str(pretrain_dir)
        if args.device is not None:
            pre_cfg["device"] = args.device
        if args.pretrain_epochs is not None:
            pre_cfg["epochs"] = int(args.pretrain_epochs)

        print(f"[seed {seed}] pretrain -> {pre_run_dir}")
        pre_metrics = _run_one_stage(pre_cfg, pre_run_dir)
        all_metrics.append({"stage": "pretrain", **pre_metrics})

        ft_cfg = dict(finetune_cfg)
        ft_run_name = f"chase_seed{seed}"
        ft_run_dir = ensure_dir(finetune_dir / ft_run_name)
        ft_cfg["seed"] = int(seed)
        ft_cfg["run_name"] = ft_run_name
        ft_cfg["results_dir"] = str(finetune_dir)
        ft_cfg["init_checkpoint"] = str(pre_run_dir / "checkpoint_best.pt")
        if args.device is not None:
            ft_cfg["device"] = args.device
        if args.finetune_epochs is not None:
            ft_cfg["epochs"] = int(args.finetune_epochs)

        print(f"[seed {seed}] finetune -> {ft_run_dir}")
        ft_metrics = _run_one_stage(ft_cfg, ft_run_dir)
        all_metrics.append({"stage": "finetune", **ft_metrics})

    save_json(session_dir / "all_metrics.json", all_metrics)
    pretrain_summary = update_segmentation_multiseed_summary(pretrain_dir, dataset=pretrain_cfg["dataset"], variants=[pretrain_cfg["variant"]])
    finetune_summary = update_segmentation_multiseed_summary(finetune_dir, dataset=finetune_cfg["dataset"], variants=[finetune_cfg["variant"]])

    session_summary = {
        "session_dir": str(session_dir),
        "seeds": seeds,
        "pretrain_dataset": pretrain_cfg["dataset"],
        "finetune_dataset": finetune_cfg["dataset"],
        "pretrain_summary": str(pretrain_summary),
        "finetune_summary": str(finetune_summary),
    }
    (session_dir / "session_summary.json").write_text(json.dumps(session_summary, indent=2), encoding="utf-8")

    print(f"Wrote all metrics to {session_dir / 'all_metrics.json'}")
    print(f"Wrote pretrain summary to {pretrain_summary}")
    print(f"Wrote finetune summary to {finetune_summary}")


if __name__ == "__main__":
    main()
