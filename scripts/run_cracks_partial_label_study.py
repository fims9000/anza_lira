#!/usr/bin/env python3
"""Run the frozen CRACKS T1 matrix without accessing expert annotations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cracks_experiment.partial_label_evaluation import (
    T1_ROOT,
    build_t1_statistics,
    evaluate_t0_control,
    evaluate_t1_run,
)
from cracks_experiment.partial_label_training import (
    T1_PROTOCOL,
    run_t1_training,
    t1_matrix,
    t1_protocol_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dry-run", "train", "evaluate", "statistics", "full"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-sections", type=int)
    args = parser.parse_args()
    matrix = t1_matrix()
    if args.stage == "dry-run":
        print(json.dumps({
            "protocol_sha256": t1_protocol_hash(),
            "expert": T1_PROTOCOL["expert"],
            "runs": [asdict(spec) | {"run_hash": spec.run_hash} for spec in matrix],
        }, indent=2, sort_keys=True))
        return 0
    if args.stage in {"train", "full"}:
        for spec in matrix:
            run_t1_training(
                spec,
                T1_ROOT,
                device=args.device,
                epochs=args.epochs,
                max_train_sections=args.max_train_sections,
            )
    if args.stage in {"evaluate", "full"}:
        if args.epochs is not None or args.max_train_sections is not None:
            raise ValueError("Full held-out evaluation refuses development-limited training flags")
        for model in ("unet", "anza_v1"):
            for seed in (41, 42, 43):
                evaluate_t0_control(model, seed, device=args.device)
        for spec in matrix:
            evaluate_t1_run(spec, device=args.device)
    if args.stage in {"statistics", "full"}:
        if args.epochs is not None or args.max_train_sections is not None:
            raise ValueError("Final statistics refuse development-limited training flags")
        result = build_t1_statistics()
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
