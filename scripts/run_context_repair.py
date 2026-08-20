#!/usr/bin/env python3
"""Resumable entry point for the frozen B0-B3 context-repair cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method_repair.context_matrix import context_matrix, context_protocol_hash
from method_repair.context_training import run_context_candidate
from method_repair.context_validation import evaluate_context_candidate, write_context_gate


RESULT_ROOT = ROOT / "results" / "context_repair"


def train(device: str) -> None:
    for spec in context_matrix():
        result = run_context_candidate(spec, RESULT_ROOT / "development", device=device)
        print(f"candidate={spec.candidate_id} train={result['status']} action={result['action']}")


def evaluate(device: str) -> None:
    for spec in context_matrix():
        result = evaluate_context_candidate(
            spec,
            RESULT_ROOT / "development",
            RESULT_ROOT / "validation",
            device=device,
        )
        metrics = result["metrics"]
        print(
            f"candidate={spec.candidate_id} dice={metrics['visible_dice']:.4f} "
            f"route_ap={metrics['route_average_precision']:.4f} "
            f"gate_auc={metrics['gate_auroc']:.4f} bridge={metrics['false_bridge_rate']:.4f}"
        )


def gate() -> dict:
    result = write_context_gate(
        RESULT_ROOT / "validation", RESULT_ROOT / "mechanism_gate.json"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def status() -> None:
    print(f"protocol_hash={context_protocol_hash()}")
    for spec in context_matrix():
        path = RESULT_ROOT / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "status.json"
        if not path.exists():
            print(f"{spec.candidate_id} MISSING")
            continue
        payload = json.loads(path.read_text())
        print(
            f"{spec.candidate_id} {payload['status']} "
            f"epoch={payload.get('epoch', 0)}/{payload.get('epoch_budget', 25)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "gate", "status", "full"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        train(args.device)
    elif args.command == "evaluate":
        evaluate(args.device)
    elif args.command == "gate":
        gate()
    elif args.command == "status":
        status()
    else:
        train(args.device)
        evaluate(args.device)
        result = gate()
        if result["status"] != "CONTEXT_MECHANISM_PASS":
            print("CRACKS=NOT_AUTHORIZED")
            return 2
        print("CONFIRM_V3=AUTHORIZED CRACKS=STILL_LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
