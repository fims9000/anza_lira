#!/usr/bin/env python3
"""Thin orchestration entry point for the frozen C0--C3 repair cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_repair.matrix import affinity_matrix, affinity_protocol_hash, freeze_affinity_protocol
from affinity_repair.training import run_candidate
from affinity_repair.validation import evaluate_candidate, write_affinity_gate
from affinity_repair.reporting import build_affinity_report, build_qualitative_figures, build_zip
from synthetic.crossing_trace_bench_v4 import freeze_benchmark_v4_config


RESULT_ROOT = ROOT / "results" / "affinity_repair"


def train(device: str) -> None:
    freeze_benchmark_v4_config(RESULT_ROOT / "benchmark_v4_config.json")
    freeze_affinity_protocol(RESULT_ROOT / "protocol.json")
    clean_checkpoint = None
    for spec in affinity_matrix():
        result = run_candidate(
            spec,
            RESULT_ROOT / "development",
            device=device,
            clean_checkpoint=clean_checkpoint,
        )
        print(f"candidate={spec.candidate_id} train={result['status']} action={result['action']}")
        if spec.candidate_id == "C1":
            clean_checkpoint = RESULT_ROOT / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"


def status() -> None:
    print(f"protocol_hash={affinity_protocol_hash()}")
    for spec in affinity_matrix():
        path = RESULT_ROOT / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "status.json"
        if not path.exists():
            print(f"{spec.candidate_id} MISSING")
            continue
        payload = json.loads(path.read_text())
        print(f"{spec.candidate_id} {payload['status']} stage={payload.get('stage')} epoch={payload.get('epoch', 0)}")


def evaluate(device: str) -> None:
    for spec in affinity_matrix():
        result = evaluate_candidate(
            spec, RESULT_ROOT / "development", RESULT_ROOT / "validation", device=device
        )
        metrics = result["metrics"]
        print(
            f"candidate={spec.candidate_id} dice={metrics['visible_dice']:.4f} "
            f"cldice={metrics['visible_cldice']:.4f} bridge={metrics['false_bridge_rate']:.4f} "
            f"hard_ap={metrics['hard_affinity_macro_ap']}"
        )


def gate() -> dict:
    result = write_affinity_gate(RESULT_ROOT / "validation", RESULT_ROOT / "mechanism_gate.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def report(device: str = "cuda") -> dict:
    result = build_affinity_report(RESULT_ROOT)
    build_qualitative_figures(RESULT_ROOT, device=device)
    package = build_zip(RESULT_ROOT)
    print(json.dumps({**result, **package}, indent=2, sort_keys=True))
    return {**result, **package}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "gate", "report", "status", "full"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        train(args.device)
    elif args.command == "evaluate":
        evaluate(args.device)
    elif args.command == "gate":
        gate()
    elif args.command == "report":
        report(args.device)
    elif args.command == "full":
        train(args.device)
        evaluate(args.device)
        result = gate()
        if result["status"] != "AFFINITY_MECHANISM_PASS":
            report(args.device)
            print("CONFIRM=NOT_AUTHORIZED CRACKS=NOT_AUTHORIZED")
            return 2
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
