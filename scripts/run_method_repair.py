#!/usr/bin/env python3
"""Run the bounded ANZA method-repair synthetic protocol without expert access."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method_repair.matrix import COMMON_PROTOCOL, protocol_hash, synthetic_matrix
from method_repair.training import run_synthetic_candidate
from method_repair.validation import evaluate_validation_candidate, write_mechanism_gate


DEFAULT_ROOT = PROJECT_ROOT / "results" / "method_repair" / "synthetic_v2"


def freeze_protocol(root: Path) -> Path:
    payload = {
        "status": "FROZEN_BEFORE_RESULTS",
        "protocol_hash": protocol_hash(),
        "common_protocol": COMMON_PROTOCOL,
        "matrix": [{**asdict(spec), "run_hash": spec.run_hash} for spec in synthetic_matrix()],
    }
    path = root / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("method-repair synthetic protocol drift after freeze")
    path.write_text(encoded)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dry-run", "train", "validate", "gate", "full"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--candidate", choices=tuple(spec.candidate_id for spec in synthetic_matrix()))
    args = parser.parse_args()
    freeze_protocol(args.root)
    selected = [spec for spec in synthetic_matrix() if args.candidate in {None, spec.candidate_id}]
    if args.phase == "dry-run":
        for spec in selected:
            print(
                f"{spec.candidate_id} model={spec.model} gate={int(spec.use_ambiguity_gate)} "
                f"direct={int(spec.direct_mode_supervision)} kernel={spec.routing_kernel_size} "
                f"seed={spec.seed} hash={spec.run_hash}"
            )
        print(f"protocol_hash={protocol_hash()} expert=LOCKED old_test=LOCKED new_test=LOCKED")
        return 0
    if args.phase in {"train", "full"}:
        for spec in selected:
            run_synthetic_candidate(spec, args.root / "development", device=args.device)
    if args.phase in {"validate", "full"}:
        for spec in selected:
            result = evaluate_validation_candidate(
                spec,
                args.root / "development",
                args.root / "validation",
                device=args.device,
            )
            metrics = result["metrics"]
            print(
                f"phase=method_repair_validation candidate={spec.candidate_id} "
                f"visible_dice={metrics['visible_dice']:.4f} "
                f"route_ap={metrics['route_average_precision']} "
                f"entropy={metrics['route_entropy_normalized']} status=COMPLETE"
            )
    if args.phase in {"gate", "full"}:
        if args.candidate is not None:
            raise ValueError("mechanism gate requires the complete A0-A4 matrix")
        gate = write_mechanism_gate(
            args.root / "validation",
            args.root / "mechanism_gate.json",
        )
        print(
            f"METHOD REPAIR SYNTHETIC: {gate['status']} "
            f"candidate={gate['selected_candidate']} cracks_authorized={gate['cracks_authorized']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
