#!/usr/bin/env python3
"""Build and train the single frozen CRACKS real-domain pair classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.cracks_pair_training import build_real_pair_dataset, train_real_pair_classifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dataset", "train", "full"), default="full", nargs="?")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.stage in {"dataset", "full"}:
        result = build_real_pair_dataset(device=args.device)
        print(json.dumps({key: result[key] for key in (
            "status", "protocol_sha256", "dataset_sha256", "train_matched_pairs",
            "validation_matched_pairs", "section_disjoint", "expert_data_accessed",
        )}, indent=2, sort_keys=True))
    if args.stage in {"train", "full"}:
        print(json.dumps(train_real_pair_classifier(device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
