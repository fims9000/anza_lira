#!/usr/bin/env python3
"""Independent P2 classical segmentation benchmark (validation-select, fixed test)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.segmentation_study import SegmentationProtocol, run_segmentation_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-per-domain", type=int, default=12)
    parser.add_argument("--test-per-domain", type=int, default=18)
    args = parser.parse_args()
    protocol = SegmentationProtocol(val_scenes_per_domain=args.val_per_domain,
                                    test_scenes_per_domain=args.test_per_domain)
    result = run_segmentation_study(protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({method: {"selected": entry["selected_parameter"],
             "test": {domain: table["macro"] for domain, table in entry["test"].items()}}
             for method, entry in result["studies"].items()}, indent=2))

if __name__ == "__main__":
    main()
