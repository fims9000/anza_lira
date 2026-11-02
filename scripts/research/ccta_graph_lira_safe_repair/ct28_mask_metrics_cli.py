#!/usr/bin/env python3
"""Post-hoc evaluation of paired 3D masks. Never used for repair inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ct28_mask_metrics import evaluate


HELD_OUT = {954, 958, 972, 973, 980, 984}


def run(*, patient: int, before: Path, after: Path, reference: Path,
        affine: Path, output: Path) -> dict:
    if patient in HELD_OUT:
        raise ValueError("Held-out CT28 evaluation forbidden before protocol freeze")
    if output.exists() or output.resolve() in {
        before.resolve(), after.resolve(), reference.resolve(), affine.resolve()
    }:
        raise ValueError("Never overwrite evaluation inputs or existing results")
    masks = [np.load(path, allow_pickle=False) for path in (before, after, reference)]
    metadata = evaluate(*masks, np.load(affine, allow_pickle=False))
    metadata["scan_id"] = patient
    metadata["phase"] = "train_validation_posthoc_only"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", type=int, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
