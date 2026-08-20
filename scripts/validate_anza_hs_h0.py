#!/usr/bin/env python3
"""Freeze and validate ANZA-HS H0 before any gradient step."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anza_hs.operators import ANZAHyperbolicConv, GenericAnisoConv, IsotropicOrientConv
from anza_hs.orientation_bank import orientation_bank_targets
from anza_hs.protocol import H0_ROOT, canonical_hash, freeze_protocol
from anza_hs.stress_bench import CASES, DEV_CALIBRATION_COUNT, SPLIT_SIZE, freeze_stressbench, generate_stress_sample


def validate() -> dict:
    failures = []
    protocol = freeze_protocol(H0_ROOT)
    benchmark = freeze_stressbench(H0_ROOT / "stressbench_v5.json")
    manifest = []
    for split in ("train", "dev"):
        for index, case in enumerate(CASES):
            sample = generate_stress_sample(split, index)
            target, valid = orientation_bank_targets(sample)
            manifest.append({
                "split": split, "index": index, "case": case,
                "image_sha256": hashlib.sha256(sample["image"].tobytes()).hexdigest(),
                "visible_pixels": int(sample["visible_fault_mask"].sum()),
                "orientation_valid_pixels": int(valid.sum()), "target_max": float(target.max()),
            })
            if sample["case"] != case or not np.isfinite(sample["image"]).all() or target.shape != (8, 64, 64):
                failures.append(f"invalid generated sample: {split}/{case}")
    layers = [IsotropicOrientConv(2), GenericAnisoConv(2), ANZAHyperbolicConv(2)]
    for layer in layers:
        kernels = layer.kernels()
        if not torch.isfinite(kernels).all() or not torch.allclose(kernels.sum((-2, -1)), torch.ones(8), atol=1e-6):
            failures.append(f"invalid kernels: {layer.kind}")
    generic, hyperbolic = layers[1], layers[2]
    if not torch.allclose(generic.kernels(), hyperbolic.kernels(), atol=1e-6):
        failures.append("generic comparator cannot reproduce frozen ANZA initialization")
    if protocol["confirm_opened"] or protocol["cracks_accessed"] or protocol["expert_accessed"]:
        failures.append("downstream access lock violated")
    if benchmark["dev_calibration"] != f"dev[0:{DEV_CALIBRATION_COUNT}]" or SPLIT_SIZE["dev"] != 264:
        failures.append("dev partition drift")
    result = {
        "status": "PASS" if not failures else "FAIL", "research_status": "ANZA_HS_H0_PASS" if not failures else "ANZA_HS_H0_INVALID",
        "failures": failures, "protocol_sha256": canonical_hash(protocol), "stressbench_sha256": benchmark["sha256"],
        "manifest": manifest, "training_performed": False, "confirm_opened": False, "cracks_accessed": False,
        "continuation_trained": False, "expert_accessed": False, "H1_authorized": not failures,
    }
    (H0_ROOT / "data_access_log.json").write_text(json.dumps({"train_manifest": True, "dev_manifest": True, "confirm": False, "cracks": False, "expert": False}, indent=2, sort_keys=True) + "\n")
    (H0_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (H0_ROOT / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    value = validate(); print(json.dumps(value, indent=2, sort_keys=True)); raise SystemExit(0 if value["status"] == "PASS" else 1)
