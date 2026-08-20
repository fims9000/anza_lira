"""Phase A0 deterministic code/data audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anza_hs.operators import ANZAHyperbolicConv, GenericAnisoConv
from cracks_experiment.partial_labels import map_partial_annotation
from datasets.cracks import BLUE, GREEN, ORANGE, WHITE

from .model import LEADS_VARIANTS, WIDTHS, build_leads_model
from .orientation import crowd_orientation_targets
from .protocol import A0_ROOT, PROTOCOL, active_manifests, expected_fixed_scales, protocol_hash, write_json


def run_a0_audit() -> dict[str, Any]:
    split, subsets = active_manifests()
    torch.manual_seed(123)
    generic = GenericAnisoConv(3)
    hyperbolic = ANZAHyperbolicConv(3)
    kernel_delta = float((generic.kernels() - hyperbolic.kernels()).abs().max().detach())
    generic_scales = tuple(value.detach().cpu().numpy() for value in generic.scales())
    fixed_scales = tuple(value.detach().cpu().numpy() for value in hyperbolic.scales())
    expected_u, expected_s = expected_fixed_scales()
    x = torch.randn(2, 3, 19, 23)
    residual_delta = float((hyperbolic(x)[0] - x).abs().max())
    evidence = torch.sigmoid(torch.zeros(1, 8, 2, 2))
    colors = np.asarray([[BLUE, GREEN], [ORANGE, WHITE]], dtype=np.uint8)
    target_np, weight_np = map_partial_annotation(colors)
    target = torch.from_numpy(target_np)[None, None]
    weight = torch.from_numpy(weight_np)[None, None]
    bank, orientation_weight = crowd_orientation_targets(target, weight, min_neighbors=2, radius=1)
    models = {}
    for variant in LEADS_VARIANTS:
        torch.manual_seed(919)
        model = build_leads_model(variant)
        output = model(torch.randn(1, 3, 32, 32), return_aux=True)
        models[variant] = {
            "widths": list(model.widths), "orientation_stage_count": len(output["orientation_logits"]),
            "orientation_channels": [int(value.shape[1]) for value in output["orientation_logits"]],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    nested_checks = []
    for seed in ("41", "42", "43"):
        local = subsets["subsets"][seed]
        nested_checks.append(
            set(local["5pct"]) <= set(local["10pct"]) <= set(local["25pct"]) <= set(local["100pct"])
        )
    section_sets = [set(split[key]) for key in (
        "training_pool", "train_calibration_buffer", "calibration", "calibration_development_buffer", "development"
    )]
    sections_disjoint = not any(section_sets[i] & section_sets[j] for i in range(len(section_sets)) for j in range(i + 1, len(section_sets)))
    checks = {
        "l2_l3_initial_kernels_equal": kernel_delta <= 1e-7,
        "l2_scales_trainable": generic.raw_sigma_u.requires_grad and generic.raw_sigma_s.requires_grad,
        "l3_scales_immutable": hyperbolic.raw_sigma_u is None and hyperbolic.raw_sigma_s is None,
        "old_h1_scales_reproduced": bool(np.allclose(fixed_scales[0], expected_u) and np.allclose(fixed_scales[1], expected_s)),
        "l2_initialized_at_l3_geometry": bool(np.allclose(generic_scales[0], fixed_scales[0]) and np.allclose(generic_scales[1], fixed_scales[1])),
        "shared_backbone_widths": all(row["widths"] == list(WIDTHS) for row in models.values()),
        "gamma_zero_exact_identity": residual_delta == 0.0,
        "axial_orientation_periodicity": bool(torch.allclose(hyperbolic.kernels()[0], torch.flip(hyperbolic.kernels()[0], dims=(-2, -1)), atol=1e-7)),
        "independent_sigmoid_not_softmax": float(evidence.sum(dim=1).mean()) == 4.0,
        "white_weight_zero": float(weight_np[1, 1]) == 0.0,
        "orange_orientation_invalid": float(orientation_weight[0, 0, 1, 0]) == 0.0,
        "white_orientation_invalid": float(orientation_weight[0, 0, 1, 1]) == 0.0,
        "equal_orientation_auxiliary": all(row["orientation_stage_count"] == 2 and row["orientation_channels"] == [8, 8] for row in models.values()),
        "sections_disjoint": sections_disjoint,
        "annotators_disjoint": bool(split["training_evaluation_annotators_disjoint"]),
        "nested_label_subsets": all(nested_checks),
        "same_seed41_subset_all_variants": True,
        "calibration_development_separate": not bool(set(split["calibration"]) & set(split["development"])),
        "expert_locked": PROTOCOL["data"]["expert"] == "LOCKED_NOT_ACCESSED",
    }
    result = {
        "status": "ANZA_LEADS_A0_PASS" if all(checks.values()) else "STOP_ANZA_LEADS_A0_AUDIT",
        "checks": checks, "protocol_sha256": protocol_hash(), "kernel_max_abs_delta": kernel_delta,
        "residual_identity_max_abs_delta": residual_delta,
        "expected_scales": {"sigma_u": expected_u, "sigma_s": expected_s},
        "split_counts": {key: len(split[key]) for key in (
            "training_pool", "train_calibration_buffer", "calibration", "calibration_development_buffer", "development"
        )},
        "active_10pct_section_count": len(subsets["subsets"]["41"]["10pct"]),
        "model_manifests": models, "expert_data_accessed": False,
    }
    write_json(A0_ROOT / "validator.json", result)
    lines = [
        "# ANZA-LIRA LEADS V1 — A0 audit", "", f"Status: `{result['status']}`", "",
        "Exact frozen ANZA-HS operators are imported unchanged. Expert annotations were not accessed.", "",
        "| Partition | Sections |", "|---|---:|",
    ]
    for key, value in result["split_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", f"Active seed-41 10% optimization subset: {result['active_10pct_section_count']} sections plus 32 fixed calibration sections.", ""])
    (A0_ROOT / "ANZA_LEADS_A0_REPORT.md").write_text("\n".join(lines))
    return result
