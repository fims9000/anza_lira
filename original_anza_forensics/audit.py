"""Phase-0 operator forensics with fail-closed paper/code and split gates."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cracks_experiment.partial_label_training import T1_PROTOCOL, _model, load_t1_checkpoint, t1_matrix
from models.azconv import AZConv2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "original_anza_forensics" / "phase0"
LEGACY_SOURCE_SHA256 = "d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197"
PREVIOUS_PHASE_A_METRICS_SHA256 = "39ab64dc07eeec60ae89748e2fe53c9e42964a809376f6f6487b15b0f5f219f3"
SEEDS = (41, 42, 43)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frozen_run(seed: int) -> tuple[Any, Path]:
    spec = next(row for row in t1_matrix() if row.model == "anza_v1" and row.seed == int(seed))
    run = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_t1" / f"{spec.run_id}-{spec.run_hash}"
    status = json.loads((run / "status.json").read_text())
    if (
        status.get("status") != "COMPLETE"
        or status.get("expert_data_accessed") is not False
        or status.get("expert_scores_used") is not False
    ):
        raise PermissionError(f"Frozen T1 ANZA seed {seed} is not complete and expert-locked")
    return spec, run / "checkpoint-last.pt"


@torch.inference_mode()
def inspect_legacy_layer(layer: AZConv2d, inputs: torch.Tensor) -> dict[str, Any]:
    """Reconstruct the exact forward interaction without modifying the layer."""

    if layer.cfg.geometry_mode != "local_hyperbolic":
        raise ValueError("forensic fixture expects local_hyperbolic geometry")
    batch, channels, height, width = inputs.shape
    patch_area, locations = layer.k * layer.k, height * width
    values = layer.value_conv(inputs)
    value_unfold = F.unfold(values, kernel_size=layer.k, padding=layer.pad).view(
        batch, channels, patch_area, locations
    )
    logits = layer.gate_conv(inputs)
    membership = F.softmax(logits / float(layer.cfg.fuzzy_temperature), dim=1)
    membership_unfold = F.unfold(membership, kernel_size=layer.k, padding=layer.pad).view(
        batch, layer.R, patch_area, locations
    )
    valid = F.unfold(
        torch.ones(batch, 1, height, width, device=inputs.device, dtype=inputs.dtype),
        kernel_size=layer.k,
        padding=layer.pad,
    ).view(batch, 1, patch_area, locations)
    center_index = (layer.k // 2) * layer.k + layer.k // 2
    membership_center = membership_unfold[:, :, center_index : center_index + 1]
    kernel, _gap, _smoothness, interpretation = layer._local_hyperbolic_kernel(inputs)
    raw_interaction = membership_center * membership_unfold * kernel * valid
    normalized = raw_interaction / raw_interaction.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    aggregate = torch.einsum("brsl,bcsl->brcl", normalized, value_unfold)
    reconstructed = layer.pointwise(aggregate.reshape(batch, layer.R * channels, height, width))
    actual = layer(inputs)
    max_error = float((actual - reconstructed).abs().max().cpu())
    theta = interpretation["theta_map"]
    rho_numerator = torch.sqrt(
        (membership * torch.cos(2.0 * theta)).sum(dim=1).square()
        + (membership * torch.sin(2.0 * theta)).sum(dim=1).square()
    )
    rho = rho_numerator / membership.sum(dim=1).clamp_min(1e-8)
    return {
        "forward_reconstruction_max_abs_error": max_error,
        "membership_sum_max_abs_error": float((membership.sum(dim=1) - 1.0).abs().max().cpu()),
        "normalization_sum_max_abs_error": float((normalized.sum(dim=(1, 2)) - 1.0).abs().max().cpu()),
        "tensor_shapes": {
            "mu": list(membership.shape),
            "theta": list(theta.shape),
            "sigma_parallel": list(interpretation["sigma_u_map"].shape),
            "sigma_perpendicular": list(interpretation["sigma_s_map"].shape),
            "G": list(kernel.shape),
            "w_raw_per_mode": list(raw_interaction.shape),
            "W_raw_mode_sum": list(raw_interaction.sum(dim=1).shape),
            "w_normalized": list(normalized.shape),
            "rho": list(rho.shape),
            "output": list(actual.shape),
        },
        "ranges": {
            "mu": [float(membership.min().cpu()), float(membership.max().cpu())],
            "G": [float(kernel.min().cpu()), float(kernel.max().cpu())],
            "w_raw": [float(raw_interaction.min().cpu()), float(raw_interaction.max().cpu())],
            "w_normalized": [float(normalized.min().cpu()), float(normalized.max().cpu())],
            "rho": [float(rho.min().cpu()), float(rho.max().cpu())],
        },
        "all_finite": bool(
            all(torch.isfinite(value).all() for value in (membership, theta, kernel, raw_interaction, normalized, rho, actual))
        ),
    }


def _split_feasibility() -> dict[str, Any]:
    image_ids = {
        int(path.stem.split("_")[-1])
        for path in (PROJECT_ROOT / "data" / "cracks" / "images").glob("section_*.png")
    }
    training = set(int(value) for value in T1_PROTOCOL["training_section_ids"])
    old_split = json.loads(
        (PROJECT_ROOT / "results" / "structural_reachability" / "phase_a" / "split_manifest.json").read_text()
    )
    old_validation = set(int(value) for value in old_split["validation_section_ids"])
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    expert = set(int(value) for value in protocol["setting_a"]["expert_evaluation_sections"])
    unseen = sorted(image_ids - training)
    annotation_root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    annotation_counts = {
        str(section_id): len(list(annotation_root.glob(f"*/section_{section_id:03d}.png")))
        for section_id in unseen
    }
    eligible = [
        section_id for section_id in unseen
        if section_id not in old_validation
        and section_id not in expert
        and annotation_counts[str(section_id)] > 0
    ]
    return {
        "available_image_count": len(image_ids),
        "segmentation_training_section_count": len(training),
        "old_phase_a_validation_section_count": len(old_validation),
        "unseen_image_section_ids": unseen,
        "unseen_annotation_counts": annotation_counts,
        "unseen_expert_overlap": sorted(set(unseen) & expert),
        "unseen_old_phase_a_overlap": sorted(set(unseen) & old_validation),
        "eligible_independent_nonexpert_confirm_section_ids": eligible,
        "eligible_section_count": len(eligible),
        "status": "PASS" if eligible else "STOP_NO_INDEPENDENT_CONFIRM_SPLIT",
        "reason": None if eligible else "All three segmentation-unseen CRACKS images have zero crowd annotations.",
    }


def phase0_protocol() -> dict[str, Any]:
    source = PROJECT_ROOT / "models" / "azconv.py"
    checkpoints = {str(seed): _sha256(_frozen_run(seed)[1]) for seed in SEEDS}
    return {
        "version": "original_anza_operator_forensics_phase0_v1",
        "question": "Does the frozen legacy checkpoint implement the published/current original ANZA interaction literally?",
        "source_sha256": _sha256(source),
        "audit_source_sha256": _sha256(Path(__file__)),
        "expected_unchanged_legacy_source_sha256": LEGACY_SOURCE_SHA256,
        "checkpoint_sha256": checkpoints,
        "published_contract_source": "docs/research/ANZA_V1_FORMULA_CODE_AUDIT.md",
        "stop_on_material_definition_mismatch": True,
        "instrumentation_allowed_only_after_match": True,
        "confirm_allowed_only_after_instrumentation_and_independent_split": True,
        "training_performed": False,
        "expert_data_accessed": False,
    }


def audit_original_anza_operator() -> dict[str, Any]:
    """Write Phase-0 evidence and stop before instrumentation on a material mismatch."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_metrics = PROJECT_ROOT / "results" / "structural_reachability" / "phase_a" / "metrics.json"
    if _sha256(previous_metrics) != PREVIOUS_PHASE_A_METRICS_SHA256:
        raise ValueError("previous frozen Phase-A metrics changed")
    if _sha256(PROJECT_ROOT / "models" / "azconv.py") != LEGACY_SOURCE_SHA256:
        raise ValueError("legacy AZConv2d source changed before forensics")
    protocol = phase0_protocol()
    protocol_hash = _canonical_hash(protocol)
    spec, checkpoint = _frozen_run(42)
    model = _model(spec)
    load_t1_checkpoint(checkpoint, spec, model)
    layer = model.enc1.spatial
    torch.manual_seed(20260818)
    runtime = inspect_legacy_layer(layer, torch.randn(1, 3, 7, 9))
    source = inspect.getsource(AZConv2d.forward)
    code_checks = {
        "softmax_membership": "F.softmax" in source,
        "pair_membership_product": "mu_center * mu_un * kern" in source,
        "global_rule_neighbor_normalization": "compat.sum(dim=(1, 2)" in source,
        "mode_aggregation_then_pointwise": "torch.einsum" in source and "self.pointwise" in source,
    }
    findings = {
        "membership": {
            "actual": "softmax(logits / temperature, dim=mode)",
            "published_contract": "independent fuzzy degree",
            "verdict": "MATERIAL_MISMATCH",
            "mode_count": int(layer.R),
        },
        "orientation": {
            "actual": "raw theta per mode; center/neighbor doubled-angle vectors are averaged then halved",
            "axial": True,
            "published_packet_formula": "directed center theta_r(p)",
            "verdict": "MATERIAL_PARAMETERIZATION_MISMATCH",
        },
        "scales": {
            "actual": "base=softplus(raw_base)+1e-4; hyper=clamp(softplus(raw_hyper), max=1); pair-average base/hyper; sigma_parallel=base*exp(hyper), sigma_perpendicular=base*exp(-hyper)",
            "strictly_positive_for_finite_parameters": True,
            "max_hyperbolicity": float(layer.cfg.max_hyperbolicity),
        },
        "geometry": {
            "actual": "exp(-d_parallel^2/sigma_parallel^2-d_perpendicular^2/sigma_perpendicular^2)",
            "published_packet_formula": "exp(-d_parallel^2/(2 sigma_parallel^2)-d_perpendicular^2/(2 sigma_perpendicular^2))",
            "verdict": "SCALE_EQUIVALENT_BUT_NOT_LITERAL",
        },
        "interaction": {
            "actual_raw": "mu_r(center)*mu_r(neighbor)*G_r*valid",
            "compatibility_floor": float(layer.cfg.compatibility_floor),
            "normalization": "global over modes and valid neighbor offsets for each destination pixel",
            "mode_handling": "normalized per-mode neighbor aggregates are concatenated and mixed by a learned 1x1 pointwise convolution",
            "W_explicitly_retained": False,
            "W_read_only_reconstructable": True,
        },
        "legacy_vs_clean": {
            "legacy": "categorical softmax memberships",
            "clean": "independent sigmoid memberships in isolated models/azconv_clean.py",
            "frozen_checkpoint_operator": "legacy",
        },
        "paper_equation_literal_match": False,
        "checkpoint_matches_legacy_source": True,
    }
    split = _split_feasibility()
    primary_status = "STOP_OPERATOR_DEFINITION_MISMATCH"
    result = {
        "status": primary_status,
        "protocol_sha256": protocol_hash,
        "phase": "0_CODE_FORENSICS",
        "findings": findings,
        "code_checks": code_checks,
        "runtime": runtime,
        "split_feasibility": split,
        "instrumentation_performed": False,
        "instrumentation_status": "NOT_RUN_STOP_OPERATOR_DEFINITION_MISMATCH",
        "confirm_performed": False,
        "confirm_status": "NOT_RUN",
        "training_performed": False,
        "expert_data_accessed": False,
        "expert_scores_used": False,
        "next_phase_allowed": False,
        "root_cause": "The frozen legacy operator uses categorical softmax and symmetric pair-averaged local geometry, so it is not the literal published/current directed independent-fuzzy interaction. Independently, no segmentation-unseen CRACKS image has crowd annotations.",
    }
    (OUTPUT_ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "protocol_hash.txt").write_text(protocol_hash + "\n")
    (OUTPUT_ROOT / "operator_forensics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "split_feasibility.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "split_manifest.json").write_text(json.dumps({
        "status": split["status"],
        "segmentation_training_section_count": split["segmentation_training_section_count"],
        "segmentation_unseen_section_ids": split["unseen_image_section_ids"],
        "segmentation_unseen_annotation_counts": split["unseen_annotation_counts"],
        "confirm_section_ids": [],
        "confirm_split_frozen": False,
        "confirm_authorized": False,
        "reason": split["reason"],
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "ORIGINAL_ANZA_CONFIRM_SPLIT.json").write_text(json.dumps({
        **split,
        "confirm_split_frozen": False,
        "confirm_authorized": False,
        "overlap_checks": {
            "training_overlap": False,
            "old_phase_a_overlap": False,
            "expert_overlap": False,
        },
        "note": "No eligible annotated sections exist; empty split is a failure receipt, not a confirm split.",
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "checkpoint_manifest.json").write_text(json.dumps({
        str(seed): {"path": str(_frozen_run(seed)[1]), "sha256": _sha256(_frozen_run(seed)[1])}
        for seed in SEEDS
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "data_access_log.json").write_text(json.dumps({
        "metadata_only": [
            "results/anza_v2_study/protocol.json",
            "results/structural_reachability/phase_a/split_manifest.json",
            "CRACKS image filenames",
            "CRACKS crowd annotation filenames",
        ],
        "image_pixels_read": [],
        "crowd_annotation_pixels_read": [],
        "expert_paths": [],
        "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu_read_only_fixture",
    }, indent=2, sort_keys=True) + "\n")
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    ).stdout
    (OUTPUT_ROOT / "code_state.json").write_text(json.dumps({
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "dirty": bool(git_status.strip()),
        "modified_or_untracked_count": len(git_status.splitlines()),
        "git_status_lines": git_status.splitlines(),
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "commit_created": False,
    }, indent=2, sort_keys=True) + "\n")
    report = f"""# ANZA operator forensics

Status: `{primary_status}`

## Exact frozen operator

| Item | Frozen legacy implementation | Verdict |
|---|---|---|
| Membership | `softmax(logits / temperature, dim=mode)`, four modes | Material mismatch with independent fuzzy-degree contract |
| Orientation | per-mode raw angle; center and neighbor combined with doubled-angle axial mean | Axial, but not the directed center-only packet formula |
| Scales | positive local base/hyper fields; pair-averaged before `sigma_parallel/perpendicular` | Actual code documented |
| Geometry | `exp(-d_parallel^2/sigma_parallel^2-d_perpendicular^2/sigma_perpendicular^2)` | Missing literal `1/2`; scale-equivalent, not literal |
| Raw interaction | `mu_center * mu_neighbor * G * valid` | Matches pair-product structure |
| Normalization | global over four modes and valid 3x3 offsets per destination | Matches current code |
| Mode fusion | aggregate per mode, concatenate, learned 1x1 pointwise mix | `W=sum_r w_r` is reconstructable but not the standalone tensor consumed by output |

Runtime reconstruction error: `{runtime['forward_reconstruction_max_abs_error']:.3e}`.  
Membership sum error: `{runtime['membership_sum_max_abs_error']:.3e}`.  
Normalized interaction sum error: `{runtime['normalization_sum_max_abs_error']:.3e}`.

The source remains unchanged at `{LEGACY_SOURCE_SHA256}` and the frozen T1 checkpoints load this legacy operator. CleanANZA is a separate sigmoid-membership implementation and is not substituted here.

## Stop decision

The packet requires an immediate stop on a material paper/code definition mismatch. Therefore read-only instrumentation, S0-S4 confirm scoring, learned affinity, and training were not run.

There is also no legal confirm split: the only segmentation-unseen images are sections `49`, `73`, and `385`, and each has zero crowd annotation files. Statistical independence cannot be manufactured from edges inside already trained-on sections.

- Training performed: no
- Expert accessed: no
- Confirm performed: no
- Next phase allowed: no
"""
    (PROJECT_ROOT / "docs" / "research" / "ANZA_OPERATOR_FORENSICS.md").write_text(report)
    (OUTPUT_ROOT / "FAILURE_ANALYSIS.md").write_text(
        "# Original ANZA Phase-0 failure\n\n"
        "Primary stop: `STOP_OPERATOR_DEFINITION_MISMATCH`. The frozen legacy operator is not a literal implementation of the current/published independent-fuzzy directed equation.\n\n"
        "Independent secondary blocker: `STOP_NO_INDEPENDENT_CONFIRM_SPLIT`. Sections 49, 73, and 385 are unseen by the segmentation checkpoint but have zero crowd annotations.\n\n"
        "No instrumentation, confirm score extraction, training, affinity head, or expert access is permitted under this packet.\n"
    )
    (OUTPUT_ROOT / "NOT_APPLICABLE.json").write_text(json.dumps({
        "reason": primary_status,
        "per_section.csv": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "per_edge.csv": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "per_gap.csv": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "per_candidate.csv": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "operating_curve.csv": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "metrics.json": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "bootstrap.json": "NOT_APPLICABLE_CONFIRM_NOT_RUN",
        "ORIGINAL_ANZA_CONFIRM_PROTOCOL.json": "NOT_CREATED_STOP_BEFORE_CONFIRM_FREEZE",
        "ORIGINAL_ANZA_FORENSIC_CONFIRM_REPORT.md": "NOT_CREATED_CONFIRM_NOT_RUN",
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "ORIGINAL_ANZA_PHASE0_REPORT.md").write_text(f"""# Original ANZA Phase 0 report

```text
PHASE: 0 — CODE FORENSICS
STATUS: {primary_status}

RESEARCH QUESTION: Does the frozen legacy checkpoint literally implement the original published/current ANZA interaction?

PROTOCOL HASH: {protocol_hash}

DATA: code, checkpoint metadata, CRACKS section/annotation filenames only
TRAIN SECTIONS: 393 frozen historical sections
VALIDATION SECTIONS: NOT APPLICABLE
CONFIRM SECTIONS: 0 eligible (unseen 49/73/385 have no crowd annotations)
EXPERT ACCESSED: NO

TRAINING: NO

BASELINES: NOT APPLICABLE BEFORE CONFIRM
PRIMARY METRIC: exact equation/code contract
RESULTS: material softmax-membership and directed-vs-pair-symmetric geometry mismatch
PRIMARY DELTA: NOT APPLICABLE
95% CI: NOT APPLICABLE
PRE-SPECIFIED GATE: literal operator definition match required before instrumentation
PASS/FAIL: FAIL
ROOT CAUSE: frozen legacy code is not the literal independent-fuzzy directed equation; no annotated segmentation-unseen CRACKS section exists
WHAT THIS PROVES: the proposed confirm cannot claim to test the literal paper interaction with this checkpoint and dataset split
WHAT THIS DOES NOT PROVE: it does not show whether a newly trained corrected operator could work
NEXT PHASE ALLOWED: NO
FILES: docs/research/ANZA_OPERATOR_FORENSICS.md, operator_forensics.json, split_feasibility.json, FAILURE_ANALYSIS.md
TESTS: targeted forensic tests required
GIT STATUS: dirty working tree preserved; no commit/push
```
""")
    return result
