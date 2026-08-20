"""Evidence-backed localization after the predeclared synthetic gate failed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from method_repair.matrix import synthetic_matrix
from method_repair.training import build_candidate_model, cached_sample, load_candidate_checkpoint


def build_root_cause_analysis(
    synthetic_root: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    diagnostic_samples: int = 32,
) -> dict[str, Any]:
    synthetic_root = Path(synthetic_root)
    gate = json.loads((synthetic_root / "mechanism_gate.json").read_text())
    if gate.get("status") != "SYNTHETIC_MECHANISM_FAIL" or gate.get("cracks_authorized") is not False:
        raise ValueError("root-cause closeout is only valid after a failed frozen mechanism gate")
    torch_device = torch.device(device)
    candidates: dict[str, Any] = {}
    for spec in synthetic_matrix()[1:]:
        run_dir = synthetic_root / "development" / f"{spec.candidate_id}-{spec.run_hash}"
        validation = json.loads(
            (synthetic_root / "validation" / f"{spec.candidate_id}-{spec.run_hash}.json").read_text()
        )
        status = json.loads((run_dir / "status.json").read_text())
        model = build_candidate_model(spec, widths=tuple(status["widths"])).to(torch_device).eval()
        load_candidate_checkpoint(
            run_dir / "checkpoint-last.pt",
            expected_hash=spec.run_hash,
            model=model,
        )
        ratios: list[float] = []
        lambdas: list[float] = []
        gates: list[float] = []
        with torch.inference_mode():
            for index in range(int(diagnostic_samples)):
                sample = cached_sample("validation", index, int(status["image_size"]))
                image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
                diagnostics = model(image, return_diagnostics=True)["transport_diagnostics"][0]
                ratios.append(float(
                    diagnostics["correction"].abs().mean()
                    / diagnostics["base_output"].abs().mean().clamp_min(1e-8)
                ))
                lambdas.append(float(diagnostics["residual_lambda"]))
                gates.append(float(diagnostics["ambiguity_gate"].mean()))
        spatial = model.enc1.spatial
        direct_history = status["history"] if spec.direct_mode_supervision else []
        candidates[spec.candidate_id] = {
            "run_hash": spec.run_hash,
            "membership_head_kernel": list(spatial.membership_head.kernel_size),
            "geometry_head_kernel": list(spatial.geometry_head.kernel_size),
            "residual_lambda": float(np.mean(lambdas)),
            "correction_to_base_abs_mean_ratio": float(np.mean(ratios)),
            "gate_global_mean": float(np.mean(gates)),
            "validation_metrics": validation["metrics"],
            "membership_set_kl": {
                "initial": float(direct_history[0]["train_membership_set_kl"]) if direct_history else None,
                "minimum": float(min(row["train_membership_set_kl"] for row in direct_history)) if direct_history else None,
                "final": float(direct_history[-1]["train_membership_set_kl"]) if direct_history else None,
            },
            "orientation_set_loss_final": (
                float(direct_history[-1]["train_orientation_set"]) if direct_history else None
            ),
        }
        del model
    root_causes = [
        {
            "id": "RC1_POINTWISE_AMBIGUITY_OBSERVABILITY",
            "status": "SUPPORTED_ROOT_CAUSE",
            "evidence": (
                "Membership and geometry heads are 1x1; A3/A4 learn low axial set error but retain "
                "high membership KL, nearly equal N_eff/J at junction and straight pixels, and a near-global gate."
            ),
            "claim_boundary": (
                "The bounded experiment supports inadequate local observability for this implementation; "
                "it does not prove that every context-aware ambiguity detector will fail."
            ),
        },
        {
            "id": "RC2_NO_NEGATIVE_GAP_OBJECTIVE",
            "status": "SUPPORTED_ROOT_CAUSE",
            "evidence": (
                "A0-A4 contain no matched-negative gap loss; all validation false-bridge endpoints equal 1.0 "
                "despite improved route AP and entropy. Route identity alone did not control pixel completion."
            ),
            "claim_boundary": "Do not use false-bridge improvement as evidence; it was not achieved.",
        },
        {
            "id": "RC3_CONTEXT_ONLY_IN_TRANSPORT_NOT_GATE",
            "status": "SUPPORTED_ROOT_CAUSE",
            "evidence": (
                "A4 expands routing context from 3x3 to 5x5 but leaves the pointwise mode/gate predictors "
                "unchanged; its mechanism gates still fail and visible Dice/clDice violate non-inferiority."
            ),
            "claim_boundary": "Larger transport support is not sufficient evidence for contextual ambiguity detection.",
        },
        {
            "id": "RC4_RESIDUAL_SAFETY_WORKED",
            "status": "POSITIVE_ENGINEERING_RESULT_NOT_MECHANISM_SUCCESS",
            "evidence": (
                "A1-A3 remain visible-noninferior to A0 and learned corrections stay small relative to the v1 base."
            ),
            "claim_boundary": "Safety/non-inferiority does not establish structural superiority.",
        },
    ]
    payload = {
        "status": "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE",
        "synthetic_gate": gate["status"],
        "cracks_training": "NOT_RUN_SYNTHETIC_GATE_FAILED",
        "expert_evaluation": "NOT_RUN",
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
        "expert_data_accessed": False,
        "diagnostic_split": "CrossingTraceBench-v2 validation only",
        "diagnostic_sample_count": int(diagnostic_samples),
        "candidates": candidates,
        "root_causes": root_causes,
        "next_experiment_not_executed": {
            "status": "REQUIRES_NEW_USER_DECISION_AND_NEW_PROTOCOL",
            "proposal": (
                "Give membership/geometry prediction an explicitly frozen local context (for example a small "
                "3x3 axial-equivariant head on base features), supervise background/straight/junction gate targets, "
                "and add a matched-negative gap objective confined to the router. Rebuild a new independent "
                "validation stream before any CRACKS training."
            ),
            "why_not_run_now": "A0-A4 search budget is exhausted; adding A5 post hoc would violate the frozen protocol.",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
