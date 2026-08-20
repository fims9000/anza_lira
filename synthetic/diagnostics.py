"""Failure localization diagnostics after the frozen synthetic validation gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.evaluation import continuation_probabilities
from synthetic.experiment_matrix import SyntheticRunSpec
from synthetic.structural_losses import branch_transition_logits
from synthetic.training import load_checkpoint


def effective_mode_count(membership: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    normalized = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    return torch.exp(-(normalized * torch.log(normalized.clamp_min(epsilon))).sum(dim=1))


def _mean_selected(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if np.any(mask) else None


def diagnose_candidate(
    spec: SyntheticRunSpec,
    development_root: Path,
    output_root: Path,
    *,
    sample_count: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    if spec.model not in {"anza_v2a", "anza_v2b", "anza_v2_full"}:
        raise ValueError("Mechanism diagnostics require a V2 model")
    run_dir = development_root / f"{spec.candidate_id}-{spec.run_hash}"
    torch_device = torch.device(device)
    model = build_comparable_model(spec.model).to(torch_device)
    load_checkpoint(run_dir / "checkpoint-last.pt", expected_hash=spec.run_hash, model=model)
    model.eval()
    regions: dict[str, dict[str, list[float]]] = {
        name: {"effective_modes": [], "junction_score": []}
        for name in ("junction", "straight", "background")
    }
    route_top1: list[float] = []
    route_confidence: list[float] = []
    route_entropy: list[float] = []
    completion_background: list[float] = []
    completion_positive_gap: list[float] = []
    completion_negative_gap: list[float] = []
    with torch.no_grad():
        for index in range(sample_count):
            sample = generate_sample("validation", index, image_size=128)
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            output = model(image, return_diagnostics=True)
            first = output["transport_diagnostics"][0]
            membership = first["membership"]
            effective = effective_mode_count(membership)[0].cpu().numpy()
            junction_score = first["junction_score"][0].cpu().numpy()
            junction = np.asarray(sample["junction_map"], dtype=bool)
            visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
            masks = {
                "junction": junction,
                "straight": visible & ~junction,
                "background": ~visible,
            }
            for name, mask in masks.items():
                effective_value = _mean_selected(effective, mask)
                junction_value = _mean_selected(junction_score, mask)
                if effective_value is not None:
                    regions[name]["effective_modes"].append(effective_value)
                if junction_value is not None:
                    regions[name]["junction_score"].append(junction_value)

            target = torch.as_tensor(sample["continuation_relation_matrix"], device=torch_device)
            eligible = torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device)
            if target.any():
                logits = branch_transition_logits(
                    first["transport"],
                    torch.as_tensor(sample["branch_masks"], device=torch_device),
                    variant=first["variant"],
                )
                probability = continuation_probabilities(logits, eligible)
                valid_rows = target.any(dim=-1)
                predicted = probability[valid_rows].argmax(dim=-1)
                truth_rows = target[valid_rows]
                correct = truth_rows.gather(1, predicted.unsqueeze(1)).squeeze(1)
                route_top1.extend(correct.float().cpu().tolist())
                route_confidence.extend(probability[valid_rows].max(dim=-1).values.cpu().tolist())
                eligible_count = eligible[valid_rows].sum(dim=-1)
                entropy = -(
                    probability[valid_rows]
                    * torch.log(probability[valid_rows].clamp_min(1e-8))
                ).sum(dim=-1)
                normalizer = torch.log(eligible_count.float()).clamp_min(1.0)
                route_entropy.extend((entropy / normalizer).cpu().tolist())

            if "completion_logits" in output:
                probability = torch.sigmoid(output["completion_logits"])[0, 0].cpu().numpy()
                completion_background.append(float(probability[~visible].mean()))
                positive = np.asarray(sample["positive_gap_mask"], dtype=bool)
                negative = np.asarray(sample["negative_gap_mask"], dtype=bool)
                if positive.any():
                    completion_positive_gap.append(float(probability[positive].mean()))
                if negative.any():
                    completion_negative_gap.append(float(probability[negative].mean()))

    result = {
        "status": "COMPLETE",
        "candidate_id": spec.candidate_id,
        "run_hash": spec.run_hash,
        "split": "validation",
        "sample_count": sample_count,
        "test_samples_opened": 0,
        "regions": {
            region: {
                metric: float(np.mean(values)) if values else None
                for metric, values in metrics.items()
            }
            for region, metrics in regions.items()
        },
        "route_top1_accuracy": float(np.mean(route_top1)) if route_top1 else None,
        "route_max_probability": float(np.mean(route_confidence)) if route_confidence else None,
        "branch_routing_entropy": float(np.mean(route_entropy)) if route_entropy else None,
        "completion_background_probability": (
            float(np.mean(completion_background)) if completion_background else None
        ),
        "completion_positive_gap_probability": (
            float(np.mean(completion_positive_gap)) if completion_positive_gap else None
        ),
        "completion_negative_gap_probability": (
            float(np.mean(completion_negative_gap)) if completion_negative_gap else None
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{spec.candidate_id}-{spec.run_hash}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
