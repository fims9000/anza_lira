"""Bounded learned-affinity experiment for ANZA-2 Phase 3.

Generic and ANZA variants start from identical backbone and generic-head
weights. Development and confirm streams remain disjoint and expert-free.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.anza2.affinity import ANZA2StructuralAffinity, GenericAffinityCombiner, LOCAL8_OFFSETS
from models.anza2.field import ANZA2Field, ANZA2FieldConfig
from models.anza2.losses import active_mode_count_loss, axis_set_coverage_loss
from synthetic.affinity_losses import balanced_affinity_bce
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3"
SEEDS = (41, 42, 43)
VARIANTS = ("generic", "generic_plus_anza")


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "anza2_phase3_learned_affinity_v1",
        "benchmark": "CrossingTraceBench-v4 independent train/validation/confirm streams",
        "variants": list(VARIANTS), "seeds": list(SEEDS), "image_size": 64,
        "train_samples": 256, "development_samples": 256, "confirm_samples": 256,
        "epochs": 5, "batch_size": 8, "learning_rate": 0.001,
        "encoder_channels": 16, "encoder_effective_receptive_field": 11,
        "initial_beta": 0.05,
        "membership_loss_weight": 0.25, "orientation_loss_weight": 0.10,
        "mode_count_loss_weight": 0.05,
        "primary_metric": "balanced local-edge TPR at FPR <= 0.05",
        "minimum_tpr_delta": 0.08, "ci_lower_required": 0.0,
        "bootstrap_unit": "synthetic sample", "bootstrap_resamples": 2000,
        "confirm_open_rule": "development gate must pass and thresholds must be frozen first",
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LearnedAffinityModel(nn.Module):
    """Common local encoder, generic edge logits, and optional ANZA prior."""

    def __init__(self, *, initial_beta: float = 0.05) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.SiLU(),
            nn.Conv2d(16, 16, 5, padding=2), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
        )
        self.generic_head = nn.Conv2d(16, len(LOCAL8_OFFSETS), 1)
        self.field = ANZA2Field(16, ANZA2FieldConfig(num_modes=4))
        self.anza_affinity = ANZA2StructuralAffinity()
        self.combiner = GenericAffinityCombiner(initial_beta=initial_beta)

    def forward(self, image: torch.Tensor, *, use_anza: bool) -> dict[str, Any]:
        features = self.encoder(image)
        generic_logits = self.generic_head(features)
        field = self.field(features)
        affinity = self.anza_affinity(field)
        logits = self.combiner(generic_logits, affinity) if use_anza else generic_logits
        return {"logits": logits, "generic_logits": generic_logits, "field": field, "anza_affinity": affinity}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=1024)
def _sample(split: str, index: int, image_size: int) -> dict[str, Any]:
    return generate_sample_v4(split, index, image_size=image_size)


def _batch(split: str, indices: list[int], image_size: int, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    samples = [_sample(split, int(index), image_size) for index in indices]
    images = torch.stack([torch.as_tensor(sample["image"]) for sample in samples]).to(device)
    edge = [build_affinity_targets(sample, LOCAL8_OFFSETS) for sample in samples]
    targets = {
        "positive": torch.stack([torch.as_tensor(item["affinity_positive"]) for item in edge]).to(device),
        "negative": torch.stack([torch.as_tensor(item["affinity_hard_negative"]) for item in edge]).to(device),
        "visible": torch.stack([torch.as_tensor(sample["visible_fault_mask"]) for sample in samples]).to(device),
        "theta": torch.stack([torch.as_tensor(sample["gt_theta_set"]) for sample in samples]).to(device),
        "theta_valid": torch.stack([torch.as_tensor(sample["gt_theta_valid"]) for sample in samples]).to(device),
        "mode_count": torch.stack([torch.as_tensor(sample["gt_mode_count"]) for sample in samples]).to(device),
    }
    return images.float(), targets


def _train_one(
    variant: str, seed: int, *, device: torch.device, train_samples: int,
    epochs: int, batch_size: int, image_size: int, output_root: Path,
) -> tuple[LearnedAffinityModel, list[dict[str, float]]]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    set_seed(seed)
    model = LearnedAffinityModel(initial_beta=0.05).to(device)
    use_anza = variant == "generic_plus_anza"
    if not use_anza:
        for parameter in model.field.parameters():
            parameter.requires_grad_(False)
        for parameter in model.combiner.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train(); losses: list[float] = []
        order = np.random.default_rng(seed * 10_000 + epoch).permutation(train_samples)
        for start in range(0, train_samples, batch_size):
            images, targets = _batch("train", order[start:start + batch_size].tolist(), image_size, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, use_anza=use_anza)
            loss = balanced_affinity_bce(output["logits"], targets["positive"], targets["negative"])
            if use_anza:
                field = output["field"]
                fuzzy_union = 1.0 - torch.prod(1.0 - field.membership, dim=1)
                membership = F.binary_cross_entropy(fuzzy_union.clamp(1e-6, 1 - 1e-6), targets["visible"].float())
                theta = targets["theta"].float()
                target_orientation = torch.stack((torch.cos(2 * theta), torch.sin(2 * theta)), dim=2)
                orientation = axis_set_coverage_loss(field.orientation, target_orientation, targets["theta_valid"].bool())
                count = active_mode_count_loss(
                    field.membership, targets["mode_count"].float(),
                    torch.ones_like(targets["visible"], dtype=torch.bool),
                )
                loss = loss + 0.25 * membership + 0.10 * orientation + 0.05 * count
            if not torch.isfinite(loss):
                raise ValueError("non-finite Phase-3 loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite Phase-3 gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        row = {"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "beta": float(model.combiner.beta.detach())}
        history.append(row)
        print(
            f"phase=anza2_phase3 variant={variant} seed={seed} "
            f"epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} beta={row['beta']:.5f}",
            flush=True,
        )
    run_dir = output_root / "runs" / f"{variant}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "variant": variant, "seed": seed,
        "protocol_sha256": canonical_hash(protocol_payload()), "history": history,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, run_dir / "checkpoint-last.pt")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "variant": variant, "seed": seed, "history": history,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return model, history


@torch.inference_mode()
def _score(
    model: LearnedAffinityModel, variant: str, split: str, count: int,
    image_size: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval(); rows: list[dict[str, Any]] = []
    for index in range(count):
        images, targets = _batch(split, [index], image_size, device)
        scores = torch.sigmoid(model(images, use_anza=variant == "generic_plus_anza")["logits"])[0]
        rows.append({
            "index": index,
            "positive_scores": scores[targets["positive"][0]].cpu().numpy().astype(np.float32),
            "negative_scores": scores[targets["negative"][0]].cpu().numpy().astype(np.float32),
        })
    return rows


def _threshold(rows: list[dict[str, Any]], target_fpr: float = 0.05) -> float:
    negatives = np.concatenate([row["negative_scores"] for row in rows if len(row["negative_scores"])])
    if negatives.size == 0:
        raise ValueError("development stream has no hard negatives")
    # Select the lowest observed threshold whose inclusive decision rule still
    # satisfies the declared FPR budget. Quantiles alone can exceed the budget
    # when several negatives tie exactly at the boundary.
    selected = float("inf")
    for candidate in np.unique(negatives)[::-1]:
        if float(np.mean(negatives >= candidate)) <= float(target_fpr):
            selected = float(candidate)
        else:
            break
    return selected


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    positives = np.concatenate([row["positive_scores"] for row in rows if len(row["positive_scores"])])
    negatives = np.concatenate([row["negative_scores"] for row in rows if len(row["negative_scores"])])
    return {
        "threshold": threshold, "tpr": float(np.mean(positives >= threshold)),
        "fpr": float(np.mean(negatives >= threshold)),
        "positive_edges": int(positives.size), "negative_edges": int(negatives.size),
    }


def _paired_bootstrap(
    generic: list[dict[str, Any]], anza: list[dict[str, Any]], thresholds: dict[str, float],
    *, resamples: int,
) -> tuple[float, list[float]]:
    deltas = []
    for base, improved in zip(generic, anza, strict=True):
        if len(base["positive_scores"]) == 0:
            continue
        base_tpr = np.mean(base["positive_scores"] >= thresholds["generic"])
        anza_tpr = np.mean(improved["positive_scores"] >= thresholds["generic_plus_anza"])
        deltas.append(float(anza_tpr - base_tpr))
    rng = np.random.default_rng(20260818)
    boot = [float(np.mean(rng.choice(deltas, len(deltas), replace=True))) for _ in range(resamples)]
    return float(np.mean(deltas)), [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def run_phase3(output_root: Path = OUTPUT_ROOT, *, mode: str = "smoke", device: str = "cpu") -> dict[str, Any]:
    if mode not in {"smoke", "development"}:
        raise ValueError("confirm is opened by a separate fail-closed command")
    protocol = protocol_payload(); protocol_hash = canonical_hash(protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("Phase-3 protocol drift")
    protocol_path.write_text(encoded); (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    if mode == "smoke":
        # v4 orders the first 128 positive gaps before 128 matched negatives;
        # include both classes so the low-FPR operating point is defined.
        seeds, train_samples, eval_samples, epochs = (41,), 160, 160, 1
    else:
        seeds, train_samples, eval_samples, epochs = SEEDS, 256, 256, 5
    torch.set_num_threads(min(2, torch.get_num_threads()))
    device_obj = torch.device(device)
    seed_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        scored: dict[str, list[dict[str, Any]]] = {}; metrics: dict[str, Any] = {}
        for variant in VARIANTS:
            model, history = _train_one(
                variant, seed, device=device_obj, train_samples=train_samples, epochs=epochs,
                batch_size=8, image_size=64, output_root=output_root / mode,
            )
            rows = _score(model, variant, "validation", eval_samples, 64, device_obj)
            threshold = _threshold(rows)
            scored[variant] = rows; metrics[variant] = {**_metrics(rows, threshold), "history": history}
        thresholds = {variant: float(metrics[variant]["threshold"]) for variant in VARIANTS}
        delta, ci = _paired_bootstrap(scored["generic"], scored["generic_plus_anza"], thresholds, resamples=2000)
        seed_metrics.append({"seed": seed, "variants": metrics, "tpr_delta": delta, "tpr_delta_ci95": ci})
    result = {
        "status": "SMOKE_PASS" if mode == "smoke" else "DEVELOPMENT_COMPLETE",
        "mode": mode, "protocol_sha256": protocol_hash, "seed_metrics": seed_metrics,
        "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    result_dir = output_root / mode; result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
