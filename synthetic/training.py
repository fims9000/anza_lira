"""Small resumable training utilities for controlled structural candidates."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.experiment_matrix import COMMON_PROTOCOL, SyntheticRunSpec
from synthetic.structural_losses import (
    branch_transition_logits,
    cone_consistency_loss,
    routing_supervision_loss,
    structural_gap_loss,
    visible_segmentation_loss,
)


SMOKE_CASES = (
    "single_straight",
    "x_junction",
    "t_junction",
    "y_junction",
    "fault_with_gap",
    "negative_gap",
    "curved_crossing",
    "nontrivial_pairing",
)
LOSS_WEIGHTS = {"route": 0.2, "positive_negative_gap": 0.2, "cone": 0.05}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@lru_cache(maxsize=1024)
def _cached_sample(split: str, index: int, image_size: int) -> dict[str, Any]:
    return generate_sample(split, index, image_size=image_size)


def _tensor(sample: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sample[name], device=device)


def candidate_loss(
    model: torch.nn.Module,
    spec: SyntheticRunSpec,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    image = _tensor(sample, "image", device).unsqueeze(0)
    visible_target = _tensor(sample, "visible_fault_mask", device).float().unsqueeze(0).unsqueeze(0)
    structural = any(name != "visible_bce_dice" for name in spec.objectives)
    output = model(image, return_diagnostics=structural)
    if structural:
        visible_logits = output["visible_logits"]
    else:
        visible_logits = output
    visible = visible_segmentation_loss(visible_logits, visible_target)
    total = visible
    parts = {"visible_bce_dice": float(visible.detach())}

    if "route" in spec.objectives:
        first = output["transport_diagnostics"][0]
        route_logits = branch_transition_logits(
            first["transport"],
            _tensor(sample, "branch_masks", device),
            variant=first["variant"],
        )
        route = routing_supervision_loss(
            route_logits,
            _tensor(sample, "continuation_relation_matrix", device),
            _tensor(sample, "continuation_eligible_matrix", device),
        )
        total = total + LOSS_WEIGHTS["route"] * route
        parts["route"] = float(route.detach())

    if "positive_negative_gap" in spec.objectives:
        completion_probability = torch.sigmoid(output["completion_logits"])
        gap, gap_parts = structural_gap_loss(
            completion_probability,
            _tensor(sample, "positive_gap_mask", device).unsqueeze(0).unsqueeze(0),
            _tensor(sample, "negative_gap_mask", device).unsqueeze(0).unsqueeze(0),
        )
        total = total + LOSS_WEIGHTS["positive_negative_gap"] * gap
        parts.update({name: float(value.detach()) for name, value in gap_parts.items()})

    if "cone" in spec.objectives:
        first = output["transport_diagnostics"][0]
        cone = cone_consistency_loss(
            first["membership"],
            first["theta"],
            first["junction_score"],
            torch.sigmoid(visible_logits),
        )
        total = total + LOSS_WEIGHTS["cone"] * cone
        parts["cone"] = float(cone.detach())
    parts["total"] = float(total.detach())
    return total, parts


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: SyntheticRunSpec,
    epoch: int,
    status: str,
) -> dict[str, Any]:
    return {
        "run_hash": spec.run_hash,
        "spec": asdict(spec),
        "epoch": int(epoch),
        "status": status,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def load_checkpoint(
    path: Path,
    *,
    expected_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_hash") != expected_hash:
        raise ValueError("Checkpoint config hash does not match the requested run")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def run_candidate_smoke(
    spec: SyntheticRunSpec,
    output_root: Path,
    *,
    epochs: int = 1,
    image_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    run_dir = output_root / f"{spec.candidate_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("run_hash") == spec.run_hash:
            return {**existing, "action": "SKIP"}

    _set_seed(spec.seed)
    torch_device = torch.device(device)
    model = build_comparable_model(spec.model, widths=(4, 8, 12, 16)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_hash=spec.run_hash,
            model=model,
            optimizer=optimizer,
        )
        start_epoch = int(checkpoint["epoch"])

    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for case_index, case in enumerate(SMOKE_CASES):
            sample = generate_sample("train", case_index, image_size=image_size, case=case)
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = candidate_loss(model, spec, sample, torch_device)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {spec.candidate_id} case={case}")
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError(f"Non-finite gradient for {spec.candidate_id} case={case}")
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(epoch_losses))})
        torch.save(_checkpoint_payload(model, optimizer, spec, epoch + 1, "IN_PROGRESS"), checkpoint_path)

    # Reload into an independent instance before declaring the smoke complete.
    reloaded = build_comparable_model(spec.model, widths=(4, 8, 12, 16))
    payload = load_checkpoint(checkpoint_path, expected_hash=spec.run_hash, model=reloaded)
    model.eval()
    reloaded.eval()
    validation = generate_sample("validation", 0, image_size=image_size, case="x_junction")
    validation_image = _tensor(validation, "image", torch_device).unsqueeze(0)
    with torch.no_grad():
        original_output = model(validation_image)
        reloaded_output = reloaded(validation_image.cpu())
    if not torch.allclose(original_output.cpu(), reloaded_output, atol=1e-6, rtol=1e-6):
        raise ValueError("Checkpoint reload changed model inference")
    torch.save(_checkpoint_payload(model, optimizer, spec, int(payload["epoch"]), "COMPLETE"), checkpoint_path)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "scientific_result": False,
        "run_hash": spec.run_hash,
        "candidate_id": spec.candidate_id,
        "model": spec.model,
        "objectives": list(spec.objectives),
        "epochs": int(payload["epoch"]),
        "checkpoint_reload": "PASS",
        "history": history,
        "test_samples_opened": 0,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _visible_dice_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    intersection = int((prediction & truth).sum())
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * intersection / denominator if denominator else 1.0


def run_candidate_development(
    spec: SyntheticRunSpec,
    output_root: Path,
    *,
    device: str = "cuda",
    epochs: int | None = None,
    train_samples: int = 256,
    validation_samples: int = 256,
    image_size: int = 128,
) -> dict[str, Any]:
    """Train one predeclared candidate without touching the synthetic test stream."""
    epoch_budget = int(COMMON_PROTOCOL["epoch_budget"] if epochs is None else epochs)
    run_dir = output_root / f"{spec.candidate_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    log_path = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("run_hash") == spec.run_hash
            and int(existing.get("epoch", 0)) >= epoch_budget
        ):
            return {**existing, "action": "SKIP"}

    _set_seed(spec.seed)
    torch_device = torch.device(device)
    model = build_comparable_model(spec.model).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(COMMON_PROTOCOL["learning_rate"]))
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_hash=spec.run_hash,
            model=model,
            optimizer=optimizer,
        )
        start_epoch = int(checkpoint["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))

    accumulation = int(COMMON_PROTOCOL["gradient_accumulation"])
    for epoch in range(start_epoch, epoch_budget):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for local_index in range(train_samples):
            sample = _cached_sample("train", local_index, image_size)
            loss, _parts = candidate_loss(model, spec, sample, torch_device)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {spec.candidate_id} index={local_index}")
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if (local_index + 1) % accumulation == 0 or local_index + 1 == train_samples:
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError(f"Non-finite gradient for {spec.candidate_id} index={local_index}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_dice = []
        with torch.no_grad():
            for local_index in range(validation_samples):
                sample = _cached_sample("validation", local_index, image_size)
                image = _tensor(sample, "image", torch_device).unsqueeze(0)
                target = _tensor(sample, "visible_fault_mask", torch_device).float().unsqueeze(0).unsqueeze(0)
                validation_dice.append(_visible_dice_from_logits(model(image), target))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_visible_dice_at_0_5": float(np.mean(validation_dice)),
        }
        history.append(row)
        torch.save(_checkpoint_payload(model, optimizer, spec, epoch + 1, "IN_PROGRESS"), checkpoint_path)
        progress = {
            "status": "IN_PROGRESS",
            "scientific_result": False,
            "run_hash": spec.run_hash,
            "candidate_id": spec.candidate_id,
            "model": spec.model,
            "objectives": list(spec.objectives),
            "epoch": epoch + 1,
            "epoch_budget": epoch_budget,
            "history": history,
            "test_samples_opened": 0,
        }
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"phase=synthetic_development model={spec.candidate_id} seed={spec.seed} "
            f"epoch={epoch + 1}/{epoch_budget} val_visible_dice={row['validation_visible_dice_at_0_5']:.4f} "
            "structural_metric=WAITING status=RUNNING"
        )

    reloaded = build_comparable_model(spec.model)
    load_checkpoint(checkpoint_path, expected_hash=spec.run_hash, model=reloaded)
    torch.save(_checkpoint_payload(model, optimizer, spec, epoch_budget, "COMPLETE"), checkpoint_path)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "scientific_result": False,
        "run_hash": spec.run_hash,
        "candidate_id": spec.candidate_id,
        "model": spec.model,
        "objectives": list(spec.objectives),
        "epoch": epoch_budget,
        "epoch_budget": epoch_budget,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "image_size": image_size,
        "checkpoint_reload": "PASS",
        "history": history,
        "test_samples_opened": 0,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
