"""Resumable synthetic A0-A4 training with direct mode/route supervision."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from method_repair.matrix import COMMON_PROTOCOL, MethodRepairSpec
from models.segmentation_repaired import build_repaired_model
from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench_v2 import generate_sample_v2
from synthetic.mode_supervision import (
    axial_mode_set_loss,
    branch_mode_masks_from_tangents,
    mode_specific_branch_transition_logits,
)
from synthetic.structural_losses import routing_supervision_loss, visible_segmentation_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_candidate_model(
    spec: MethodRepairSpec,
    *,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
) -> torch.nn.Module:
    base = build_comparable_model("anza_v1", widths=widths)
    if spec.model == "anza_v1":
        return base
    if spec.model != "repaired":
        raise ValueError(f"Unknown method-repair model: {spec.model}")
    return build_repaired_model(
        widths=widths,
        routing_kernel_size=spec.routing_kernel_size,
        use_ambiguity_gate=spec.use_ambiguity_gate,
        seed_matched_v1=base,
    )


@lru_cache(maxsize=600)
def cached_sample(split: str, index: int, image_size: int) -> dict[str, Any]:
    return generate_sample_v2(split, index, image_size=image_size)


def _tensor(sample: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sample[key], device=device)


def repaired_candidate_loss(
    model: torch.nn.Module,
    spec: MethodRepairSpec,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    image = _tensor(sample, "image", device).unsqueeze(0)
    target = _tensor(sample, "visible_fault_mask", device).float().unsqueeze(0).unsqueeze(0)
    output = model(image, return_diagnostics=spec.direct_mode_supervision)
    logits = output["visible_logits"] if isinstance(output, dict) else output
    visible = visible_segmentation_loss(logits, target)
    total = visible
    parts = {"visible_bce_dice": float(visible.detach())}
    if spec.direct_mode_supervision:
        diagnostics = output["transport_diagnostics"][0]
        mode_loss, mode_details = axial_mode_set_loss(
            diagnostics["theta"],
            diagnostics["membership"],
            _tensor(sample, "gt_theta_set", device).float().unsqueeze(0),
            _tensor(sample, "gt_theta_valid", device).bool().unsqueeze(0),
        )
        branch_modes = branch_mode_masks_from_tangents(
            diagnostics["theta"],
            _tensor(sample, "gt_branch_theta", device).float().unsqueeze(0),
            _tensor(sample, "gt_branch_theta_valid", device).bool().unsqueeze(0),
        )[0]
        route_logits = mode_specific_branch_transition_logits(
            diagnostics["transport"],
            branch_modes,
            kernel_size=spec.routing_kernel_size,
        )
        route_loss = routing_supervision_loss(
            route_logits,
            _tensor(sample, "continuation_relation_matrix", device).bool(),
            _tensor(sample, "continuation_eligible_matrix", device).bool(),
        )
        total = (
            total
            + float(COMMON_PROTOCOL["mode_set_weight"]) * mode_loss
            + float(COMMON_PROTOCOL["mode_route_weight"]) * route_loss
        )
        parts.update({
            "mode_set": float(mode_loss.detach()),
            "orientation_set": float(mode_details["orientation_set_loss"].detach()),
            "membership_set_kl": float(mode_details["membership_set_kl"].detach()),
            "mode_specific_route": float(route_loss.detach()),
        })
    parts["total"] = float(total.detach())
    return total, parts


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: MethodRepairSpec,
    epoch: int,
    status: str,
) -> dict[str, Any]:
    return {
        "run_hash": spec.run_hash,
        "spec": asdict(spec),
        "protocol": COMMON_PROTOCOL,
        "epoch": int(epoch),
        "status": status,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "expert_data_accessed": False,
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
    }


def load_candidate_checkpoint(
    path: Path,
    *,
    expected_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_hash") != expected_hash:
        raise ValueError("method-repair checkpoint config hash mismatch")
    if payload.get("expert_data_accessed") is not False:
        raise ValueError("candidate checkpoint does not prove expert lock")
    if payload.get("old_test_samples_opened") != 0 or payload.get("new_test_samples_opened") != 0:
        raise ValueError("candidate checkpoint accessed a locked test stream")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _dice_at_half(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * int((prediction & truth).sum()) / denominator if denominator else 1.0


def run_synthetic_candidate(
    spec: MethodRepairSpec,
    output_root: Path,
    *,
    device: str = "cuda",
    epochs: int | None = None,
    train_samples: int = 256,
    validation_samples: int = 256,
    image_size: int = 128,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
) -> dict[str, Any]:
    epoch_budget = int(COMMON_PROTOCOL["epoch_budget"] if epochs is None else epochs)
    run_dir = Path(output_root) / f"{spec.candidate_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    heartbeat_path = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("run_hash") == spec.run_hash
            and int(existing.get("epoch", 0)) >= epoch_budget
        ):
            return {**existing, "action": "SKIP"}

    set_seed(spec.seed)
    torch_device = torch.device(device)
    model = build_candidate_model(spec, widths=widths).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(COMMON_PROTOCOL["learning_rate"]))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = load_candidate_checkpoint(
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
        losses = []
        component_rows: list[dict[str, float]] = []
        for index in range(int(train_samples)):
            loss, parts = repaired_candidate_loss(
                model,
                spec,
                cached_sample("train", index, image_size),
                torch_device,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss candidate={spec.candidate_id} index={index}")
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            component_rows.append(parts)
            if (index + 1) % accumulation == 0 or index + 1 == train_samples:
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError(f"non-finite gradient candidate={spec.candidate_id} index={index}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_dice = []
        with torch.inference_mode():
            for index in range(int(validation_samples)):
                sample = cached_sample("validation", index, image_size)
                image = _tensor(sample, "image", torch_device).unsqueeze(0)
                target = _tensor(sample, "visible_fault_mask", torch_device).float().unsqueeze(0).unsqueeze(0)
                validation_dice.append(_dice_at_half(model(image), target))
        component_names = sorted({name for row in component_rows for name in row})
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_visible_dice_at_0_5": float(np.mean(validation_dice)),
            **{
                f"train_{name}": float(np.mean([item[name] for item in component_rows if name in item]))
                for name in component_names
            },
        }
        history.append(row)
        torch.save(_checkpoint_payload(model, optimizer, spec, epoch + 1, "IN_PROGRESS"), checkpoint_path)
        progress = {
            "status": "IN_PROGRESS",
            "run_hash": spec.run_hash,
            "candidate_id": spec.candidate_id,
            "epoch": epoch + 1,
            "epoch_budget": epoch_budget,
            "history": history,
            "expert_data_accessed": False,
            "old_test_samples_opened": 0,
            "new_test_samples_opened": 0,
        }
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"phase=method_repair_synthetic candidate={spec.candidate_id} "
            f"epoch={epoch + 1}/{epoch_budget} val_dice={row['validation_visible_dice_at_0_5']:.4f} "
            "expert=LOCKED test=LOCKED status=RUNNING"
        )

    torch.save(_checkpoint_payload(model, optimizer, spec, epoch_budget, "COMPLETE"), checkpoint_path)
    reloaded = build_candidate_model(spec, widths=widths)
    load_candidate_checkpoint(checkpoint_path, expected_hash=spec.run_hash, model=reloaded)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_hash": spec.run_hash,
        "candidate_id": spec.candidate_id,
        "epoch": epoch_budget,
        "epoch_budget": epoch_budget,
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "image_size": int(image_size),
        "widths": list(widths),
        "checkpoint_reload": "PASS",
        "history": history,
        "expert_data_accessed": False,
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
