"""Exact seed-41 LEADS training on the frozen RC1 cross-fit sections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from cracks_experiment.partial_label_training import NORMALIZATION
from cracks_experiment.partial_labels import CRACKSMultiAnnotatorDataset, average_annotator_loss

from .model import build_leads_model
from .orientation import crowd_orientation_loss, crowd_orientation_targets
from .protocol import PROTOCOL as PARENT_PROTOCOL, canonical_hash, write_json
from .rc1_protocol import CHECKPOINT_ROOT, ROOT, VARIANTS, load_frozen


def set_seed(seed: int = 41) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def run_hash(variant: str) -> str:
    protocol, split = load_frozen()
    return canonical_hash({
        "variant": variant, "seed": 41, "fraction": 0.10,
        "protocol": canonical_hash(protocol), "split": split["sha256"],
        "optimization_sections": split["optimization_10pct_seed41"],
    })[:16]


def training_dataset() -> CRACKSMultiAnnotatorDataset:
    protocol, split = load_frozen()
    project = Path(__file__).resolve().parents[1]
    return CRACKSMultiAnnotatorDataset(
        image_root=project / "data" / "cracks" / "images",
        annotation_root=project / "data" / "cracks" / "annotations",
        section_ids=split["optimization_10pct_seed41"],
        annotators=PARENT_PROTOCOL["data"]["training_annotators"],
        mean=NORMALIZATION["mean"], std=NORMALIZATION["std"],
        crop_size=int(protocol["training"]["crop_size"]),
        foreground_probability=float(protocol["training"]["foreground_crop_probability"]),
        annotators_per_section=int(protocol["training"]["annotators_per_section"]), seed=41,
    )


def _checkpoint_payload(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, variant: str, epoch: int,
) -> dict[str, Any]:
    protocol, split = load_frozen()
    return {
        "status": "IN_PROGRESS", "variant": variant, "run_hash": run_hash(variant),
        "protocol_sha256": canonical_hash(protocol), "split_sha256": split["sha256"],
        "seed": 41, "label_fraction": 0.10, "epoch": int(epoch),
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "expert_data_accessed": False, "calibration_data_accessed": False,
        "development_data_accessed": False,
    }


def load_checkpoint(
    path: Path, variant: str, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    protocol, split = load_frozen()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "variant": variant, "run_hash": run_hash(variant),
        "protocol_sha256": canonical_hash(protocol), "split_sha256": split["sha256"],
        "seed": 41, "label_fraction": 0.10, "expert_data_accessed": False,
        "calibration_data_accessed": False, "development_data_accessed": False,
    }
    drift = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    if drift:
        raise ValueError(f"RC1 checkpoint provenance drift: {drift}")
    state = dict(payload["model_state"])
    skipped = {key for key in state if key.endswith(".offset_x") or key.endswith(".offset_y")}
    for key in skipped:
        state.pop(key)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != skipped or incompatible.unexpected_keys:
        raise ValueError(f"RC1 state mismatch: {incompatible}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def train_variant(variant: str, *, device: str = "cuda") -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"variant outside RC1 matrix: {variant}")
    protocol, _split = load_frozen()
    run_dir = ROOT / "training" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_ROOT / f"{variant}-{run_hash(variant)}.pt"
    status_path = run_dir / "status.json"
    epochs = int(protocol["training"]["epochs"])
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("run_hash") == run_hash(variant):
            return {**status, "action": "SKIP"}

    set_seed()
    torch_device = torch.device(device)
    model = build_leads_model(variant).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    dataset = training_dataset()
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, variant, model, optimizer)
        start_epoch = int(saved["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))
    accumulation = int(protocol["training"]["effective_batch_size"])
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    started = time.monotonic()
    optimizer_steps = start_epoch * ((len(dataset) + accumulation - 1) // accumulation)
    for epoch in range(start_epoch, epochs):
        model.train(); dataset.set_epoch(epoch)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(41 + epoch)).tolist()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []; segs: list[float] = []; orientations: list[float] = []
        last_norm = 0.0
        for position, index in enumerate(order):
            batch = dataset[index]
            image = batch["image"].unsqueeze(0).to(torch_device)
            targets = batch["targets"].to(torch_device); weights = batch["weights"].to(torch_device)
            output = model(image, return_aux=True)
            segmentation, _ = average_annotator_loss(
                output["visible_logits"], targets, weights, topology_weight=0.2,
                topology_num_iters=int(PARENT_PROTOCOL["training"]["topology_iterations"]),
            )
            bank, confidence = crowd_orientation_targets(
                targets, weights, radius=int(PARENT_PROTOCOL["orientation_auxiliary"]["radius"]),
                min_neighbors=int(PARENT_PROTOCOL["orientation_auxiliary"]["minimum_positive_neighbors"]),
                sigma_theta=float(PARENT_PROTOCOL["orientation_auxiliary"]["sigma_theta"]),
            )
            orientation = crowd_orientation_loss(output["orientation_logits"], bank, confidence)
            loss = segmentation + float(PARENT_PROTOCOL["orientation_auxiliary"]["weight"]) * orientation
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite RC1 loss: {variant}")
            (loss / accumulation).backward()
            losses.append(float(loss.detach())); segs.append(float(segmentation.detach())); orientations.append(float(orientation.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ValueError(f"non-finite RC1 gradient: {variant}")
                last_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        history.append({
            "epoch": epoch + 1, "loss": float(np.mean(losses)),
            "segmentation_loss": float(np.mean(segs)), "orientation_loss": float(np.mean(orientations)),
            "gradient_norm_last_step": last_norm, "optimizer_steps_cumulative": optimizer_steps,
        })
        torch.save(_checkpoint_payload(model, optimizer, variant=variant, epoch=epoch + 1), checkpoint)
        write_json(status_path, {
            "status": "IN_PROGRESS", "variant": variant, "run_hash": run_hash(variant),
            "epoch": epoch + 1, "epoch_budget": epochs, "training_sections": len(dataset),
            "history": history, "checkpoint": str(checkpoint), "expert_data_accessed": False,
            "calibration_data_accessed": False, "development_data_accessed": False,
        })
        print(f"phase=ANZA-LEADS-RC1 variant={variant} epoch={epoch + 1}/{epochs} loss={history[-1]['loss']:.5f}", flush=True)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False); saved["status"] = "COMPLETE"; torch.save(saved, checkpoint)
    reload_model = build_leads_model(variant); load_checkpoint(checkpoint, variant, reload_model)
    result = {
        "status": "COMPLETE", "action": "RUN", "variant": variant, "run_hash": run_hash(variant),
        "protocol_sha256": canonical_hash(protocol), "seed": 41, "label_fraction": 0.10,
        "epoch": epochs, "epoch_budget": epochs, "training_sections": len(dataset),
        "optimizer_steps": optimizer_steps,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "trainable_parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "wall_time_seconds_this_invocation": float(time.monotonic() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_reload": "PASS", "history": history, "expert_data_accessed": False,
        "calibration_data_accessed_during_training": False, "development_data_accessed": False,
    }
    write_json(status_path, result)
    return result


def one_batch_smoke(*, device: str = "cuda") -> dict[str, Any]:
    protocol, _ = load_frozen(); dataset = training_dataset(); dataset.set_epoch(0); batch = dataset[0]
    image = batch["image"].unsqueeze(0).to(device); targets = batch["targets"].to(device); weights = batch["weights"].to(device)
    result = {}
    for variant in VARIANTS:
        set_seed(); model = build_leads_model(variant).to(device); output = model(image, return_aux=True)
        segmentation, _ = average_annotator_loss(
            output["visible_logits"], targets, weights, topology_weight=0.2,
            topology_num_iters=int(PARENT_PROTOCOL["training"]["topology_iterations"]),
        )
        bank, confidence = crowd_orientation_targets(targets, weights)
        orientation = crowd_orientation_loss(output["orientation_logits"], bank, confidence)
        loss = segmentation + 0.10 * orientation; loss.backward()
        result[variant] = {"loss": float(loss.detach()), "finite_gradients": bool(torch.isfinite(loss) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ))}
    return result
