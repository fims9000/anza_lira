"""Resumable expert-blind A1 low-label training."""

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
from .protocol import A1_ROOT, CHECKPOINT_ROOT, PROTOCOL, active_manifests, canonical_hash, protocol_hash, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def run_hash(variant: str) -> str:
    split, subsets = active_manifests()
    return canonical_hash({
        "variant": variant, "seed": 41, "fraction": "10pct",
        "protocol": protocol_hash(), "split": split["sha256"], "subsets": subsets["sha256"],
    })[:16]


def training_dataset() -> CRACKSMultiAnnotatorDataset:
    _split, subsets = active_manifests()
    section_ids = subsets["subsets"]["41"]["10pct"]
    return CRACKSMultiAnnotatorDataset(
        image_root=Path(PROTOCOL_ROOT()) / "data" / "cracks" / "images",
        annotation_root=Path(PROTOCOL_ROOT()) / "data" / "cracks" / "annotations",
        section_ids=section_ids,
        annotators=PROTOCOL["data"]["training_annotators"],
        mean=NORMALIZATION["mean"], std=NORMALIZATION["std"], crop_size=256,
        foreground_probability=float(PROTOCOL["training"]["foreground_crop_probability"]),
        annotators_per_section=int(PROTOCOL["training"]["annotators_per_section"]), seed=41,
    )


def PROTOCOL_ROOT() -> str:
    return str(Path(__file__).resolve().parents[1])


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, variant: str, epoch: int) -> None:
    torch.save({
        "status": "IN_PROGRESS", "variant": variant, "run_hash": run_hash(variant),
        "protocol_sha256": protocol_hash(), "seed": 41, "label_fraction": 0.10,
        "epoch": int(epoch), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "expert_data_accessed": False, "development_data_accessed": False,
    }, path)


def load_checkpoint(path: Path, variant: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "variant": variant, "run_hash": run_hash(variant), "protocol_sha256": protocol_hash(),
        "seed": 41, "label_fraction": 0.10, "expert_data_accessed": False,
        "development_data_accessed": False,
    }
    drift = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    if drift:
        raise ValueError(f"LEADS checkpoint provenance drift: {drift}")
    # Frozen H1 registers meshgrid views as deterministic buffers. On current
    # PyTorch those views cannot be copy_ targets because elements alias memory.
    # Recreate them from the constructor and load every learned/stateful value.
    state = dict(payload["model_state"])
    skipped = {key for key in state if key.endswith(".offset_x") or key.endswith(".offset_y")}
    for key in skipped:
        state.pop(key)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != skipped or incompatible.unexpected_keys:
        raise ValueError(
            f"LEADS checkpoint state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def train_variant(variant: str, *, device: str = "cuda") -> dict[str, Any]:
    run_dir = A1_ROOT / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_ROOT / f"{variant}-{run_hash(variant)}.pt"
    status_path = run_dir / "status.json"
    epochs = int(PROTOCOL["training"]["epochs"])
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("run_hash") == run_hash(variant) and int(status.get("epoch", 0)) == epochs:
            return {**status, "action": "SKIP"}
    set_seed(41)
    torch_device = torch.device(device)
    model = build_leads_model(variant).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(PROTOCOL["training"]["learning_rate"]),
        weight_decay=float(PROTOCOL["training"]["weight_decay"]),
    )
    dataset = training_dataset()
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, variant, model, optimizer)
        start_epoch = int(saved["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    start_time = time.monotonic()
    accumulation = int(PROTOCOL["training"]["effective_batch_size"])
    total_steps = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        dataset.set_epoch(epoch)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(41 + epoch)).tolist()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss: list[float] = []
        seg_losses: list[float] = []
        orient_losses: list[float] = []
        last_gradient_norm = 0.0
        for position, index in enumerate(order):
            batch = dataset[index]
            image = batch["image"].unsqueeze(0).to(torch_device)
            targets = batch["targets"].to(torch_device)
            weights = batch["weights"].to(torch_device)
            output = model(image, return_aux=True)
            segmentation, _ = average_annotator_loss(
                output["visible_logits"], targets, weights, topology_weight=0.2,
                topology_num_iters=int(PROTOCOL["training"]["topology_iterations"]),
            )
            bank, confidence = crowd_orientation_targets(
                targets, weights, radius=int(PROTOCOL["orientation_auxiliary"]["radius"]),
                min_neighbors=int(PROTOCOL["orientation_auxiliary"]["minimum_positive_neighbors"]),
                sigma_theta=float(PROTOCOL["orientation_auxiliary"]["sigma_theta"]),
            )
            orientation = crowd_orientation_loss(output["orientation_logits"], bank, confidence)
            loss = segmentation + float(PROTOCOL["orientation_auxiliary"]["weight"]) * orientation
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite LEADS loss for {variant}")
            (loss / accumulation).backward()
            epoch_loss.append(float(loss.detach()))
            seg_losses.append(float(segmentation.detach()))
            orient_losses.append(float(orientation.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ValueError(f"non-finite LEADS gradient for {variant}")
                last_gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_steps += 1
        row = {
            "epoch": epoch + 1, "loss": float(np.mean(epoch_loss)),
            "segmentation_loss": float(np.mean(seg_losses)), "orientation_loss": float(np.mean(orient_losses)),
            "gradient_norm_last_step": last_gradient_norm, "optimizer_steps_cumulative": total_steps,
        }
        history.append(row)
        _save_checkpoint(checkpoint, model, optimizer, variant=variant, epoch=epoch + 1)
        write_json(status_path, {
            "status": "IN_PROGRESS", "variant": variant, "run_hash": run_hash(variant),
            "protocol_sha256": protocol_hash(), "epoch": epoch + 1, "epoch_budget": epochs,
            "training_sections": len(dataset), "history": history, "checkpoint": str(checkpoint),
            "expert_data_accessed": False, "calibration_data_accessed": False,
            "development_data_accessed": False,
        })
        print(
            f"phase=ANZA-LEADS-A1 variant={variant} seed=41 labels=10pct epoch={epoch + 1}/{epochs} "
            f"loss={row['loss']:.5f} status=RUNNING", flush=True,
        )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["status"] = "COMPLETE"
    torch.save(saved, checkpoint)
    reloaded = build_leads_model(variant)
    load_checkpoint(checkpoint, variant, reloaded)
    wall = time.monotonic() - start_time
    result = {
        "status": "COMPLETE", "action": "RUN", "variant": variant, "run_hash": run_hash(variant),
        "protocol_sha256": protocol_hash(), "seed": 41, "label_fraction": 0.10,
        "epoch": epochs, "epoch_budget": epochs, "training_sections": len(dataset),
        "optimizer_steps": sum(math_steps(len(dataset), accumulation) for _ in range(epochs)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "wall_time_seconds_this_invocation": float(wall),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_reload": "PASS", "history": history, "expert_data_accessed": False,
        "calibration_data_accessed_during_training": False, "development_data_accessed": False,
    }
    write_json(status_path, result)
    return result


def math_steps(items: int, accumulation: int) -> int:
    return (int(items) + int(accumulation) - 1) // int(accumulation)


def one_batch_smoke(*, device: str = "cuda") -> dict[str, Any]:
    """Exercise the exact real-data objective once without updating weights."""
    dataset = training_dataset()
    dataset.set_epoch(0)
    batch = dataset[0]
    image = batch["image"].unsqueeze(0).to(device)
    targets = batch["targets"].to(device)
    weights = batch["weights"].to(device)
    result = {}
    for variant in ("L0_backbone", "L1_isotropic", "L2_generic_aniso", "L3_anza_hs"):
        set_seed(41)
        model = build_leads_model(variant).to(device)
        output = model(image, return_aux=True)
        segmentation, _ = average_annotator_loss(
            output["visible_logits"], targets, weights, topology_weight=0.2,
            topology_num_iters=int(PROTOCOL["training"]["topology_iterations"]),
        )
        bank, confidence = crowd_orientation_targets(targets, weights)
        orientation = crowd_orientation_loss(output["orientation_logits"], bank, confidence)
        loss = segmentation + float(PROTOCOL["orientation_auxiliary"]["weight"]) * orientation
        loss.backward()
        result[variant] = {
            "loss": float(loss.detach()), "segmentation_loss": float(segmentation.detach()),
            "orientation_loss": float(orientation.detach()),
            "finite_gradients": bool(torch.isfinite(loss) and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
            )),
        }
        del model
    return result
