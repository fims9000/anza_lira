"""Resumable, fail-closed C0--C3 training on CrossingTraceBench-v4."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from affinity_repair.matrix import AFFINITY_PROTOCOL, AffinityRepairSpec
from models.azconv import AZConv2d
from models.azconv_affinity import LOCAL8_OFFSETS, RADIUS2_OFFSETS, StructuralAffinityAZConv2d
from models.segmentation_affinity import build_affinity_model
from models.segmentation_v2 import build_comparable_model
from synthetic.affinity_losses import (
    balanced_affinity_bce,
    configure_affinity_stage1,
    matched_hard_negative_ranking,
)
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v4 import generate_sample_v4
from synthetic.structural_losses import visible_segmentation_loss


def set_affinity_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=2048)
def cached_v4_sample(split: str, index: int, image_size: int) -> dict[str, Any]:
    return generate_sample_v4(split, index, image_size=image_size)


def _soft_skeletonize(probability: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    def erode(value: torch.Tensor) -> torch.Tensor:
        vertical = -F.max_pool2d(-value, (3, 1), 1, (1, 0))
        horizontal = -F.max_pool2d(-value, (1, 3), 1, (0, 1))
        return torch.minimum(vertical, horizontal)

    def opened(value: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(erode(value), 3, 1, 1)

    value = probability.clamp(0.0, 1.0)
    skeleton = F.relu(value - opened(value))
    for _ in range(max(0, int(iterations) - 1)):
        value = erode(value)
        delta = F.relu(value - opened(value))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    truth = target.to(dtype=probability.dtype)
    skeleton_prediction = _soft_skeletonize(probability)
    skeleton_truth = _soft_skeletonize(truth)
    tprec = (skeleton_prediction * truth).sum() / skeleton_prediction.sum().clamp_min(1e-8)
    tsens = (skeleton_truth * probability).sum() / skeleton_truth.sum().clamp_min(1e-8)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens).clamp_min(1e-8)


def _batch_plan(sample_count: int, batch_size: int, *, seed: int) -> list[list[int]]:
    if sample_count != 512 or batch_size != 4:
        indices = list(range(sample_count))
        random.Random(seed).shuffle(indices)
        return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
    paired = [[first, 128 + first, second, 128 + second] for first, second in zip(range(0, 128, 2), range(1, 128, 2))]
    context = [list(range(start, start + 4)) for start in range(256, 512, 4)]
    batches = paired + context
    random.Random(seed).shuffle(batches)
    return batches


def _batch(split: str, indices: Iterable[int], image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    samples = [cached_v4_sample(split, int(index), image_size) for index in indices]
    images = torch.stack([torch.as_tensor(sample["image"]) for sample in samples]).to(device)
    targets = torch.stack([
        torch.as_tensor(sample["visible_fault_mask"], dtype=torch.float32) for sample in samples
    ]).unsqueeze(1).to(device)
    return images, targets, samples


def _edge_targets(samples: list[dict[str, Any]], offsets: tuple[tuple[int, int], ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    targets = [build_affinity_targets(sample, offsets) for sample in samples]
    positive = torch.stack([torch.as_tensor(item["affinity_positive"]) for item in targets]).to(device)
    negative = torch.stack([torch.as_tensor(item["affinity_hard_negative"]) for item in targets]).to(device)
    return positive, negative


def _affinity_layer(model: torch.nn.Module) -> StructuralAffinityAZConv2d:
    layers = [module for module in model.modules() if isinstance(module, StructuralAffinityAZConv2d)]
    if len(layers) != 1:
        raise ValueError(f"expected exactly one affinity layer, found {len(layers)}")
    return layers[0]


def build_candidate(
    spec: AffinityRepairSpec,
    *,
    widths: tuple[int, int, int, int],
    clean_state: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    set_affinity_seed(spec.seed)
    v1 = build_comparable_model("anza_v1", widths=widths)
    model = build_affinity_model(spec.candidate_id, widths=widths, seed_matched_v1=v1)
    if clean_state is not None:
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        allowed = ("context_encoder", "affinity_mlp", "raw_beta")
        if unexpected or any(not any(token in key for token in allowed) for key in missing):
            raise ValueError(f"clean C1 state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: AffinityRepairSpec,
    *,
    stage: str,
    epoch: int,
    clean_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "run_hash": spec.run_hash,
        "spec": asdict(spec),
        "protocol": AFFINITY_PROTOCOL,
        "stage": stage,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "clean_checkpoint_sha256": clean_checkpoint_sha256,
        "expert_data_accessed": False,
        "legacy_test_samples_opened": 0,
        "v4_test_samples_opened": 0,
        "cracks_samples_opened": 0,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: Path,
    *,
    spec: AffinityRepairSpec,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    clean_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_hash") != spec.run_hash:
        raise ValueError("affinity-repair checkpoint hash mismatch")
    if payload.get("clean_checkpoint_sha256") != clean_checkpoint_sha256:
        raise ValueError("affinity-repair clean C1 dependency hash mismatch")
    for field in ("expert_data_accessed",):
        if payload.get(field) is not False:
            raise ValueError(f"checkpoint lock violation: {field}")
    for field in ("legacy_test_samples_opened", "v4_test_samples_opened", "cracks_samples_opened"):
        if payload.get(field) != 0:
            raise ValueError(f"checkpoint lock violation: {field}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _optimizer(model: torch.nn.Module, spec: AffinityRepairSpec, stage: str) -> torch.optim.Optimizer:
    if stage == "S1":
        trainable, _frozen = configure_affinity_stage1(model)
        return torch.optim.Adam(trainable, lr=float(AFFINITY_PROTOCOL["affinity_learning_rate"]))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if not spec.affinity:
        return torch.optim.Adam(model.parameters(), lr=float(AFFINITY_PROTOCOL["affinity_learning_rate"]))
    affinity_parameters: list[torch.nn.Parameter] = []
    layer = _affinity_layer(model)
    for child in (layer.context_encoder, layer.affinity_mlp):
        affinity_parameters.extend(child.parameters())
    affinity_parameters.append(layer.raw_beta)
    affinity_ids = {id(parameter) for parameter in affinity_parameters}
    base = [parameter for parameter in model.parameters() if id(parameter) not in affinity_ids]
    return torch.optim.Adam(
        [
            {"params": base, "lr": float(AFFINITY_PROTOCOL["base_learning_rate"])},
            {"params": affinity_parameters, "lr": float(AFFINITY_PROTOCOL["affinity_learning_rate"])},
        ]
    )


@torch.no_grad()
def project_affinity_constraints(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, StructuralAffinityAZConv2d):
            module.raw_beta.clamp_(min=0.0)


def _epoch(
    model: torch.nn.Module,
    spec: AffinityRepairSpec,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    epoch: int,
    sample_count: int,
    image_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for indices in _batch_plan(sample_count, 4, seed=spec.seed * 1000 + epoch):
        images, visible, samples = _batch("train", indices, image_size, device)
        optimizer.zero_grad(set_to_none=True)
        if stage == "S1":
            layer = _affinity_layer(model)
            edge = layer.edge_logits(images, include_radius2=spec.radius2)
            offsets = LOCAL8_OFFSETS + (RADIUS2_OFFSETS if spec.radius2 else ())
            positive, negative = _edge_targets(samples, offsets, device)
            affinity_loss = balanced_affinity_bce(edge["logits"], positive, negative)
            ranking = matched_hard_negative_ranking(
                edge["logits"], positive, negative, margin=float(AFFINITY_PROTOCOL["ranking_margin"])
            ) if spec.hard_ranking else affinity_loss.new_zeros(())
            loss = affinity_loss + float(AFFINITY_PROTOCOL["ranking_loss_weight"]) * ranking
            parts = {"affinity": float(affinity_loss.detach()), "ranking": float(ranking.detach())}
        else:
            if spec.affinity:
                output = model(images, return_diagnostics=True)
                logits = output["visible_logits"]
            else:
                logits = model(images)
            segmentation = visible_segmentation_loss(logits, visible) + 0.2 * soft_cldice_loss(logits, visible)
            loss = segmentation
            parts = {"segmentation": float(segmentation.detach())}
            if spec.affinity:
                layer = _affinity_layer(model)
                diagnostics = output["affinity_diagnostics"]
                edge = diagnostics["radius2_affinity"] if spec.radius2 else diagnostics["affinity"]
                offsets = LOCAL8_OFFSETS + (RADIUS2_OFFSETS if spec.radius2 else ())
                positive, negative = _edge_targets(samples, offsets, device)
                affinity_loss = balanced_affinity_bce(edge["logits"], positive, negative)
                ranking = matched_hard_negative_ranking(
                    edge["logits"], positive, negative, margin=float(AFFINITY_PROTOCOL["ranking_margin"])
                ) if spec.hard_ranking else affinity_loss.new_zeros(())
                loss = loss + float(AFFINITY_PROTOCOL["affinity_loss_weight"]) * affinity_loss
                loss = loss + float(AFFINITY_PROTOCOL["ranking_loss_weight"]) * ranking
                parts.update({"affinity": float(affinity_loss.detach()), "ranking": float(ranking.detach())})
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss {spec.candidate_id} {stage} epoch={epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError(f"non-finite gradient {spec.candidate_id} {stage} epoch={epoch}")
        optimizer.step()
        project_affinity_constraints(model)
        rows.append({"total": float(loss.detach()), **parts})
    names = sorted({name for row in rows for name in row})
    return {name: float(np.mean([row[name] for row in rows if name in row])) for name in names}


def _validation_dice(model: torch.nn.Module, *, sample_count: int, image_size: int, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, sample_count, 8):
            images, target, _samples = _batch("validation", range(start, min(start + 8, sample_count)), image_size, device)
            probability = torch.sigmoid(model(images)) >= 0.5
            truth = target >= 0.5
            intersection = (probability & truth).sum(dim=(1, 2, 3)).float()
            denominator = probability.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
            values.extend(torch.where(denominator > 0, 2 * intersection / denominator, torch.ones_like(intersection)).cpu().tolist())
    return float(np.mean(values))


def run_candidate(
    spec: AffinityRepairSpec,
    output_root: Path,
    *,
    device: str = "cuda",
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    train_samples: int = 512,
    validation_samples: int = 512,
    image_size: int = 128,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
    clean_checkpoint: Path | None = None,
) -> dict[str, Any]:
    s1_budget = int(AFFINITY_PROTOCOL["stage1_epochs"] if stage1_epochs is None else stage1_epochs) if spec.affinity else 0
    s2_budget = int(AFFINITY_PROTOCOL["stage2_epochs"] if stage2_epochs is None else stage2_epochs)
    run_dir = Path(output_root) / f"{spec.candidate_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    heartbeat_path = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("run_hash") == spec.run_hash:
            return {**status, "action": "SKIP"}

    clean_state = None
    clean_sha = None
    if spec.affinity:
        if clean_checkpoint is None or not Path(clean_checkpoint).exists():
            raise ValueError(f"{spec.candidate_id} requires completed C1 checkpoint")
        clean_payload = torch.load(clean_checkpoint, map_location="cpu", weights_only=False)
        if clean_payload.get("spec", {}).get("candidate_id") != "C1":
            raise ValueError("affinity candidate requires C1 checkpoint")
        clean_state = clean_payload["model_state"]
        clean_sha = checkpoint_sha256(clean_checkpoint)
    model = build_candidate(spec, widths=widths, clean_state=clean_state).to(torch.device(device))
    history: list[dict[str, Any]] = []
    resume = None
    if checkpoint_path.exists():
        resume = load_checkpoint(
            checkpoint_path, spec=spec, model=model, clean_checkpoint_sha256=clean_sha
        )
        history = list(json.loads(status_path.read_text()).get("history", [])) if status_path.exists() else []

    stages = (("S1", s1_budget), ("S2", s2_budget)) if spec.affinity else (("S2", s2_budget),)
    for stage, budget in stages:
        optimizer = _optimizer(model, spec, stage)
        start = 0
        if resume is not None and resume.get("stage") == stage:
            optimizer.load_state_dict(resume["optimizer_state"])
            start = int(resume["epoch"])
        elif resume is not None and stage == "S1" and resume.get("stage") == "S2":
            continue
        for epoch in range(start, budget):
            losses = _epoch(
                model, spec, optimizer, stage=stage, epoch=epoch, sample_count=train_samples,
                image_size=image_size, device=torch.device(device),
            )
            validation_dice = _validation_dice(
                model, sample_count=validation_samples, image_size=image_size, device=torch.device(device)
            ) if stage == "S2" else None
            row = {"stage": stage, "epoch": epoch + 1, **losses, "validation_visible_dice_at_0_5": validation_dice}
            history.append(row)
            torch.save(_checkpoint_payload(
                model, optimizer, spec, stage=stage, epoch=epoch + 1, clean_checkpoint_sha256=clean_sha
            ), checkpoint_path)
            progress = {
                "status": "IN_PROGRESS", "run_hash": spec.run_hash, "candidate_id": spec.candidate_id,
                "stage": stage, "epoch": epoch + 1, "stage_budget": budget, "history": history,
                "image_size": image_size, "widths": list(widths), "clean_checkpoint_sha256": clean_sha,
                "expert_data_accessed": False, "legacy_test_samples_opened": 0,
                "v4_test_samples_opened": 0, "cracks_samples_opened": 0,
            }
            status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
            with heartbeat_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"phase=affinity_repair candidate={spec.candidate_id} stage={stage} "
                f"epoch={epoch + 1}/{budget} val_dice={validation_dice if validation_dice is not None else 'NA'} "
                "expert=LOCKED test_v4=LOCKED status=RUNNING",
                flush=True,
            )
        resume = None

    final_optimizer = _optimizer(model, spec, "S2")
    torch.save(_checkpoint_payload(
        model, final_optimizer, spec, stage="S2", epoch=s2_budget, clean_checkpoint_sha256=clean_sha
    ), checkpoint_path)
    result = {
        "status": "COMPLETE", "action": "RUN", "run_hash": spec.run_hash,
        "candidate_id": spec.candidate_id, "stage": "S2", "epoch": s2_budget,
        "stage1_epochs": s1_budget, "stage2_epochs": s2_budget,
        "train_samples": train_samples, "validation_samples": validation_samples,
        "image_size": image_size, "widths": list(widths), "history": history,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "clean_checkpoint_sha256": clean_sha, "checkpoint_reload": "PENDING",
        "expert_data_accessed": False, "legacy_test_samples_opened": 0,
        "v4_test_samples_opened": 0, "cracks_samples_opened": 0,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # Verify the exact persisted state and dependency lock.
    reloaded = build_candidate(spec, widths=widths, clean_state=clean_state)
    load_checkpoint(checkpoint_path, spec=spec, model=reloaded, clean_checkpoint_sha256=clean_sha)
    result["checkpoint_reload"] = "PASS"
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
