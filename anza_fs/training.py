"""Resumable identical-budget ANZA-FS H3 training and inference."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anza_hs.orientation_bank import orientation_bank_loss
from synthetic.structural_losses import visible_segmentation_loss

from .foliation_conv import ANZAFoliationConv, FreeFoliationConv
from .hard_bench_v6 import SPLIT_SIZE, generate_hard_sample
from .model import build_h3_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@lru_cache(maxsize=4096)
def cached_sample(split: str, index: int) -> dict[str, Any]:
    return generate_hard_sample(split, int(index))


class HardBenchDataset(Dataset):
    def __init__(self, split: str, indices: list[int]) -> None:
        self.records = []
        for index in indices:
            sample = cached_sample(split, index)
            self.records.append({
                "image": torch.from_numpy(sample["image"]),
                "target": torch.from_numpy(np.asarray(sample["visible_fault_mask"], dtype=np.float32))[None],
                "bank": torch.from_numpy(np.asarray(sample["orientation_bank_target"], dtype=np.float32)),
                "valid": torch.from_numpy(np.asarray(sample["orientation_valid"], dtype=np.float32))[None],
                "index": index,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return self.records[index]


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
    topological_precision = (skeleton_prediction * truth).sum() / skeleton_prediction.sum().clamp_min(1e-8)
    topological_sensitivity = (skeleton_truth * probability).sum() / skeleton_truth.sum().clamp_min(1e-8)
    return 1.0 - 2.0 * topological_precision * topological_sensitivity / (topological_precision + topological_sensitivity).clamp_min(1e-8)


def common_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return visible_segmentation_loss(logits, target) + 0.25 * soft_cldice_loss(logits, target)


def run_hash(protocol_hash: str, variant: str, seed: int) -> str:
    return hashlib.sha256(f"{protocol_hash}:{variant}:seed{seed}".encode()).hexdigest()[:16]


def _checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, variant: str, protocol_hash: str, seed: int) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "variant": variant,
        "protocol_hash": protocol_hash,
        "seed": seed,
        "run_hash": run_hash(protocol_hash, variant, seed),
    }, path)


def train_variant(
    variant: str,
    *,
    protocol: dict[str, Any],
    protocol_hash: str,
    output_root: Path,
    checkpoint_root: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    settings = protocol["training"]
    epochs = int(settings["epochs"])
    seed = int(settings["seed"])
    run_dir = output_root / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{variant}-{run_hash(protocol_hash, variant, seed)}.pt"
    status_path = run_dir / "status.json"
    heartbeat = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("protocol_sha256") == protocol_hash and existing.get("epoch") == epochs:
            return {**existing, "action": "SKIP"}
    set_seed(seed)
    torch_device = torch.device(device)
    model = build_h3_model(variant).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        if saved.get("protocol_hash") != protocol_hash or saved.get("variant") != variant or saved.get("seed") != seed:
            raise ValueError("H3 checkpoint protocol, variant, or seed mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))
    dataset = HardBenchDataset("train", list(range(int(settings["train_samples"]))))
    for epoch in range(start_epoch, epochs):
        loader = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=0,
            pin_memory=torch_device.type == "cuda",
        )
        model.train()
        losses: list[float] = []
        segmentation_losses: list[float] = []
        orientation_losses: list[float] = []
        for batch in loader:
            image = batch["image"].to(torch_device, non_blocking=True)
            target = batch["target"].to(torch_device, non_blocking=True)
            bank = batch["bank"].to(torch_device, non_blocking=True)
            valid = batch["valid"].to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(image, return_aux=True)
            segmentation = common_segmentation_loss(output["visible_logits"], target)
            orientation = orientation_bank_loss(
                output["orientation_logits"], bank, valid,
                background_weight=float(protocol["orientation_target"]["background_weight"]),
            )
            loss = segmentation + float(settings["orientation_loss_weight"]) * orientation
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite H3 loss: {variant}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError(f"non-finite H3 gradient: {variant}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            segmentation_losses.append(float(segmentation.detach()))
            orientation_losses.append(float(orientation.detach()))
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "segmentation_loss": float(np.mean(segmentation_losses)),
            "orientation_loss": float(np.mean(orientation_losses)),
        }
        history.append(row)
        _checkpoint(checkpoint_path, model, optimizer, epoch + 1, variant, protocol_hash, seed)
        progress = {
            "status": "IN_PROGRESS",
            "variant": variant,
            "run_hash": run_hash(protocol_hash, variant, seed),
            "protocol_sha256": protocol_hash,
            "seed": seed,
            "epoch": epoch + 1,
            "epoch_budget": epochs,
            "history": history,
            "checkpoint": str(checkpoint_path),
            "confirm_opened": False,
            "cracks_accessed": False,
            "expert_accessed": False,
        }
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=ANZA-FS-H3 variant={variant} seed={seed} epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} status=RUNNING", flush=True)
    reloaded = build_h3_model(variant).to(torch_device)
    saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    reloaded.load_state_dict(saved["model"])
    reloaded.eval()
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "variant": variant,
        "run_hash": run_hash(protocol_hash, variant, seed),
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "epoch": epochs,
        "epoch_budget": epochs,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_reload": "PASS",
        "confirm_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _orientation_diagnostics(logits: list[torch.Tensor], bank: torch.Tensor, valid: torch.Tensor) -> tuple[list[float], list[float]]:
    accuracies: list[float] = []
    entropies: list[float] = []
    for prediction in logits:
        local_target = F.interpolate(bank, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        local_valid = F.interpolate(valid, size=prediction.shape[-2:], mode="nearest").bool()[:, 0]
        predicted_index = prediction.argmax(dim=1)
        selected = local_target.gather(1, predicted_index[:, None])[:, 0]
        accuracies.append(float((selected[local_valid] >= 0.5).float().mean()) if local_valid.any() else 1.0)
        evidence = torch.sigmoid(prediction).clamp(1e-6, 1 - 1e-6)
        entropy = -(evidence * torch.log(evidence) + (1 - evidence) * torch.log(1 - evidence)) / math.log(2.0)
        entropies.append(float(entropy.mean()))
    return accuracies, entropies


def predict_variant(variant: str, checkpoint_path: Path, *, device: str = "cuda") -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    torch_device = torch.device(device)
    model = build_h3_model(variant).to(torch_device)
    saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    samples = [cached_sample("calibration", index) for index in range(SPLIT_SIZE["calibration"])]
    samples += [cached_sample("development", index) for index in range(SPLIT_SIZE["development"])]
    probabilities: list[np.ndarray] = []
    orientation_accuracy: list[float] = []
    evidence_entropy: list[float] = []
    response_stats: dict[str, list[float]] = {"center_mean": [], "longitudinal_minus_center_mean": [], "center_minus_transverse_mean": [], "center_minus_transverse_positive_fraction": []}
    with torch.inference_mode():
        for start in range(0, len(samples), 8):
            local = samples[start : start + 8]
            image = torch.from_numpy(np.stack([sample["image"] for sample in local])).to(torch_device)
            bank = torch.from_numpy(np.stack([sample["orientation_bank_target"] for sample in local])).to(torch_device)
            valid = torch.from_numpy(np.stack([sample["orientation_valid"] for sample in local]))[:, None].float().to(torch_device)
            output = model(image, return_aux=True, operator_diagnostics=variant in {"F2_free_foliation", "F3_anza_fs"})
            probabilities.extend(torch.sigmoid(output["visible_logits"]).cpu().numpy()[:, 0].astype(np.float32))
            accuracy, entropy = _orientation_diagnostics(output["orientation_logits"], bank, valid)
            orientation_accuracy.extend(accuracy)
            evidence_entropy.extend(entropy)
            for aux in output["operator_aux"]:
                center = aux["center"]
                longitudinal = aux["longitudinal_minus_center"]
                transverse = aux["center_minus_transverse"]
                response_stats["center_mean"].append(float(center.mean()))
                response_stats["longitudinal_minus_center_mean"].append(float(longitudinal.mean()))
                response_stats["center_minus_transverse_mean"].append(float(transverse.mean()))
                response_stats["center_minus_transverse_positive_fraction"].append(float((transverse > 0).float().mean()))
    stage_geometry = []
    gammas = []
    for block in (model.bank_quarter, model.bank_half):
        if block is None:
            continue
        gammas.append(float(block.gamma.detach()))
        if isinstance(block, (FreeFoliationConv, ANZAFoliationConv)):
            stage_geometry.append([float(value.detach()) for value in block.geometry()])
        else:
            sigma_u, sigma_s = block.scales()
            stage_geometry.append({"sigma_u_mean": float(sigma_u.mean()), "sigma_s_mean": float(sigma_s.mean())})
    diagnostics = {
        "orientation_accuracy": float(np.mean(orientation_accuracy)) if orientation_accuracy else None,
        "average_evidence_entropy": float(np.mean(evidence_entropy)) if evidence_entropy else None,
        "gamma_by_stage": gammas,
        "geometry_by_stage": stage_geometry,
        "responses": {key: (float(np.mean(values)) if values else None) for key, values in response_stats.items()},
    }
    return probabilities, samples, diagnostics


def one_batch_smoke(protocol: dict[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    dataset = HardBenchDataset("train", list(range(8)))
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    result = {}
    for variant in protocol["matrix"]:
        set_seed(int(protocol["training"]["seed"]))
        model = build_h3_model(variant).to(device)
        model.train()
        output = model(batch["image"].to(device), return_aux=True)
        segmentation = common_segmentation_loss(output["visible_logits"], batch["target"].to(device))
        orientation = orientation_bank_loss(output["orientation_logits"], batch["bank"].to(device), batch["valid"].to(device))
        loss = segmentation + float(protocol["training"]["orientation_loss_weight"]) * orientation
        loss.backward()
        finite = bool(torch.isfinite(loss) and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        result[variant] = {"loss": float(loss.detach()), "finite_gradients": finite, "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    return result
