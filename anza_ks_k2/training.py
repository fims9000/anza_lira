"""Resumable identical-budget K2 seed-41 training."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anza_fs.training import common_segmentation_loss

from .benchmark import SPLIT_SIZES, generate_sample
from .model import METHOD, VARIANTS, build_k2_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class K2Dataset(Dataset):
    def __init__(self, split: str, indices: list[int]) -> None:
        self.split = split; self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        index = self.indices[item]
        sample = generate_sample(self.split, index)
        return {
            "image": torch.from_numpy(sample["image"]),
            "target": torch.from_numpy(sample["target"])[None],
            "distractor": torch.from_numpy(sample["distractor"])[None],
            "bank": torch.from_numpy(sample["orientation_bank"]),
            "valid": torch.from_numpy(sample["orientation_valid"])[None],
            "index": index,
        }


def load_feature_norm(path: Path, variant: str) -> dict[str, torch.Tensor] | None:
    if variant == "M0_backbone":
        return None
    values = json.loads(path.read_text())["methods"][METHOD[variant]]
    return {"mean": torch.tensor(values["mean"], dtype=torch.float32), "std": torch.tensor(values["std"], dtype=torch.float32)}


def orientation_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    local_target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    local_valid = F.interpolate(valid, size=logits.shape[-2:], mode="nearest")
    loss = F.binary_cross_entropy_with_logits(logits, local_target, reduction="none") * local_valid
    return loss.sum() / (local_valid.sum() * logits.shape[1]).clamp_min(1.0)


def balanced_occupancy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    local = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    loss = F.binary_cross_entropy_with_logits(logits, local, reduction="none")
    positive = local > 0.5; negative = ~positive
    zero = loss.sum() * 0.0
    return 0.5 * (loss[positive].mean() if positive.any() else zero) + 0.5 * (loss[negative].mean() if negative.any() else zero)


def _run_hash(protocol_hash: str, variant: str, seed: int) -> str:
    return hashlib.sha256(f"{protocol_hash}:{variant}:seed{seed}".encode()).hexdigest()[:16]


def train_variant(
    variant: str,
    *,
    protocol: dict[str, Any],
    protocol_hash: str,
    feature_norm_path: Path,
    result_root: Path,
    checkpoint_root: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    seed = int(protocol["seed"]); epochs = int(protocol["epochs"]); batch_size = int(protocol["batch_size"])
    run_dir = result_root / "runs" / variant; run_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_hash = _run_hash(protocol_hash, variant, seed)
    checkpoint = checkpoint_root / f"{variant}-{run_hash}.pt"; status_path = run_dir / "status.json"; heartbeat = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("protocol_sha256") == protocol_hash and existing.get("epoch") == epochs:
            return {**existing, "action": "SKIP"}
    set_seed(seed)
    norm = load_feature_norm(feature_norm_path, variant)
    model = build_k2_model(variant, norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["optimizer"]["learning_rate"]), weight_decay=float(protocol["optimizer"]["weight_decay"]))
    start_epoch = 0; history: list[dict[str, Any]] = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get("protocol_sha256"), saved.get("variant"), saved.get("seed")) != (protocol_hash, variant, seed):
            raise ValueError("K2 checkpoint provenance mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start_epoch = int(saved["epoch"])
        if status_path.exists(): history = list(json.loads(status_path.read_text()).get("history", []))
    dataset = K2Dataset("train", list(range(SPLIT_SIZES["train"])))
    for epoch in range(start_epoch, epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), num_workers=2, pin_memory=True, persistent_workers=False)
        model.train(); totals = []; segmentations = []; orientations = []; occupancies = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(image, return_aux=True)
            segmentation = common_segmentation_loss(output["visible_logits"], target)
            if variant == "M0_backbone":
                orientation = segmentation * 0.0; occupancy = segmentation * 0.0; loss = segmentation
            else:
                aux = output["symbolic_aux"]
                orientation = orientation_loss(aux["orientation_logits"], batch["bank"].to(device, non_blocking=True), batch["valid"].to(device, non_blocking=True))
                occupancy = balanced_occupancy_loss(aux["occupancy_logits"], target)
                loss = segmentation + 0.10 * orientation + 0.05 * occupancy
            if not torch.isfinite(loss): raise ValueError(f"non-finite K2 loss for {variant}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError(f"non-finite K2 gradients for {variant}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            totals.append(float(loss.detach())); segmentations.append(float(segmentation.detach())); orientations.append(float(orientation.detach())); occupancies.append(float(occupancy.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(totals)), "segmentation_loss": float(np.mean(segmentations)), "orientation_loss": float(np.mean(orientations)), "occupancy_loss": float(np.mean(occupancies))}
        history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "variant": variant, "seed": seed, "protocol_sha256": protocol_hash, "run_hash": run_hash}, checkpoint)
        progress = {"status": "IN_PROGRESS", "variant": variant, "seed": seed, "epoch": epoch + 1, "epoch_budget": epochs, "protocol_sha256": protocol_hash, "run_hash": run_hash, "history": history, "checkpoint": str(checkpoint), "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=ANZA-KS-K2 variant={variant} seed=41 epoch={epoch + 1}/15 loss={row['loss']:.5f} status=RUNNING", flush=True)
    reloaded = build_k2_model(variant, norm).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False); reloaded.load_state_dict(saved["model"]); reloaded.eval()
    result = {"status": "COMPLETE", "action": "RUN", "variant": variant, "seed": seed, "epoch": epochs, "epoch_budget": epochs, "protocol_sha256": protocol_hash, "run_hash": run_hash, "history": history, "parameter_count": sum(p.numel() for p in model.parameters()), "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "checkpoint_reload": "PASS", "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def one_batch_smoke(protocol: dict[str, Any], feature_norm_path: Path, *, device: str = "cuda") -> dict[str, Any]:
    batch = next(iter(DataLoader(K2Dataset("train", list(range(8))), batch_size=8)))
    result = {}
    for variant in VARIANTS:
        set_seed(41); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_k2_model(variant, load_feature_norm(feature_norm_path, variant)).to(device).train()
        output = model(batch["image"].to(device), return_aux=True); target = batch["target"].to(device)
        loss = common_segmentation_loss(output["visible_logits"], target)
        if variant != "M0_backbone":
            aux = output["symbolic_aux"]
            loss = loss + 0.10 * orientation_loss(aux["orientation_logits"], batch["bank"].to(device), batch["valid"].to(device)) + 0.05 * balanced_occupancy_loss(aux["occupancy_logits"], target)
        loss.backward()
        result[variant] = {"loss": float(loss.detach()), "finite_gradients": bool(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())), "parameter_count": sum(p.numel() for p in model.parameters()), "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20)}
    return result


def predict_split(
    variant: str,
    checkpoint: Path,
    split: str,
    indices: list[int],
    feature_norm_path: Path,
    *,
    device: str = "cuda",
    perturb: bool = False,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    norm = load_feature_norm(feature_norm_path, variant)
    model = build_k2_model(variant, norm).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["model"]); model.eval()
    probabilities: list[np.ndarray] = []; samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            local = [generate_sample(split, index) for index in indices[start : start + 16]]
            image_array = np.stack([sample["image"] for sample in local])
            if perturb:
                perturbed = np.roll(image_array, shift=(1, -1), axis=(-2, -1)) * 0.94
                rng = np.random.default_rng(2_019_452 + start)
                image_array = perturbed + rng.normal(0.0, 0.025, size=perturbed.shape).astype(np.float32)
            logits = model(torch.from_numpy(image_array).to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy()[:, 0].astype(np.float32))
            samples.extend(local)
    return probabilities, samples
