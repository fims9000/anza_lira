"""Resumable identical-budget H1 training and prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from synthetic.structural_losses import visible_segmentation_loss

from .model import build_h1_model
from .orientation_bank import orientation_bank_loss, orientation_bank_targets
from .stress_bench import generate_stress_sample


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class StressDataset(Dataset):
    def __init__(self, split: str, indices: list[int]) -> None:
        self.records = []
        for index in indices:
            sample = generate_stress_sample(split, index)
            bank, valid = orientation_bank_targets(sample)
            self.records.append({
                "image": torch.from_numpy(sample["image"]),
                "target": torch.from_numpy(np.asarray(sample["visible_fault_mask"], dtype=np.float32))[None],
                "bank": torch.from_numpy(bank), "valid": torch.from_numpy(valid)[None], "index": index,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return self.records[index]


def run_hash(protocol_hash: str, variant: str) -> str:
    return hashlib.sha256(f"{protocol_hash}:{variant}:seed41".encode()).hexdigest()[:16]


def _checkpoint(path: Path, model, optimizer, epoch: int, variant: str, protocol_hash: str) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "variant": variant, "protocol_hash": protocol_hash, "run_hash": run_hash(protocol_hash, variant)}, path)


def train_variant(
    variant: str, *, protocol: dict[str, Any], protocol_hash: str, output_root: Path,
    checkpoint_root: Path, device: str = "cuda",
) -> dict[str, Any]:
    settings = protocol["training"]; epochs = int(settings["epochs"]); seed = int(settings["seed"])
    run_dir = output_root / variant; run_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{variant}-{run_hash(protocol_hash, variant)}.pt"
    status_path = run_dir / "status.json"; heartbeat = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("protocol_sha256") == protocol_hash and existing.get("epoch") == epochs:
            return {**existing, "action": "SKIP"}
    set_seed(seed); torch_device = torch.device(device)
    model = build_h1_model(variant).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    start_epoch = 0; history = []
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        if saved.get("protocol_hash") != protocol_hash or saved.get("variant") != variant:
            raise ValueError("H1 checkpoint protocol/variant mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start_epoch = int(saved["epoch"])
        if status_path.exists(): history = list(json.loads(status_path.read_text()).get("history", []))
    dataset = StressDataset("train", list(range(int(settings["train_samples"]))))
    for epoch in range(start_epoch, epochs):
        loader = DataLoader(
            dataset, batch_size=int(settings["batch_size"]), shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch), num_workers=0, pin_memory=True,
        )
        model.train(); losses = []; segmentation_losses = []; orientation_losses = []
        for batch in loader:
            image = batch["image"].to(torch_device, non_blocking=True); target = batch["target"].to(torch_device, non_blocking=True)
            bank = batch["bank"].to(torch_device, non_blocking=True); valid = batch["valid"].to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(image, return_aux=True)
            segmentation = visible_segmentation_loss(output["visible_logits"], target)
            orientation = orientation_bank_loss(output["orientation_logits"], bank, valid, background_weight=protocol["orientation_target"]["background_weight"])
            loss = segmentation + float(settings["orientation_loss_weight"]) * orientation
            if not torch.isfinite(loss): raise ValueError(f"non-finite H1 loss: {variant}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError(f"non-finite H1 gradient: {variant}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            losses.append(float(loss.detach())); segmentation_losses.append(float(segmentation.detach())); orientation_losses.append(float(orientation.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "segmentation_loss": float(np.mean(segmentation_losses)), "orientation_loss": float(np.mean(orientation_losses))}
        history.append(row); _checkpoint(checkpoint_path, model, optimizer, epoch + 1, variant, protocol_hash)
        progress = {"status": "IN_PROGRESS", "variant": variant, "run_hash": run_hash(protocol_hash, variant), "protocol_sha256": protocol_hash,
                    "epoch": epoch + 1, "epoch_budget": epochs, "history": history, "checkpoint": str(checkpoint_path), "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=ANZA-HS-H1 variant={variant} seed=41 epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} status=RUNNING", flush=True)
    reloaded = build_h1_model(variant).to(torch_device)
    saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False); reloaded.load_state_dict(saved["model"]); reloaded.eval()
    result = {"status": "COMPLETE", "action": "RUN", "variant": variant, "run_hash": run_hash(protocol_hash, variant),
              "protocol_sha256": protocol_hash, "epoch": epochs, "epoch_budget": epochs, "history": history,
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "checkpoint": str(checkpoint_path),
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "checkpoint_reload": "PASS",
              "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def predict_variant(variant: str, checkpoint_path: Path, *, device: str = "cuda") -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    torch_device = torch.device(device); model = build_h1_model(variant).to(torch_device)
    saved = torch.load(checkpoint_path, map_location=torch_device, weights_only=False); model.load_state_dict(saved["model"]); model.eval()
    samples = [generate_stress_sample("dev", index) for index in range(264)]
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(samples), 16):
            image = torch.from_numpy(np.stack([sample["image"] for sample in samples[start:start + 16]])).to(torch_device)
            probabilities.extend(torch.sigmoid(model(image)).cpu().numpy()[:, 0].astype(np.float32))
    return probabilities, samples


def one_batch_smoke(protocol: dict[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    dataset = StressDataset("train", list(range(16))); batch = next(iter(DataLoader(dataset, batch_size=4)))
    result = {}
    for variant in protocol["matrix"]:
        set_seed(41); model = build_h1_model(variant).to(device); model.train()
        output = model(batch["image"].to(device), return_aux=True)
        segmentation = visible_segmentation_loss(output["visible_logits"], batch["target"].to(device))
        orientation = orientation_bank_loss(output["orientation_logits"], batch["bank"].to(device), batch["valid"].to(device))
        loss = segmentation + float(protocol["training"]["orientation_loss_weight"]) * orientation
        loss.backward()
        finite = torch.isfinite(loss) and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        result[variant] = {"loss": float(loss.detach()), "finite_gradients": bool(finite), "parameter_count": sum(p.numel() for p in model.parameters())}
    return result
