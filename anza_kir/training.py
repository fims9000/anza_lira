"""IR1 base and frozen-backbone IR2 training for ANZA-KIR."""

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
from anza_ks_k2.dense_features import METHODS, dense_orientation_features
from anza_ks_k2.training import orientation_loss

from .benchmark import BASE_PRETRAIN_SIZE, generate_sample
from .model import KIR_VARIANTS, build_base_model, build_kir_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


class StreamDataset(Dataset):
    def __init__(self, records: list[tuple[str, int]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int | str]:
        stream, index = self.records[item]
        sample = generate_sample(stream, index)
        return {
            "image": torch.from_numpy(sample["image"]),
            "target": torch.from_numpy(sample["target"])[None],
            "distractor": torch.from_numpy(sample["distractor"])[None],
            "bank": torch.from_numpy(sample["orientation_bank"]),
            "valid": torch.from_numpy(sample["orientation_valid"])[None],
            "stream": stream,
            "index": index,
        }


def balanced_evidence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    local = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    hard_positive = local > 0.10; hard_negative = local <= 0.10
    element = F.binary_cross_entropy_with_logits(logits, local, reduction="none")
    zero = element.sum() * 0.0
    bce = 0.5 * (element[hard_positive].mean() if hard_positive.any() else zero) + 0.5 * (element[hard_negative].mean() if hard_negative.any() else zero)
    probability = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (probability * local).sum() + 1e-6) / (probability.sum() + local.sum() + 1e-6)
    return bce + 0.5 * dice


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_ir1_base(protocol: dict[str, Any], protocol_sha256: str, checkpoint_root: Path, result_root: Path, *, device: str) -> dict[str, Any]:
    settings = protocol["base_training"]; seed = int(protocol["seed"]); epochs = int(settings["epochs"])
    checkpoint_root.mkdir(parents=True, exist_ok=True); result_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f"IR1-base-{protocol_sha256[:16]}.pt"; status_path = result_root / "status.json"; heartbeat = result_root / "heartbeat.jsonl"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("protocol_sha256") == protocol_sha256 and status.get("epoch") == epochs:
            return {**status, "action": "SKIP"}
    set_seed(seed); model = build_base_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    start = 0; history: list[dict[str, float | int]] = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("protocol_sha256") != protocol_sha256: raise ValueError("IR1 base checkpoint provenance mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = int(saved["epoch"])
        if status_path.exists(): history = list(json.loads(status_path.read_text()).get("history", []))
    dataset = StreamDataset([("base-pretrain", index) for index in range(BASE_PRETRAIN_SIZE)])
    for epoch in range(start, epochs):
        loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), num_workers=2, pin_memory=True)
        model.train(); rows = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); output = model(image, return_aux=True)
            segmentation = common_segmentation_loss(output["visible_logits"], target)
            evidence = balanced_evidence_loss(output["evidence_logits"], target)
            orientation = orientation_loss(output["orientation_logits"], batch["bank"].to(device, non_blocking=True), batch["valid"].to(device, non_blocking=True))
            loss = segmentation + 0.10 * evidence + 0.10 * orientation
            if not torch.isfinite(loss): raise ValueError("non-finite IR1 base loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            rows.append((float(loss.detach()), float(segmentation.detach()), float(evidence.detach()), float(orientation.detach())))
        row = {"epoch": epoch + 1, "loss": float(np.mean([x[0] for x in rows])), "segmentation_loss": float(np.mean([x[1] for x in rows])), "evidence_loss": float(np.mean([x[2] for x in rows])), "orientation_loss": float(np.mean([x[3] for x in rows]))}
        history.append(row); torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "protocol_sha256": protocol_sha256}, checkpoint)
        progress = {"status": "IN_PROGRESS", "epoch": epoch + 1, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "checkpoint": str(checkpoint), "history": history, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=ANZA-KIR-IR1 epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} status=RUNNING", flush=True)
    receipt = {"status": "COMPLETE", "action": "RUN", "epoch": epochs, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "history": history, "parameter_count": sum(p.numel() for p in model.parameters()), "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n"); return receipt


def load_base_state(checkpoint: Path, device: str) -> dict[str, torch.Tensor]:
    return torch.load(checkpoint, map_location=device, weights_only=False)["model"]


def compute_feature_norm(base_checkpoint: Path, records: list[tuple[str, int]], *, device: str) -> dict[str, Any]:
    model = build_base_model().to(device); model.load_state_dict(load_base_state(base_checkpoint, device)); model.eval()
    totals = {method: torch.zeros(104, dtype=torch.float64, device=device) for method in METHODS}
    squares = {method: torch.zeros(104, dtype=torch.float64, device=device) for method in METHODS}; counts = {method: 0 for method in METHODS}
    loader = DataLoader(StreamDataset(records), batch_size=8, num_workers=2, pin_memory=True)
    with torch.inference_mode():
        for batch in loader:
            evidence = model(batch["image"].to(device), return_aux=True)["evidence_probability"]
            for method in METHODS:
                values = dense_orientation_features(evidence, method).reshape(-1, 104).to(torch.float64)
                totals[method] += values.sum(0); squares[method] += values.square().sum(0); counts[method] += len(values)
    result: dict[str, Any] = {"source": "frozen IR1 evidence probability", "records": [[s, i] for s, i in records], "methods": {}}
    for method in METHODS:
        mean = totals[method] / counts[method]; variance = (squares[method] / counts[method] - mean.square()).clamp_min(0); std = variance.sqrt(); std = torch.where(std < 1e-6, torch.ones_like(std), std)
        result["methods"][method] = {"mean": mean.cpu().tolist(), "std": std.cpu().tolist(), "count": counts[method]}
    return result


def train_ir2_variant(variant: str, protocol: dict[str, Any], protocol_sha256: str, base_checkpoint: Path, feature_norm: dict[str, Any], hard_indices: list[int], checkpoint_root: Path, result_root: Path, *, device: str) -> dict[str, Any]:
    if variant not in KIR_VARIANTS: raise ValueError(variant)
    settings = protocol["residual_training"]; seed = int(protocol["seed"]); epochs = int(settings["epochs"])
    checkpoint_root.mkdir(parents=True, exist_ok=True); run_dir = result_root / variant; run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f"{variant}-{protocol_sha256[:16]}.pt"; status_path = run_dir / "status.json"; heartbeat = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("protocol_sha256") == protocol_sha256 and status.get("epoch") == epochs: return {**status, "action": "SKIP"}
    set_seed(seed); base_state = load_base_state(base_checkpoint, device); model = build_kir_model(variant, base_state, feature_norm["methods"]).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    start = 0; history: list[dict[str, float | int]] = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get("protocol_sha256"), saved.get("variant")) != (protocol_sha256, variant): raise ValueError("IR2 checkpoint provenance mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = int(saved["epoch"])
        if status_path.exists(): history = list(json.loads(status_path.read_text()).get("history", []))
    natural_count = int(settings["natural_samples"]); hard_count = int(settings["hard_samples"])
    if len(hard_indices) < hard_count: raise ValueError("insufficient frozen hard-train indices")
    records = [("residual-train-natural", index) for index in range(natural_count)] + [("mine-train", index) for index in hard_indices[:hard_count]]
    dataset = StreamDataset(records)
    for epoch in range(start, epochs):
        loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), num_workers=2, pin_memory=True)
        model.train(); losses = []; corrections = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); output = model(image, return_aux=True)
            segmentation = common_segmentation_loss(output["visible_logits"], target)
            correction = output["residual_aux"]["gated_correction"].square().mean()
            loss = segmentation + float(settings["correction_l2"]) * correction
            if not torch.isfinite(loss): raise ValueError(f"non-finite IR2 loss: {variant}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable): raise ValueError(f"non-finite IR2 gradient: {variant}")
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step(); losses.append(float(loss.detach())); corrections.append(float(correction.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "correction_l2": float(np.mean(corrections)), "gamma": float(model.residual.gamma.detach())}; history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "variant": variant, "protocol_sha256": protocol_sha256}, checkpoint)
        progress = {"status": "IN_PROGRESS", "variant": variant, "epoch": epoch + 1, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "checkpoint": str(checkpoint), "history": history, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=ANZA-KIR-IR2 variant={variant} seed={seed} epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} gamma={row['gamma']:.5f} status=RUNNING", flush=True)
    receipt = {"status": "COMPLETE", "action": "RUN", "variant": variant, "epoch": epochs, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "history": history, "total_parameter_count": sum(p.numel() for p in model.parameters()), "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "base_checkpoint_sha256": _sha(base_checkpoint), "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n"); return receipt


def predict_records(model: torch.nn.Module, records: list[tuple[str, int]], *, device: str, batch_size: int = 16) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    model.eval(); probabilities: list[np.ndarray] = []; samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            local_records = records[start : start + batch_size]; local = [generate_sample(stream, index) for stream, index in local_records]
            images = torch.from_numpy(np.stack([sample["image"] for sample in local])).to(device)
            probabilities.extend(torch.sigmoid(model(images)).cpu().numpy()[:, 0].astype(np.float32)); samples.extend(local)
    return probabilities, samples
