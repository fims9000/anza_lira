"""Resumable seed-41 TG2 relation training."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .batch import K_MAX, RelationDataset
from .models import VARIANTS, build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def p0_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(logits); positive = labels < K_MAX
    if positive.any(): target[positive, labels[positive]] = 1.0
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none"); positive_mask = (target > 0.5) & mask; negative_mask = (target <= 0.5) & mask; zero = element.sum() * 0.0
    return 0.5 * (element[positive_mask].mean() if positive_mask.any() else zero) + 0.5 * (element[negative_mask].mean() if negative_mask.any() else zero)


def train_variant(variant: str, *, protocol: dict[str, Any], protocol_sha256: str, result_root: Path, checkpoint_root: Path, device: str = "cuda") -> dict[str, Any]:
    settings = protocol["training"]; seed = int(protocol["seed"]); epochs = int(settings["epochs"]); run_dir = result_root / "training" / variant; run_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_hash = hashlib.sha256(f"{protocol_sha256}:{variant}:seed{seed}".encode()).hexdigest()[:16]; checkpoint = checkpoint_root / f"{variant}-{run_hash}.pt"; status_path = run_dir / "status.json"; heartbeat = run_dir / "heartbeat.jsonl"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "COMPLETE" and status.get("protocol_sha256") == protocol_sha256 and status.get("epoch") == epochs: return {**status, "action": "SKIP"}
    set_seed(seed); model = build_model(variant).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])); start = 0; history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get("variant"), saved.get("protocol_sha256"), saved.get("seed")) != (variant, protocol_sha256, seed): raise ValueError("TraceGraph checkpoint provenance mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = int(saved["epoch"])
        if status_path.exists(): history = list(json.loads(status_path.read_text()).get("history", []))
    dataset = RelationDataset("train", list(range(int(protocol["splits"]["train"]))))
    for epoch in range(start, epochs):
        loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), num_workers=2, pin_memory=True)
        model.train(); losses = []; betas = []; bias_values = []
        for raw in loader:
            batch = to_device(raw, device); optimizer.zero_grad(set_to_none=True); output = model(batch, return_aux=True)
            loss = p0_loss(output["pair_logits"], batch["label"], batch["candidate_mask"]) if variant == "P0_pair" else F.cross_entropy(output["logits"], batch["label"])
            if not torch.isfinite(loss): raise ValueError(f"non-finite TraceGraph loss: {variant}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()): raise ValueError(f"non-finite TraceGraph gradient: {variant}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
            if variant != "P0_pair": betas.append(float(output["beta"].detach())); bias_values.append(float(output["bias_mean_abs"].detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "beta": float(np.mean(betas)) if betas else 0.0, "bias_mean_abs": float(np.mean(bias_values)) if bias_values else 0.0}; history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "variant": variant, "protocol_sha256": protocol_sha256, "seed": seed}, checkpoint)
        progress = {"status": "IN_PROGRESS", "variant": variant, "epoch": epoch + 1, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "run_hash": run_hash, "checkpoint": str(checkpoint), "history": history, "confirm_opened": False, "tg3_opened": False, "cracks_accessed": False, "expert_accessed": False}
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        with heartbeat.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"phase=TRACEGRAPH-TG2 variant={variant} seed={seed} epoch={epoch + 1}/{epochs} loss={row['loss']:.5f} beta={row['beta']:.5f} status=RUNNING", flush=True)
    receipt = {"status": "COMPLETE", "action": "RUN", "variant": variant, "seed": seed, "epoch": epochs, "epoch_budget": epochs, "protocol_sha256": protocol_sha256, "run_hash": run_hash, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "parameter_count": sum(p.numel() for p in model.parameters()), "history": history, "confirm_opened": False, "tg3_opened": False, "cracks_accessed": False, "expert_accessed": False}
    status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n"); return receipt


def load_model(variant: str, checkpoint: Path, device: str) -> torch.nn.Module:
    model = build_model(variant).to(device); model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); model.eval(); return model


def predict(variant: str, checkpoint: Path, split: str, indices: list[int], *, device: str = "cuda") -> list[dict[str, Any]]:
    model = load_model(variant, checkpoint, device); dataset = RelationDataset(split, indices); loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True); rows = []
    with torch.inference_mode():
        for raw in loader:
            batch = to_device(raw, device); output = model(batch, return_aux=True); pair_logits = output["pair_logits"]
            scene_probabilities = None if variant == "P0_pair" else torch.softmax(output["logits"], dim=-1)
            for offset in range(len(batch["dense"])):
                count = int(batch["candidate_mask"][offset].sum()); rows.append({"index": int(batch["index"][offset]), "scene_type": raw["scene_type"][offset], "label": int(batch["label"][offset]), "candidate_count": count, "pair_logits": pair_logits[offset, :count].cpu().tolist(), "pair_probabilities": torch.sigmoid(pair_logits[offset, :count]).cpu().tolist(), "scene_probabilities": None if scene_probabilities is None else scene_probabilities[offset].cpu().tolist(), "beta": 0.0 if variant == "P0_pair" else float(output["beta"].detach()), "bias_mean_abs": 0.0 if variant != "P2_anza_tracegraph" else float(output["bias_mean_abs"].detach()), "bias_active_fraction": 0.0 if variant != "P2_anza_tracegraph" else float(output["bias_active_fraction"].detach())})
    return rows
