"""Fixed-budget, source-balanced training of the exact historical P0."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..protocol import PROTOCOL, protocol_hash
from .dataset import SourceCorridorDataset, collate_sources
from .legacy_loader import build_exact_p0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_balanced_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    source_bce = (element * mask).sum(1) / mask.sum(1).clamp_min(1)
    loss = source_bce.mean()
    ranking_terms = []
    for row in range(len(logits)):
        if not bool(positive[row]):
            continue
        negative = logits[row][mask[row] & (labels[row] < 0.5)]
        if len(negative):
            ranking_terms.append(F.softplus(-(logits[row, 0] - negative) / float(PROTOCOL["p0"]["ranking_temperature"])).mean())
    if ranking_terms:
        loss = loss + float(PROTOCOL["p0"]["ranking_weight"]) * torch.stack(ranking_terms).mean()
    return loss


def train_p0(cache_dir: Path, output_dir: Path, *, device: str) -> dict[str, Any]:
    settings = PROTOCOL["p0"]
    seed = int(settings["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    manifest_path = output_dir / "training_manifest.json"
    if checkpoint.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("epochs_completed") == int(settings["epochs"]) and existing.get("checkpoint_sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            return {**existing, "action": "SKIP_COMPLETE"}
    set_seed(seed)
    dataset = SourceCorridorDataset(cache_dir / "relation_train_sources.csv", cache_dir / "relation_train_corridors.npy")
    model = build_exact_p0().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("protocol_sha256") != protocol_hash() or payload.get("seed") != seed:
            raise ValueError("P0 resume checkpoint provenance drift")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"])
    for epoch in range(start_epoch, int(settings["epochs"])):
        loader = DataLoader(
            dataset,
            batch_size=int(settings["batch_sources"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            # Python 3.14 forkserver workers are fragile for resumable CLI/IDE
            # launches; the memmapped source batches are small enough to load
            # synchronously while the model still trains on the requested GPU.
            num_workers=0,
            pin_memory=device.startswith("cuda"),
            collate_fn=collate_sources,
        )
        model.train()
        losses = []
        for batch in loader:
            corridors = batch["corridors"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            logits = model(corridors.flatten(0, 1)).reshape(corridors.shape[:2])
            loss = source_balanced_loss(logits, labels, mask, positive)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite P0 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "epoch": epoch + 1, "history": history, "protocol_sha256": protocol_hash()}, checkpoint)
        progress = {"status": "IN_PROGRESS", "seed": seed, "epochs_completed": epoch + 1, "epoch_budget": int(settings["epochs"]), "checkpoint": str(checkpoint), "path_opened": False, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False, "transformer_built": False}
        manifest_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        print(f"phase=ENDGAME-E1 seed={seed} epoch={epoch + 1}/{settings['epochs']} loss={row['loss']:.6f}", flush=True)
    with (output_dir / "training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)
    receipt = {
        "seed": seed,
        "epochs_completed": int(settings["epochs"]),
        "early_stopping": False,
        "training_sources": len(dataset),
        "final_loss": history[-1]["loss"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "path_opened": False,
        "confirm_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
        "transformer_built": False,
    }
    manifest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_trained_p0(checkpoint: Path, *, device: str) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("protocol_sha256") != protocol_hash() or payload.get("epoch") != int(PROTOCOL["p0"]["epochs"]):
        raise ValueError("P0 checkpoint provenance drift")
    model = build_exact_p0().to(device)
    model.load_state_dict(payload["model"])
    return model.eval()
