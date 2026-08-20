"""Resumable crowd-only training for CRACKS Setting A."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from cracks_experiment.matrix import CRACKSRunSpec, FROZEN_V2, PROJECT_ROOT, SETTING_A_PROTOCOL
from datasets.cracks import CRACKSSectionDataset
from models.segmentation_v2 import ComparableStructuralUNet, ComparableUNetConfig, build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.experiment_matrix import development_matrix
from synthetic.training import candidate_loss
import utils


NORMALIZATION = json.loads(
    (PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "crowd_target" / "normalization.json").read_text()
)
CRACKS_PROTOCOL = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_real_model(spec: CRACKSRunSpec) -> torch.nn.Module:
    if spec.model in {"unet", "deformable_unet", "anza_v1"}:
        return build_comparable_model(spec.model)
    if spec.model == "clean_anza":
        from models.segmentation_affinity import build_affinity_model

        seed_matched_v1 = build_comparable_model("anza_v1")
        return build_affinity_model("C1", seed_matched_v1=seed_matched_v1)
    operator = "v2b" if spec.directional_half_modes else "v2a"
    return ComparableStructuralUNet(
        cfg=ComparableUNetConfig(
            operator=operator,
            use_fuzzy=spec.use_fuzzy,
            structural_completion_head=False,
        )
    )


def _load_frozen_v2(model: torch.nn.Module) -> None:
    checkpoint = torch.load(FROZEN_V2["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: CRACKSRunSpec,
    epoch: int,
    status: str,
) -> dict[str, Any]:
    return {
        "run_hash": spec.run_hash,
        "spec": asdict(spec),
        "epoch": int(epoch),
        "status": status,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "expert_scores_used": False,
    }


def load_real_checkpoint(
    path: Path,
    expected_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_hash") != expected_hash:
        raise ValueError("CRACKS checkpoint config hash mismatch")
    if payload.get("expert_scores_used") is not False:
        raise ValueError("Setting A checkpoint provenance does not prove expert lock")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _crop_datasets(spec: CRACKSRunSpec, policy: str) -> tuple[CRACKSSectionDataset, CRACKSSectionDataset]:
    common = {
        "image_root": PROJECT_ROOT / "data" / "cracks" / "images",
        "mean": NORMALIZATION["mean"],
        "std": NORMALIZATION["std"],
        "crop_size": 256,
        "foreground_probability": 0.7,
        "seed": spec.seed,
    }
    train = CRACKSSectionDataset(
        target_root=PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / policy / "train",
        section_ids=CRACKS_PROTOCOL["setting_a"]["training_section_ids"],
        **common,
    )
    validation = CRACKSSectionDataset(
        target_root=PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / policy / "heldout",
        section_ids=CRACKS_PROTOCOL["setting_a"]["held_out_validation_section_ids"],
        **common,
    )
    return train, validation


def _crop_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    selected = valid.bool()
    denominator = int(prediction[selected].sum() + truth[selected].sum())
    return 2.0 * int((prediction[selected] & truth[selected]).sum()) / denominator if denominator else 1.0


def run_setting_a_training(
    spec: CRACKSRunSpec,
    output_root: Path,
    *,
    policy: str = "paper_like",
    epochs: int | None = None,
    max_train_sections: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    budget = int(SETTING_A_PROTOCOL["epochs"] if epochs is None else epochs)
    run_dir = output_root / f"{spec.run_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and int(existing.get("epoch", 0)) >= budget:
            return {**existing, "action": "SKIP"}

    _set_seed(spec.seed)
    torch_device = torch.device(device)
    model = build_real_model(spec)
    if spec.model.startswith("anza_v2"):
        _load_frozen_v2(model)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(SETTING_A_PROTOCOL["learning_rate"]))
    train, validation = _crop_datasets(spec, policy)
    start_epoch = 0
    history = []
    if checkpoint_path.exists():
        payload = load_real_checkpoint(checkpoint_path, spec.run_hash, model, optimizer)
        start_epoch = int(payload["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))
    accumulation = int(SETTING_A_PROTOCOL["effective_batch_size"])
    replay_spec = next(run for run in development_matrix() if run.candidate_id == "C3")
    for epoch in range(start_epoch, budget):
        model.train()
        train.set_epoch(epoch)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(spec.seed + epoch)).tolist()
        if max_train_sections is not None:
            order = order[: int(max_train_sections)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, dataset_index in enumerate(order):
            batch = train[dataset_index]
            image = batch["image"].unsqueeze(0).to(torch_device)
            target = batch["target"].unsqueeze(0).to(torch_device)
            valid = batch["valid"].unsqueeze(0).float().to(torch_device)
            logits = model(image)
            real_loss, _logs, _ = utils.segmentation_objective(
                logits,
                target,
                valid,
                topology_weight=0.2,
                topology_num_iters=int(SETTING_A_PROTOCOL["topology_iterations"]),
            )
            total = real_loss
            if spec.structural_replay and (position + 1) % 3 == 0:
                replay_sample = generate_sample(
                    "train", (epoch * len(order) + position) % 10_000, image_size=128
                )
                replay_loss, _ = candidate_loss(model, replay_spec, replay_sample, torch_device)
                total = total + 0.25 * replay_loss
            (total / accumulation).backward()
            losses.append(float(total.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError(f"Non-finite CRACKS gradient in {spec.run_id}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation.set_epoch(0)
        monitoring = []
        with torch.no_grad():
            for index in range(min(int(SETTING_A_PROTOCOL["monitoring_validation_crops"]), len(validation))):
                batch = validation[index]
                image = batch["image"].unsqueeze(0).to(torch_device)
                target = batch["target"].unsqueeze(0).to(torch_device)
                valid = batch["valid"].unsqueeze(0).to(torch_device)
                monitoring.append(_crop_dice(model(image), target, valid))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "heldout_crop_dice_at_0_5": float(np.mean(monitoring)),
        }
        history.append(row)
        torch.save(_checkpoint(model, optimizer, spec, epoch + 1, "IN_PROGRESS"), checkpoint_path)
        progress = {
            "status": "IN_PROGRESS",
            "run_hash": spec.run_hash,
            "run_id": spec.run_id,
            "model": spec.model,
            "seed": spec.seed,
            "policy": policy,
            "epoch": epoch + 1,
            "epoch_budget": budget,
            "expert_scores_used": False,
            "history": history,
        }
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        print(
            f"phase=cracks_setting_a model={spec.run_id} seed={spec.seed} epoch={epoch + 1}/{budget} "
            f"val_metric={row['heldout_crop_dice_at_0_5']:.4f} structural_metric=NA "
            "expert=LOCKED status=RUNNING"
        )
    torch.save(_checkpoint(model, optimizer, spec, budget, "COMPLETE"), checkpoint_path)
    reloaded = build_real_model(spec)
    load_real_checkpoint(checkpoint_path, spec.run_hash, reloaded)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_hash": spec.run_hash,
        "run_id": spec.run_id,
        "model": spec.model,
        "seed": spec.seed,
        "policy": policy,
        "epoch": budget,
        "epoch_budget": budget,
        "structural_replay": spec.structural_replay,
        "checkpoint_reload": "PASS",
        "expert_scores_used": False,
        "history": history,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
