"""Frozen, resumable CRACKS T1 partial-label training matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from cracks_experiment.matrix import PROJECT_ROOT, SETTING_A_PROTOCOL
from cracks_experiment.partial_labels import CRACKSMultiAnnotatorDataset, average_annotator_loss
from cracks_experiment.training import NORMALIZATION, build_real_model


_SOURCE_FILES = (
    PROJECT_ROOT / "cracks_experiment" / "partial_labels.py",
    PROJECT_ROOT / "cracks_experiment" / "partial_label_training.py",
    PROJECT_ROOT / "models" / "segmentation_v2.py",
    PROJECT_ROOT / "models" / "azconv.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CRACKS_PROTOCOL = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
T1_PROTOCOL: dict[str, Any] = {
    "version": "cracks_partial_label_t1_v1",
    "source_setting_a_protocol_sha256": CRACKS_PROTOCOL["sha256"],
    "target_semantics": {
        "blue": [1.0, 1.0],
        "green": [1.0, 0.5],
        "orange": [0.0, 1.0],
        "white": [0.0, 0.0],
    },
    "loss": "mean_over_annotators(bce+dice+0.2*soft_cldice); no mask fusion",
    "annotators_per_training_section": 4,
    "training_annotators": CRACKS_PROTOCOL["setting_a"]["training_annotators"],
    "heldout_annotators": CRACKS_PROTOCOL["setting_a"]["held_out_annotators"]["all"],
    "training_section_ids": CRACKS_PROTOCOL["setting_a"]["training_section_ids"],
    "heldout_section_ids": CRACKS_PROTOCOL["setting_a"]["held_out_validation_section_ids"],
    "epochs": SETTING_A_PROTOCOL["epochs"],
    "optimizer": SETTING_A_PROTOCOL["optimizer"],
    "learning_rate": SETTING_A_PROTOCOL["learning_rate"],
    "effective_batch_size": SETTING_A_PROTOCOL["effective_batch_size"],
    "microbatch_size": 1,
    "crop_size": SETTING_A_PROTOCOL["crop_size"],
    "foreground_crop_probability": SETTING_A_PROTOCOL["foreground_crop_probability"],
    "topology_iterations": SETTING_A_PROTOCOL["topology_iterations"],
    "threshold_candidates": SETTING_A_PROTOCOL["threshold_candidates"],
    "models": ["unet", "anza_v1"],
    "seeds": [41, 42, 43],
    "expert": "LOCKED_NOT_ACCESSED",
    "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in _SOURCE_FILES},
}


def t1_protocol_hash() -> str:
    encoded = json.dumps(T1_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class T1RunSpec:
    run_id: str
    model: str
    seed: int

    @property
    def run_hash(self) -> str:
        payload = {"spec": asdict(self), "protocol_sha256": t1_protocol_hash()}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]


def t1_matrix() -> tuple[T1RunSpec, ...]:
    return tuple(
        T1RunSpec(f"t1_{model}_s{seed}", model, seed)
        for model in ("unet", "anza_v1")
        for seed in (41, 42, 43)
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(spec: T1RunSpec) -> torch.nn.Module:
    # build_real_model uses only model/seed/run_hash attributes for these baselines.
    return build_real_model(spec)  # type: ignore[arg-type]


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: T1RunSpec,
    epoch: int,
) -> dict[str, Any]:
    return {
        "status": "IN_PROGRESS",
        "run_id": spec.run_id,
        "run_hash": spec.run_hash,
        "protocol_sha256": t1_protocol_hash(),
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "expert_scores_used": False,
        "expert_data_accessed": False,
    }


def load_t1_checkpoint(
    path: Path,
    spec: T1RunSpec,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "run_hash": spec.run_hash,
        "protocol_sha256": t1_protocol_hash(),
        "expert_scores_used": False,
        "expert_data_accessed": False,
    }
    drift = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    if drift:
        raise ValueError(f"T1 checkpoint provenance mismatch: {drift}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _datasets(spec: T1RunSpec) -> tuple[CRACKSMultiAnnotatorDataset, CRACKSMultiAnnotatorDataset]:
    common = {
        "image_root": PROJECT_ROOT / "data" / "cracks" / "images",
        "annotation_root": PROJECT_ROOT / "data" / "cracks" / "annotations",
        "mean": NORMALIZATION["mean"],
        "std": NORMALIZATION["std"],
        "crop_size": 256,
        "foreground_probability": float(T1_PROTOCOL["foreground_crop_probability"]),
        "seed": spec.seed,
    }
    train = CRACKSMultiAnnotatorDataset(
        section_ids=T1_PROTOCOL["training_section_ids"],
        annotators=T1_PROTOCOL["training_annotators"],
        annotators_per_section=int(T1_PROTOCOL["annotators_per_training_section"]),
        **common,
    )
    heldout = CRACKSMultiAnnotatorDataset(
        section_ids=T1_PROTOCOL["heldout_section_ids"],
        annotators=T1_PROTOCOL["heldout_annotators"],
        annotators_per_section=None,
        **common,
    )
    return train, heldout


def run_t1_training(
    spec: T1RunSpec,
    output_root: Path,
    *,
    device: str = "cuda",
    epochs: int | None = None,
    max_train_sections: int | None = None,
) -> dict[str, Any]:
    budget = int(T1_PROTOCOL["epochs"] if epochs is None else epochs)
    run_dir = Path(output_root) / f"{spec.run_id}-{spec.run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("run_hash") != spec.run_hash or status.get("protocol_sha256") != t1_protocol_hash():
            raise ValueError(f"T1 run directory hash drift: {run_dir}")
        if status.get("training_section_limit") != max_train_sections:
            raise ValueError("T1 training section limit changed inside an existing run")
        if status.get("status") == "COMPLETE" and int(status.get("epoch", 0)) >= budget:
            return {**status, "action": "SKIP"}

    _seed_everything(spec.seed)
    torch_device = torch.device(device)
    model = _model(spec).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(T1_PROTOCOL["learning_rate"]))
    train, heldout = _datasets(spec)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        payload = load_t1_checkpoint(checkpoint_path, spec, model, optimizer)
        start_epoch = int(payload["epoch"])
        history = list(json.loads(status_path.read_text()).get("history", []))
    accumulation = int(T1_PROTOCOL["effective_batch_size"])
    for epoch in range(start_epoch, budget):
        model.train()
        train.set_epoch(epoch)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(spec.seed + epoch)).tolist()
        if max_train_sections is not None:
            order = order[: int(max_train_sections)]
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for position, dataset_index in enumerate(order):
            batch = train[dataset_index]
            image = batch["image"].unsqueeze(0).to(torch_device)
            targets = batch["targets"].to(torch_device)
            weights = batch["weights"].to(torch_device)
            loss, _ = average_annotator_loss(
                model(image),
                targets,
                weights,
                topology_weight=0.2,
                topology_num_iters=int(T1_PROTOCOL["topology_iterations"]),
            )
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError(f"Non-finite T1 gradient in {spec.run_id}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        heldout.set_epoch(0)
        monitoring: list[float] = []
        with torch.no_grad():
            for index in range(min(16, len(heldout))):
                batch = heldout[index]
                probability = torch.sigmoid(model(batch["image"].unsqueeze(0).to(torch_device))).cpu()[0, 0]
                annotation_scores = []
                for target, weight in zip(batch["targets"][:, 0], batch["weights"][:, 0]):
                    valid = weight > 0
                    pred = probability >= 0.5
                    truth = target >= 0.5
                    denominator = int(pred[valid].sum() + truth[valid].sum())
                    annotation_scores.append(
                        2.0 * int((pred[valid] & truth[valid]).sum()) / denominator if denominator else 1.0
                    )
                monitoring.append(float(np.mean(annotation_scores)))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "heldout_explicit_crop_dice_at_0_5": float(np.mean(monitoring)),
        }
        history.append(row)
        torch.save(_checkpoint(model, optimizer, spec, epoch + 1), checkpoint_path)
        progress = {
            "status": "IN_PROGRESS",
            "run_id": spec.run_id,
            "run_hash": spec.run_hash,
            "protocol_sha256": t1_protocol_hash(),
            "model": spec.model,
            "seed": spec.seed,
            "epoch": epoch + 1,
            "epoch_budget": budget,
            "training_section_limit": max_train_sections,
            "expert_scores_used": False,
            "expert_data_accessed": False,
            "history": history,
        }
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        print(
            f"phase=cracks_t1 model={spec.run_id} epoch={epoch + 1}/{budget} "
            f"loss={row['train_loss']:.5f} val={row['heldout_explicit_crop_dice_at_0_5']:.5f} "
            "expert=LOCKED status=RUNNING",
            flush=True,
        )
    final_checkpoint = _checkpoint(model, optimizer, spec, budget)
    final_checkpoint["status"] = "COMPLETE"
    torch.save(final_checkpoint, checkpoint_path)
    reloaded = _model(spec)
    load_t1_checkpoint(checkpoint_path, spec, reloaded)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_id": spec.run_id,
        "run_hash": spec.run_hash,
        "protocol_sha256": t1_protocol_hash(),
        "model": spec.model,
        "seed": spec.seed,
        "epoch": budget,
        "epoch_budget": budget,
        "training_section_limit": max_train_sections,
        "checkpoint_reload": "PASS",
        "expert_scores_used": False,
        "expert_data_accessed": False,
        "history": history,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
