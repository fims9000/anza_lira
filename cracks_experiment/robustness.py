"""Setting C: image-disjoint crowd training and expert-fold robustness."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from cracks_experiment.evaluation import evaluate_binary_section
from cracks_experiment.finetuning import FOLDS, verify_setting_a_complete
from cracks_experiment.matrix import CRACKSRunSpec, PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import NORMALIZATION, build_real_model
from cracks_experiment.validation import _binary_metrics, select_threshold, tiled_probability
from datasets.cracks import CRACKSAnnotatedSectionDataset, CRACKSSectionDataset
from synthetic.crossing_trace_bench import generate_sample
from synthetic.experiment_matrix import development_matrix
from synthetic.training import candidate_loss
import utils


CRACKS_PROTOCOL = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
SETTING_C_PROTOCOL = {
    "setting": "C_image_disjoint_robustness",
    "models": ["unet", "anza_v1", "anza_v2b"],
    "seed": 42,
    "fold_manifest_sha256": FOLDS["sha256"],
    "neighbor_guard": 2,
    "epochs": 20,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "effective_batch_size": 4,
    "crop_size": 256,
    "foreground_crop_probability": 0.7,
    "loss": "bce+dice+0.2*soft_cldice",
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "test_access": "after crowd-only training and non-expert threshold freeze",
}


def setting_c_models() -> tuple[CRACKSRunSpec, ...]:
    allowed = set(SETTING_C_PROTOCOL["models"])
    return tuple(
        spec
        for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == 42 and spec.model in allowed
    )


def setting_c_run_hash(spec: CRACKSRunSpec, fold: dict[str, Any]) -> str:
    payload = {"spec": asdict(spec), "fold": fold, "protocol": SETTING_C_PROTOCOL}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: CRACKSRunSpec,
    run_hash: str,
    epoch: int,
) -> None:
    torch.save(
        {
            "run_hash": run_hash,
            "spec": asdict(spec),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "expert_scores_used": False,
            "test_scores_used": False,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_hash: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("run_hash") != run_hash
        or payload.get("expert_scores_used") is not False
        or payload.get("test_scores_used") is not False
    ):
        raise ValueError("Setting C checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _crowd_dataset(section_ids: list[int], *, crop: bool, seed: int) -> CRACKSSectionDataset:
    return CRACKSSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / "paper_like" / ("train" if crop else "heldout"),
        section_ids,
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
        crop_size=256 if crop else None,
        foreground_probability=0.7,
        seed=seed,
    )


def run_setting_c_fold(
    spec: CRACKSRunSpec,
    fold: dict[str, Any],
    setting_a_root: Path,
    setting_a_expert_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    epochs: int | None = None,
    max_train_sections: int | None = None,
) -> dict[str, Any]:
    # This does not supply weights; it proves Setting A was closed before the
    # secondary protocol started and its expert masks became available.
    setting_a_receipt = verify_setting_a_complete(setting_a_root, setting_a_expert_root)
    excluded = set(int(value) for value in fold["setting_c_excluded_section_ids"])
    test_ids = set(int(value) for value in fold["test"])
    if not test_ids.issubset(excluded):
        raise ValueError("Setting C guard does not contain every expert test section")
    train_ids = [
        int(value) for value in CRACKS_PROTOCOL["setting_a"]["training_section_ids"]
        if int(value) not in excluded
    ]
    validation_ids = [
        int(value) for value in CRACKS_PROTOCOL["setting_a"]["held_out_validation_section_ids"]
        if int(value) not in excluded
    ]
    if excluded & set(train_ids) or excluded & set(validation_ids):
        raise AssertionError("Setting C exclusion leaked into crowd train/validation")
    run_hash = setting_c_run_hash(spec, fold)
    run_id = f"{spec.model}_fold{fold['fold']}"
    run_dir = Path(output_root) / f"{run_id}-{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("run_hash") == run_hash:
            return {**existing, "action": "SKIP"}

    budget = int(epochs or SETTING_C_PROTOCOL["epochs"])
    seed = int(spec.seed)
    _set_seed(seed)
    torch_device = torch.device(device)
    model = build_real_model(spec)
    if spec.model.startswith("anza_v2"):
        frozen = json.loads(
            (PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "frozen_v2.json").read_text()
        )
        payload = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTING_C_PROTOCOL["learning_rate"])
    train = _crowd_dataset(train_ids, crop=True, seed=seed)
    checkpoint_path = run_dir / "checkpoint-last.pt"
    history = []
    start_epoch = 0
    if checkpoint_path.exists():
        payload = _load_checkpoint(checkpoint_path, model, optimizer, run_hash)
        start_epoch = int(payload["epoch"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))
    replay_spec = next(run for run in development_matrix() if run.candidate_id == "C3")
    for epoch in range(start_epoch, budget):
        model.train()
        train.set_epoch(epoch)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        if max_train_sections is not None:
            order = order[: int(max_train_sections)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accumulation = int(SETTING_C_PROTOCOL["effective_batch_size"])
        for position, dataset_index in enumerate(order):
            batch = train[dataset_index]
            logits = model(batch["image"].unsqueeze(0).to(torch_device))
            real_loss, _logs, _ = utils.segmentation_objective(
                logits,
                batch["target"].unsqueeze(0).to(torch_device),
                batch["valid"].unsqueeze(0).float().to(torch_device),
                topology_weight=0.2,
                topology_num_iters=5,
            )
            total = real_loss
            if spec.structural_replay and (position + 1) % 3 == 0:
                sample = generate_sample("train", epoch * len(order) + position, image_size=128)
                replay_loss, _ = candidate_loss(model, replay_spec, sample, torch_device)
                total = total + 0.25 * replay_loss
            (total / accumulation).backward()
            losses.append(float(total.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        _save_checkpoint(checkpoint_path, model, optimizer, spec, run_hash, epoch + 1)
        status_path.write_text(
            json.dumps(
                {
                    "status": "IN_PROGRESS",
                    "run_id": run_id,
                    "run_hash": run_hash,
                    "epoch": epoch + 1,
                    "excluded_section_ids": sorted(excluded),
                    "train_section_count": len(train_ids),
                    "expert_scores_used": False,
                    "test_scores_used": False,
                    "history": history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"phase=cracks_setting_c model={spec.model} fold={fold['fold']} epoch={epoch + 1}/{budget} "
            "expert_test=LOCKED status=RUNNING"
        )

    model.eval()
    validation = _crowd_dataset(validation_ids, crop=False, seed=seed)
    thresholds = tuple(float(value) for value in SETTING_C_PROTOCOL["threshold_candidates"])
    validation_rows: list[dict[str, Any]] = []
    for index in range(len(validation)):
        batch = validation[index]
        probability = tiled_probability(model, batch["image"]).numpy()[:255, :701]
        target = batch["target"][0, :255, :701].numpy() >= 0.5
        valid = batch["valid"][0, :255, :701].numpy().astype(bool)
        for threshold in thresholds:
            validation_rows.append(
                {
                    "section_id": int(batch["section_id"]),
                    "threshold": threshold,
                    **_binary_metrics(probability >= threshold, target, valid),
                }
            )
    selection = select_threshold(validation_rows)
    selection_core = {
        "run_id": run_id,
        "run_hash": run_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "selected_threshold": selection["selected_threshold"],
        "validation_section_ids": validation_ids,
        "excluded_section_ids": sorted(excluded),
        "expert_scores_used": False,
        "test_scores_used": False,
    }
    selection_payload = {
        **selection_core,
        "sha256": hashlib.sha256(
            json.dumps(selection_core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (run_dir / "selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
    )

    expert_test = CRACKSAnnotatedSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "annotations" / "expert",
        list(fold["test"]),
        policy_name="paper_like",
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
        seed=seed,
    )
    test_rows = []
    for index in range(len(expert_test)):
        batch = expert_test[index]
        probability = tiled_probability(model, batch["image"]).numpy()[:255, :701]
        target = batch["target"][0, :255, :701].numpy() >= 0.5
        valid = batch["valid"][0, :255, :701].numpy().astype(bool)
        test_rows.append(
            {
                "model": spec.model,
                "fold": int(fold["fold"]),
                "section_id": int(batch["section_id"]),
                "threshold": selection_payload["selected_threshold"],
                **evaluate_binary_section(
                    probability,
                    target,
                    valid,
                    selection_payload["selected_threshold"],
                    orientation_sensitivity_radii=(3, 7),
                ),
            }
        )
    rows_path = run_dir / "test_sections.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(test_rows)
    keys = ("dice", "iou", "cldice", "skeleton_f1_at_2px", "fragmentation", "trace_orientation_error_median_deg")
    summary = {key: float(np.mean([float(row[key]) for row in test_rows])) for key in keys}
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_id": run_id,
        "run_hash": run_hash,
        "model": spec.model,
        "fold": int(fold["fold"]),
        "setting_a_receipt_sha256": setting_a_receipt["sha256"],
        "selection_sha256": selection_payload["sha256"],
        "excluded_section_ids": sorted(excluded),
        "train_section_count": len(train_ids),
        "validation_section_count": len(validation_ids),
        "test_section_count": len(test_rows),
        "test_scores_used_after_selection": True,
        "summary": summary,
        "history": history,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
