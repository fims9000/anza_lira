"""Setting B: frozen-fold limited-expert fine-tuning and test evaluation."""

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

from cracks_experiment.evaluation import evaluate_binary_section, verify_threshold_freeze
from cracks_experiment.matrix import CRACKSRunSpec, PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import (
    _binary_metrics,
    _sha256,
    select_threshold,
    tiled_probability,
)
from datasets.cracks import CRACKSAnnotatedSectionDataset
import utils


FOLDS_PATH = PROJECT_ROOT / "data" / "cracks" / "splits" / "anza_v2_folds.json"
FOLDS = json.loads(FOLDS_PATH.read_text())
SETTING_B_PROTOCOL = {
    "setting": "B_limited_expert_fine_tuning",
    "source_seed": 42,
    "models": ["unet", "deformable_unet", "anza_v1", "anza_v2b"],
    "fold_manifest_sha256": FOLDS["sha256"],
    "fold_count": 5,
    "train_sections": 28,
    "validation_sections": 4,
    "test_sections": 8,
    "policy": "paper_like",
    "learning_rate": 1e-4,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "effective_batch_size": 4,
    "crop_size": 256,
    "foreground_crop_probability": 0.7,
    "loss": "bce+dice+0.2*soft_cldice",
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "test_access": "after best epoch and threshold frozen on expert validation",
}


def setting_b_sources() -> tuple[CRACKSRunSpec, ...]:
    return tuple(
        spec
        for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == SETTING_B_PROTOCOL["source_seed"]
    )


def setting_b_run_hash(spec: CRACKSRunSpec, fold: dict[str, Any], source_checkpoint_sha256: str) -> str:
    payload = {
        "source_spec": asdict(spec),
        "fold": fold,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "protocol": SETTING_B_PROTOCOL,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def verify_setting_a_complete(training_root: Path, expert_root: Path) -> dict[str, Any]:
    threshold_receipt = verify_threshold_freeze(training_root)
    path = Path(expert_root) / "complete.json"
    if not path.exists():
        raise PermissionError("Setting B locked: complete Setting A expert receipt missing")
    payload = json.loads(path.read_text())
    sha = payload.pop("sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if sha != hashlib.sha256(encoded).hexdigest() or payload.get("status") != "COMPLETE":
        raise PermissionError("Setting B locked: invalid Setting A expert receipt")
    if payload.get("run_count") != len(setting_a_matrix()) or payload.get("expert_section_count") != 40:
        raise PermissionError("Setting B locked: incomplete Setting A expert evaluation")
    if payload.get("threshold_freeze_sha256") != threshold_receipt["freeze_sha256"]:
        raise PermissionError("Setting B locked: threshold freeze receipt changed")
    for row in payload["runs"]:
        result = Path(expert_root) / f"{row['run_id']}-{row['run_hash']}.json"
        rows = Path(expert_root) / f"{row['run_id']}-{row['run_hash']}.csv"
        if _sha256(result) != row["result_sha256"] or _sha256(rows) != row["rows_sha256"]:
            raise PermissionError("Setting B locked: Setting A expert artifacts changed")
    return {**payload, "sha256": sha}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expert_dataset(section_ids: list[int], *, crop: bool, seed: int) -> CRACKSAnnotatedSectionDataset:
    return CRACKSAnnotatedSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "annotations" / "expert",
        section_ids,
        policy_name=SETTING_B_PROTOCOL["policy"],
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
        crop_size=256 if crop else None,
        foreground_probability=SETTING_B_PROTOCOL["foreground_crop_probability"],
        seed=seed,
    )


def _expert_validation_rows(
    model: torch.nn.Module,
    section_ids: list[int],
    thresholds: tuple[float, ...],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    dataset = _expert_dataset(section_ids, crop=False, seed=seed)
    rows: list[dict[str, Any]] = []
    model.eval()
    for index in range(len(dataset)):
        batch = dataset[index]
        probability = tiled_probability(model, batch["image"]).numpy()[:255, :701]
        target = batch["target"][0, :255, :701].numpy() >= 0.5
        valid = batch["valid"][0, :255, :701].numpy().astype(bool)
        for threshold in thresholds:
            rows.append(
                {
                    "section_id": int(batch["section_id"]),
                    "threshold": threshold,
                    **_binary_metrics(probability >= threshold, target, valid),
                }
            )
    return rows


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    run_hash: str,
    epoch: int,
    best_metric: float,
    patience_used: int,
) -> None:
    torch.save(
        {
            "run_hash": run_hash,
            "epoch": epoch,
            "best_metric": best_metric,
            "patience_used": patience_used,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "test_scores_used": False,
        },
        path,
    )


def run_setting_b_fold(
    spec: CRACKSRunSpec,
    fold: dict[str, Any],
    setting_a_root: Path,
    setting_a_expert_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    max_epochs: int | None = None,
    max_train_sections: int | None = None,
) -> dict[str, Any]:
    setting_a_receipt = verify_setting_a_complete(setting_a_root, setting_a_expert_root)
    source_dir = Path(setting_a_root) / f"{spec.run_id}-{spec.run_hash}"
    source_checkpoint = source_dir / "checkpoint-last.pt"
    source_sha = _sha256(source_checkpoint)
    run_hash = setting_b_run_hash(spec, fold, source_sha)
    run_id = f"{spec.model}_fold{fold['fold']}"
    run_dir = Path(output_root) / f"{run_id}-{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("run_hash") == run_hash:
            return {**existing, "action": "SKIP"}

    budget = int(max_epochs or SETTING_B_PROTOCOL["max_epochs"])
    seed = int(spec.seed)
    _set_seed(seed)
    torch_device = torch.device(device)
    model = build_real_model(spec).to(torch_device)
    load_real_checkpoint(source_checkpoint, spec.run_hash, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTING_B_PROTOCOL["learning_rate"])
    train = _expert_dataset(list(fold["train"]), crop=True, seed=seed)
    thresholds = tuple(float(value) for value in SETTING_B_PROTOCOL["threshold_candidates"])
    last_path = run_dir / "checkpoint-last.pt"
    best_path = run_dir / "checkpoint-best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_metric = -1.0
    patience_used = 0
    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if payload.get("run_hash") != run_hash or payload.get("test_scores_used") is not False:
            raise ValueError("Setting B resume provenance mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"])
        best_metric = float(payload["best_metric"])
        patience_used = int(payload["patience_used"])
        if status_path.exists():
            history = list(json.loads(status_path.read_text()).get("history", []))

    patience_budget = int(SETTING_B_PROTOCOL["early_stopping_patience"])
    epochs_to_run = range(start_epoch, budget) if patience_used < patience_budget else ()
    for epoch in epochs_to_run:
        model.train()
        train.set_epoch(epoch)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        if max_train_sections is not None:
            order = order[: int(max_train_sections)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accumulation = int(SETTING_B_PROTOCOL["effective_batch_size"])
        for position, dataset_index in enumerate(order):
            batch = train[dataset_index]
            logits = model(batch["image"].unsqueeze(0).to(torch_device))
            loss, _logs, _ = utils.segmentation_objective(
                logits,
                batch["target"].unsqueeze(0).to(torch_device),
                batch["valid"].unsqueeze(0).float().to(torch_device),
                topology_weight=0.2,
                topology_num_iters=5,
            )
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_rows = _expert_validation_rows(
            model, list(fold["validation"]), thresholds, seed=seed
        )
        selection = select_threshold(validation_rows)
        selected_row = next(
            row for row in selection["sweep"]
            if row["threshold"] == selection["selected_threshold"]
        )
        metric = float(selected_row["macro_dice"])
        improved = metric > best_metric + 1e-12
        if improved:
            best_metric = metric
            patience_used = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                run_hash=run_hash,
                epoch=epoch + 1,
                best_metric=best_metric,
                patience_used=patience_used,
            )
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            best_payload["selected_threshold"] = selection["selected_threshold"]
            torch.save(best_payload, best_path)
        else:
            patience_used += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_macro_dice": metric,
                "selected_threshold": selection["selected_threshold"],
                "improved": improved,
            }
        )
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            run_hash=run_hash,
            epoch=epoch + 1,
            best_metric=best_metric,
            patience_used=patience_used,
        )
        status_path.write_text(
            json.dumps(
                {
                    "status": "IN_PROGRESS",
                    "run_id": run_id,
                    "run_hash": run_hash,
                    "epoch": epoch + 1,
                    "history": history,
                    "test_scores_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"phase=cracks_setting_b model={spec.model} fold={fold['fold']} epoch={epoch + 1}/{budget} "
            f"val_metric={metric:.4f} test=LOCKED status=RUNNING"
        )
        if patience_used >= patience_budget:
            break

    if not best_path.exists():
        raise AssertionError("Setting B did not produce a best checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("test_scores_used") is not False:
        raise ValueError("Setting B best checkpoint was selected using test scores")
    selection_core = {
        "run_id": run_id,
        "run_hash": run_hash,
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_dice": float(best["best_metric"]),
        "selected_threshold": float(best["selected_threshold"]),
        "validation_section_ids": list(fold["validation"]),
        "test_scores_used": False,
        "best_checkpoint_sha256": _sha256(best_path),
    }
    selection = {**selection_core, "sha256": hashlib.sha256(json.dumps(selection_core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    selection_path = run_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    model.load_state_dict(best["model_state"])
    model.to(torch_device).eval()
    test_dataset = _expert_dataset(list(fold["test"]), crop=False, seed=seed)
    test_rows = []
    for index in range(len(test_dataset)):
        batch = test_dataset[index]
        probability = tiled_probability(model, batch["image"]).numpy()[:255, :701]
        target = batch["target"][0, :255, :701].numpy() >= 0.5
        valid = batch["valid"][0, :255, :701].numpy().astype(bool)
        test_rows.append(
            {
                "model": spec.model,
                "fold": int(fold["fold"]),
                "section_id": int(batch["section_id"]),
                "threshold": selection["selected_threshold"],
                **evaluate_binary_section(
                    probability,
                    target,
                    valid,
                    selection["selected_threshold"],
                    orientation_sensitivity_radii=(3, 7),
                ),
            }
        )
    rows_path = run_dir / "test_sections.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(test_rows)
    summary_keys = ("dice", "iou", "cldice", "skeleton_f1_at_2px", "fragmentation", "trace_orientation_error_median_deg")
    summary = {key: float(np.mean([float(row[key]) for row in test_rows])) for key in summary_keys}
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_id": run_id,
        "run_hash": run_hash,
        "model": spec.model,
        "fold": int(fold["fold"]),
        "source_checkpoint_sha256": source_sha,
        "setting_a_receipt_sha256": setting_a_receipt["sha256"],
        "selection_sha256": selection["sha256"],
        "test_section_count": len(test_rows),
        "test_scores_used_after_selection": True,
        "summary": summary,
        "history": history,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
