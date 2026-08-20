"""Bounded RC1 membership-only repair for the frozen ANZA-2 field.

The module deliberately has no confirm/CRACKS/expert entry point.  Geometry,
the generic head, and fusion beta remain bitwise frozen during membership repair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from anza2.eval.low_fpr import low_fpr_metrics, sampled_operating_curve
from anza2.eval.mechanism_metrics import aggregate_mechanism, mechanism_observations
from anza2.forensics.component_replacement import align_learned_field, oracle_field_from_sample
from anza2.forensics.field_fidelity import aggregate_fidelity, field_fidelity_row
from anza2_experiment.learned_affinity import LearnedAffinityModel, _batch, _sample, canonical_hash, set_seed
from anza2_experiment.learned_affinity_repair import protocol_payload as phase3b_protocol_payload
from models.anza2.field import ANZA2FieldOutput
from models.anza2.losses import rc1_membership_loss
from synthetic.affinity_targets import build_affinity_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3c_b_rc1"
PARENT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3b"
PHASE3C_A_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3c_a"
SEEDS = (41, 42, 43)
CONFIGS = {"M-A": 0.25, "M-B": 0.50}
TRAIN_SAMPLES = 256
DEVELOPMENT_SAMPLES = 512
MONITOR_SAMPLES = 64
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.001
LAMBDA_COUNT = 0.25
LOW_FPR_BUDGET = 0.05
MINIMUM_TPR_DELTA = 0.08
ABSOLUTE_MECHANISM_THRESHOLD = 0.04482836276292801


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parent_checkpoints() -> dict[int, Path]:
    return {seed: PARENT_ROOT / "runs" / f"causal_s{seed}" / "checkpoint-last.pt" for seed in SEEDS}


def protocol_payload() -> dict[str, Any]:
    checkpoints = parent_checkpoints()
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen Phase-3B checkpoints: {missing}")
    return {
        "version": "anza2_phase3c_b_rc1_membership_only_v1",
        "parent_phase3c_a_protocol_sha256": (PHASE3C_A_ROOT / "protocol_hash.txt").read_text().strip(),
        "parent_status": "PHASE3C_A_FORENSIC_PASS_ROOT_CAUSE_MEMBERSHIP_LEARNING",
        "parent_phase3b_checkpoints": {str(seed): _digest(path) for seed, path in checkpoints.items()},
        "configurations": {name: {"lambda_bg": value} for name, value in CONFIGS.items()},
        "lambda_count": LAMBDA_COUNT,
        "coverage_gamma": 2.0,
        "selection_seed": 41,
        "confirmation_seeds": list(SEEDS),
        "train_stream": "CrossingTraceBench-v4 train[0:256]",
        "development_stream": "CrossingTraceBench-v4 validation[0:512]",
        "monitor_stream": "CrossingTraceBench-v4 train[0:64]; diagnostics only",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "trainable_parameters_membership_stage": "field.membership_head only",
        "membership_safety_gate": {
            "recall_min": 0.90,
            "all_zero_target_max": 0.05,
            "target_median_gt_inactive_median": True,
            "parallel_false_bridge_max": 0.02,
        },
        "single_seed_mechanism_gate": {
            "raw_tpr_at_fpr_0_05_min": 0.45,
            "overall_branch_recall_min": 0.98,
            "x_branch_recall_min": 0.95,
            "x_two_mode_fraction_min": 0.90,
        },
        "three_seed_gate": {
            "median_membership_recall_min": 0.90,
            "each_membership_recall_min": 0.85,
            "mean_all_zero_target_max": 0.05,
            "mean_raw_tpr_at_fpr_0_05_min": 0.45,
            "each_parallel_false_bridge_max": 0.02,
        },
        "beta_fit_stream": "train only",
        "beta_optimizer": "deterministic LBFGS one scalar with softplus nonnegative parameterization",
        "primary_metric": "OFF/ON TPR delta at independently selected FPR<=0.05 thresholds",
        "minimum_tpr_delta": MINIMUM_TPR_DELTA,
        "ci_lower_required": 0.0,
        "pauc_delta_required": 0.0,
        "bootstrap_unit": "synthetic sample after averaging three seeds",
        "bootstrap_resamples": 10_000,
        "confirm_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }


def configure_membership_only(model: LearnedAffinityModel) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.field.membership_head.parameters():
        parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def configure_beta_only(model: LearnedAffinityModel) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.combiner.beta_raw.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _load_parent(seed: int, device: torch.device) -> LearnedAffinityModel:
    path = parent_checkpoints()[seed]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("protocol_sha256") != canonical_hash(phase3b_protocol_payload()):
        raise ValueError("frozen Phase-3B checkpoint identity mismatch")
    if payload.get("cracks_data_accessed") is not False or payload.get("expert_data_accessed") is not False:
        raise ValueError("parent checkpoint data-lock violation")
    model = LearnedAffinityModel(initial_beta=0.05)
    model.load_state_dict(payload["model_state"])
    return model.to(device)


def _frozen_snapshot(model: LearnedAffinityModel, *, except_prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(except_prefix)
    }


def _assert_unchanged(model: LearnedAffinityModel, snapshot: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    changed = [name for name, before in snapshot.items() if not torch.equal(before, current[name].detach().cpu())]
    if changed:
        raise AssertionError(f"frozen parameters changed: {changed}")


def _target_orientation(targets: dict[str, torch.Tensor]) -> torch.Tensor:
    theta = targets["theta"].float()
    return torch.stack((torch.cos(2.0 * theta), torch.sin(2.0 * theta)), dim=2)


@torch.inference_mode()
def _monitor_membership(model: LearnedAffinityModel, device: torch.device) -> dict[str, float | None]:
    model.eval()
    target_values: list[np.ndarray] = []
    inactive_values: list[np.ndarray] = []
    active_hits = total_targets = all_zero = target_pixels = 0
    x_correct = x_pixels = 0
    background_unions: list[np.ndarray] = []
    for start in range(0, MONITOR_SAMPLES, BATCH_SIZE):
        images, targets = _batch("train", list(range(start, start + BATCH_SIZE)), 64, device)
        features = model.encoder(images)
        field = model.field(features)
        target_orientation = _target_orientation(targets)
        similarity = torch.einsum("brchw,bkchw->brkhw", field.orientation, target_orientation)
        compatibility = ((1.0 + similarity) / 2.0).clamp(0.0, 1.0).square()
        best_modes = compatibility.argmax(dim=1)
        matched = torch.gather(field.membership, 1, best_modes)
        valid = targets["theta_valid"].bool()
        target_values.append(matched[valid].cpu().numpy())
        active_hits += int((matched[valid] >= 0.5).sum())
        total_targets += int(valid.sum())
        predicted_count = (field.membership >= 0.5).sum(dim=1)
        count = targets["mode_count"]
        positive = count > 0
        all_zero += int((predicted_count[positive] == 0).sum()); target_pixels += int(positive.sum())
        x_mask = count >= 2
        x_correct += int((predicted_count[x_mask] == count[x_mask]).sum()); x_pixels += int(x_mask.sum())
        selected = torch.zeros_like(field.membership, dtype=torch.bool)
        selected.scatter_(1, best_modes, valid)
        inactive_values.append(field.membership[~selected].cpu().numpy())
        background = ~valid.any(dim=1)
        union = 1.0 - torch.prod(1.0 - field.membership, dim=1)
        background_unions.append(union[background].cpu().numpy())
    target_array = np.concatenate(target_values); inactive_array = np.concatenate(inactive_values)
    return {
        "membership_active_recall": active_hits / max(total_targets, 1),
        "all_zero_fraction_target_pixels": all_zero / max(target_pixels, 1),
        "target_membership_median": float(np.median(target_array)),
        "inactive_membership_median": float(np.median(inactive_array)),
        "x_correct_count_fraction": x_correct / max(x_pixels, 1),
        "background_fuzzy_union_mean": float(np.mean(np.concatenate(background_unions))),
    }


def train_membership_repair(
    config_name: str,
    seed: int,
    run_dir: Path,
    *,
    device: torch.device,
    protocol_hash: str,
) -> LearnedAffinityModel:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown RC1 config: {config_name}")
    set_seed(seed)
    model = _load_parent(seed, device)
    trainable = configure_membership_only(model)
    if trainable != ["field.membership_head.weight", "field.membership_head.bias"]:
        raise AssertionError(f"unexpected RC1 trainable parameters: {trainable}")
    snapshot = _frozen_snapshot(model, except_prefix="field.membership_head.")
    optimizer = torch.optim.Adam(model.field.membership_head.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        model.train(); totals: list[float] = []; parts = {"cover": [], "background": [], "count_positive": []}
        order = np.random.default_rng(seed * 30_000 + epoch).permutation(TRAIN_SAMPLES)
        for start in range(0, TRAIN_SAMPLES, BATCH_SIZE):
            images, targets = _batch("train", order[start:start + BATCH_SIZE].tolist(), 64, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = model.encoder(images)
            field = model.field(features)
            loss, terms = rc1_membership_loss(
                field.orientation,
                field.membership,
                _target_orientation(targets),
                targets["theta_valid"].bool(),
                targets["mode_count"].float(),
                lambda_bg=CONFIGS[config_name],
                lambda_count=LAMBDA_COUNT,
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite RC1 membership loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.field.membership_head.parameters()):
                raise ValueError("missing or non-finite membership-head gradient")
            optimizer.step()
            totals.append(float(loss.detach()))
            for name, value in terms.items():
                parts[name].append(float(value.detach()))
        _assert_unchanged(model, snapshot)
        monitor = _monitor_membership(model, device)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(totals)),
            **{f"loss_{name}": float(np.mean(values)) for name, values in parts.items()},
            **monitor,
        }
        history.append(row)
        print(
            f"phase=rc1_membership config={config_name} seed={seed} epoch={epoch + 1}/{EPOCHS} "
            f"loss={row['loss']:.5f} recall={row['membership_active_recall']:.4f} "
            f"all_zero={row['all_zero_fraction_target_pixels']:.4f} confirm=CLOSED",
            flush=True,
        )
    _assert_unchanged(model, snapshot)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": config_name,
        "lambda_bg": CONFIGS[config_name],
        "seed": seed,
        "history": history,
        "protocol_sha256": protocol_hash,
        "parent_checkpoint_sha256": _digest(parent_checkpoints()[seed]),
        "trainable_parameters": trainable,
        "frozen_parameters_bitwise_unchanged": True,
        "confirm_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }, run_dir / "checkpoint-last.pt")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "COMPLETE",
        "config": config_name,
        "seed": seed,
        "history": history,
        "frozen_parameters_bitwise_unchanged": True,
        "confirm_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return model


def _slice_field(field: ANZA2FieldOutput, index: int) -> ANZA2FieldOutput:
    return ANZA2FieldOutput(
        field.membership[index:index + 1],
        field.orientation[index:index + 1],
        field.base_scale[index:index + 1],
        field.hyperbolicity[index:index + 1],
        field.sigma_parallel[index:index + 1],
        field.sigma_perpendicular[index:index + 1],
    )


@torch.inference_mode()
def evaluate_raw_anza(
    model: LearnedAffinityModel,
    *,
    seed: int,
    split: str,
    count: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    positive_scores: list[np.ndarray] = []; negative_scores: list[np.ndarray] = []
    mechanism_rows: list[dict[str, Any]] = []; fidelity_rows: list[dict[str, Any]] = []
    sample_scores: list[dict[str, Any]] = []
    for start in range(0, count, BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, count)))
        images, targets = _batch(split, indices, 64, device)
        output = model(images, use_anza=True)
        for local, sample_index in enumerate(indices):
            sample = _sample(split, sample_index, 64)
            relation = output["anza_affinity"][local].cpu().numpy().astype(np.float32)
            edge_targets = build_affinity_targets(sample, tuple(model.anza_affinity.offsets))
            positive = np.asarray(edge_targets["affinity_positive"], dtype=bool)
            negative = np.asarray(edge_targets["affinity_hard_negative"], dtype=bool)
            positive_values = relation[positive]; negative_values = relation[negative]
            if positive_values.size: positive_scores.append(positive_values)
            if negative_values.size: negative_scores.append(negative_values)
            sample_scores.append({
                "index": sample_index,
                "positive_scores": positive_values,
                "negative_scores": negative_values,
            })
            for row in mechanism_observations(sample, relation):
                mechanism_rows.append({"seed": seed, "sample_index": sample_index, "case": sample["case"], **row})
            oracle, valid = oracle_field_from_sample(sample, device=device)
            learned, _mapping = align_learned_field(_slice_field(output["field"], local), oracle, valid)
            fidelity_rows.append(field_fidelity_row(
                sample, learned, oracle, valid, seed=seed, sample_index=sample_index
            ))
    edge_metrics = low_fpr_metrics(np.concatenate(positive_scores), np.concatenate(negative_scores))
    mechanism = aggregate_mechanism(mechanism_rows, threshold=ABSOLUTE_MECHANISM_THRESHOLD)
    fidelity = aggregate_fidelity(fidelity_rows)
    x_two_mode_fraction = 1.0 - float(fidelity["overall"]["one_mode_collapse_fraction_crossing"])
    metrics = {
        "seed": seed,
        "split": split,
        "count": count,
        "edge": edge_metrics,
        "mechanism": mechanism,
        "membership_fidelity": fidelity,
        "x_two_mode_fraction": x_two_mode_fraction,
    }
    return metrics, sample_scores, mechanism_rows


def membership_safety_pass(metrics: dict[str, Any]) -> bool:
    field = metrics["membership_fidelity"]["overall"]
    mechanism = metrics["mechanism"]
    return bool(
        field["active_mode_recall"] >= 0.90
        and field["all_zero_fraction_target_pixels"] <= 0.05
        and field["membership_active_median"] > field["membership_inactive_median"]
        and mechanism["parallel_fault_false_bridge"] <= 0.02
    )


def single_seed_gate_pass(metrics: dict[str, Any]) -> bool:
    mechanism = metrics["mechanism"]
    return bool(
        membership_safety_pass(metrics)
        and metrics["x_two_mode_fraction"] >= 0.90
        and metrics["edge"]["tpr_at_fpr_0_05"] >= 0.45
        and mechanism["overall_branch_recall"] >= 0.98
        and mechanism["x_branch_recall"] >= 0.95
    )


def _load_repaired_checkpoint(path: Path, device: torch.device, protocol_hash: str) -> LearnedAffinityModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_sha256") != protocol_hash:
        raise ValueError("RC1 checkpoint protocol mismatch")
    if any(payload.get(key) is not False for key in ("confirm_opened", "cracks_data_accessed", "expert_data_accessed")):
        raise ValueError("RC1 checkpoint data-lock violation")
    model = LearnedAffinityModel(initial_beta=0.05)
    model.load_state_dict(payload["model_state"])
    return model.to(device)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _train_or_load(
    config_name: str,
    seed: int,
    run_dir: Path,
    *,
    device: torch.device,
    protocol_hash: str,
) -> LearnedAffinityModel:
    checkpoint = run_dir / "checkpoint-last.pt"
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("config") != config_name or payload.get("seed") != seed:
            raise ValueError("existing RC1 run identity mismatch")
        print(f"phase=rc1_membership config={config_name} seed={seed} status=RESUME_COMPLETE", flush=True)
        return _load_repaired_checkpoint(checkpoint, device, protocol_hash)
    return train_membership_repair(config_name, seed, run_dir, device=device, protocol_hash=protocol_hash)


@torch.inference_mode()
def _fixed_edge_records(
    model: LearnedAffinityModel,
    *,
    split: str,
    count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval(); records: list[dict[str, Any]] = []
    for index in range(count):
        images, targets = _batch(split, [index], 64, device)
        output = model(images, use_anza=True)
        generic_logits = output["generic_logits"][0]
        prior_logits = torch.logit(output["anza_affinity"][0].clamp(1e-6, 1.0 - 1e-6))
        positive = targets["positive"][0].bool(); negative = targets["negative"][0].bool()
        records.append({
            "index": index,
            "positive_generic_logits": generic_logits[positive].cpu().numpy().astype(np.float32),
            "negative_generic_logits": generic_logits[negative].cpu().numpy().astype(np.float32),
            "positive_prior_logits": prior_logits[positive].cpu().numpy().astype(np.float32),
            "negative_prior_logits": prior_logits[negative].cpu().numpy().astype(np.float32),
        })
    return records


def fit_beta_train_only(
    model: LearnedAffinityModel,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    records = _fixed_edge_records(model, split="train", count=TRAIN_SAMPLES, device=device)
    positive_generic = torch.as_tensor(
        np.concatenate([row["positive_generic_logits"] for row in records]), device=device
    )
    negative_generic = torch.as_tensor(
        np.concatenate([row["negative_generic_logits"] for row in records]), device=device
    )
    positive_prior = torch.as_tensor(
        np.concatenate([row["positive_prior_logits"] for row in records]), device=device
    )
    negative_prior = torch.as_tensor(
        np.concatenate([row["negative_prior_logits"] for row in records]), device=device
    )
    configure_beta_only(model)
    snapshot = _frozen_snapshot(model, except_prefix="combiner.beta_raw")
    before = float(model.combiner.beta.detach())
    optimizer = torch.optim.LBFGS(
        [model.combiner.beta_raw], lr=0.5, max_iter=100,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def balanced_loss() -> torch.Tensor:
        beta = model.combiner.beta
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_generic + beta * positive_prior, torch.ones_like(positive_generic)
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_generic + beta * negative_prior, torch.zeros_like(negative_generic)
        )
        return 0.5 * (positive_loss + negative_loss)

    initial_loss = float(balanced_loss().detach())

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_loss()
        if not torch.isfinite(loss):
            raise ValueError("non-finite beta loss")
        loss.backward()
        return loss

    optimizer.step(closure)
    _assert_unchanged(model, snapshot)
    after = float(model.combiner.beta.detach())
    return {
        "seed": seed,
        "fit_stream": "train[0:256]",
        "optimizer": "LBFGS",
        "beta_before": before,
        "beta_after": after,
        "balanced_loss_before": initial_loss,
        "balanced_loss_after": float(balanced_loss().detach()),
        "only_beta_raw_trainable": True,
        "all_other_parameters_bitwise_unchanged": True,
        "development_used_for_fit": False,
    }


def _fusion_scores(records: list[dict[str, Any]], beta: float) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    generic_rows: list[dict[str, Any]] = []; fused_rows: list[dict[str, Any]] = []
    for row in records:
        gp = row["positive_generic_logits"]; gn = row["negative_generic_logits"]
        ap = row["positive_prior_logits"]; an = row["negative_prior_logits"]
        generic_rows.append({
            "index": row["index"],
            "positive_scores": 1.0 / (1.0 + np.exp(-gp)),
            "negative_scores": 1.0 / (1.0 + np.exp(-gn)),
        })
        fused_rows.append({
            "index": row["index"],
            "positive_scores": 1.0 / (1.0 + np.exp(-(gp + beta * ap))),
            "negative_scores": 1.0 / (1.0 + np.exp(-(gn + beta * an))),
        })
    def pooled(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([row["positive_scores"] for row in rows if len(row["positive_scores"])]),
            np.concatenate([row["negative_scores"] for row in rows if len(row["negative_scores"])]),
        )
    generic_positive, generic_negative = pooled(generic_rows)
    fused_positive, fused_negative = pooled(fused_rows)
    return (
        low_fpr_metrics(generic_positive, generic_negative),
        low_fpr_metrics(fused_positive, fused_negative),
        generic_rows,
        fused_rows,
    )


def _bootstrap_delta(
    generic_by_seed: dict[int, list[dict[str, Any]]],
    fused_by_seed: dict[int, list[dict[str, Any]]],
    metrics_by_seed: dict[int, dict[str, Any]],
) -> tuple[float, list[float]]:
    sample_deltas = []
    for index in range(DEVELOPMENT_SAMPLES):
        seed_deltas = []
        for seed in SEEDS:
            generic = generic_by_seed[seed][index]["positive_scores"]
            fused = fused_by_seed[seed][index]["positive_scores"]
            if len(generic):
                seed_deltas.append(float(
                    np.mean(fused >= metrics_by_seed[seed]["on"]["threshold"])
                    - np.mean(generic >= metrics_by_seed[seed]["off"]["threshold"])
                ))
        if seed_deltas:
            sample_deltas.append(float(np.mean(seed_deltas)))
    rng = np.random.default_rng(20260818)
    array = np.asarray(sample_deltas, dtype=np.float64)
    boot = np.asarray([
        np.mean(rng.choice(array, len(array), replace=True)) for _ in range(10_000)
    ])
    return float(array.mean()), [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def three_seed_gate_pass(seed_metrics: list[dict[str, Any]]) -> bool:
    recalls = [row["membership_fidelity"]["overall"]["active_mode_recall"] for row in seed_metrics]
    all_zero = [row["membership_fidelity"]["overall"]["all_zero_fraction_target_pixels"] for row in seed_metrics]
    tprs = [row["edge"]["tpr_at_fpr_0_05"] for row in seed_metrics]
    false_bridges = [row["mechanism"]["parallel_fault_false_bridge"] for row in seed_metrics]
    return bool(
        np.median(recalls) >= 0.90
        and min(recalls) >= 0.85
        and np.mean(all_zero) <= 0.05
        and np.mean(tprs) >= 0.45
        and max(false_bridges) <= 0.02
    )


def _curve_rows(source: str, seed: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = np.concatenate([row["positive_scores"] for row in rows if len(row["positive_scores"])])
    negative = np.concatenate([row["negative_scores"] for row in rows if len(row["negative_scores"])])
    return [{"source": source, "seed": seed, **row} for row in sampled_operating_curve(positive, negative)]


def _phase_report(result: dict[str, Any]) -> str:
    selected = result.get("selected_config")
    lines = [
        "# ANZA-2 Phase 3C-B RC1 report", "",
        f"- Status: `{result['status']}`.",
        "- Repair scope: only `field.membership_head`; ANZA geometry, encoder, generic head, and beta were frozen during membership repair.",
        f"- Selected configuration: `{selected}`." if selected else "- Selected configuration: none.",
        "- Confirm, CRACKS, expert: closed.", "",
        "## Questions", "",
    ]
    answers = result.get("answers", {})
    for number, key in enumerate((
        "membership_fixed", "raw_selectivity_restored", "false_bridge_controlled",
        "beta_incremental_gain", "practical_gate_passed", "confirm_allowed",
    ), start=1):
        lines.append(f"{number}. {answers.get(key, 'NOT_REACHED')}")
    lines.extend(["", "## Claim boundary", "",
        "All results are from frozen CrossingTraceBench-v4 train/development streams. No untouched confirm, CRACKS, or expert data were opened. A positive raw ANZA result is not a real-data claim.", ""])
    return "\n".join(lines)


def run_rc1(output_root: Path = OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(); protocol_hash = canonical_hash(protocol)
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("RC1 protocol drift")
    protocol_path.write_text(encoded)
    (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    (output_root / "parent_phase3c_a_hash.txt").write_text(protocol["parent_phase3c_a_protocol_sha256"] + "\n")
    _write_json(output_root / "membership_loss_spec.json", {
        "coverage": "-mean(log(max_r(mu_r * ((1 + o_r dot t_k)/2)^2) + eps)) over valid target axes",
        "count_positive": "mean((sum_r(mu_r) - K)^2) over K>0 pixels only",
        "background": "-mean(log(1 - fuzzy_union + eps)) over K=0 pixels only",
        "total": "coverage + lambda_bg * background + 0.25 * count_positive",
        "configurations": protocol["configurations"],
    })
    torch.set_num_threads(min(2, torch.get_num_threads()))
    device_obj = torch.device(device)

    selection_rows: list[dict[str, Any]] = []
    selection_scores: dict[str, list[dict[str, Any]]] = {}
    curve_rows: list[dict[str, Any]] = []
    for config_name in CONFIGS:
        tag = config_name.lower().replace("-", "")
        model = _train_or_load(
            config_name, 41, output_root / "runs" / f"{tag}_s41",
            device=device_obj, protocol_hash=protocol_hash,
        )
        metrics, scores, _mechanism = evaluate_raw_anza(
            model, seed=41, split="validation", count=DEVELOPMENT_SAMPLES, device=device_obj
        )
        metrics["config"] = config_name
        metrics["membership_safety_pass"] = membership_safety_pass(metrics)
        metrics["single_seed_mechanism_pass"] = single_seed_gate_pass(metrics)
        selection_rows.append(metrics); selection_scores[config_name] = scores
        curve_rows.extend(_curve_rows(f"raw_{config_name}", 41, scores))
        print(
            f"phase=rc1_selection config={config_name} safety={metrics['membership_safety_pass']} "
            f"gate={metrics['single_seed_mechanism_pass']} tpr={metrics['edge']['tpr_at_fpr_0_05']:.4f} "
            f"false_bridge={metrics['mechanism']['parallel_fault_false_bridge']:.4f}", flush=True,
        )

    eligible = [row for row in selection_rows if row["membership_safety_pass"]]
    selected = max(eligible, key=lambda row: row["edge"]["tpr_at_fpr_0_05"])["config"] if eligible else None
    selection = {
        "status": "SELECTED" if selected else "STOP_RC1_MEMBERSHIP_REPAIR_FAILED",
        "selected_config": selected,
        "selection_rule": "membership safety first, then maximum raw ANZA TPR@FPR<=0.05",
        "config_metrics": selection_rows,
        "confirm_opened": False,
    }
    _write_json(output_root / "selected_config.json", selection)
    _write_json(output_root / "membership_fidelity.json", {
        "selection": {row["config"]: row["membership_fidelity"] for row in selection_rows},
        "three_seed": None,
    })
    _write_json(output_root / "raw_anza_metrics.json", {"selection": selection_rows, "three_seed": None})

    if selected is None or not next(row for row in selection_rows if row["config"] == selected)["single_seed_mechanism_pass"]:
        status = "STOP_RC1_MEMBERSHIP_REPAIR_FAILED"
        result = {
            "status": status, "protocol_sha256": protocol_hash,
            "selected_config": selected, "selection": selection,
            "answers": {
                "membership_fixed": "NO: no predeclared M-A/M-B configuration passed the full single-seed gate.",
                "raw_selectivity_restored": "NO", "false_bridge_controlled": "NO",
                "beta_incremental_gain": "NOT_REACHED", "practical_gate_passed": "NO",
                "confirm_allowed": "NO",
            },
            "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
        }
        _finalize_artifacts(output_root, result, curve_rows, protocol_hash)
        return result

    three_seed_metrics: list[dict[str, Any]] = []
    repaired_models: dict[int, LearnedAffinityModel] = {}
    for seed in SEEDS:
        model = _train_or_load(
            selected, seed, output_root / "three_seed_runs" / f"{selected.lower().replace('-', '')}_s{seed}",
            device=device_obj, protocol_hash=protocol_hash,
        )
        metrics, scores, _mechanism = evaluate_raw_anza(
            model, seed=seed, split="validation", count=DEVELOPMENT_SAMPLES, device=device_obj
        )
        metrics["config"] = selected
        three_seed_metrics.append(metrics); repaired_models[seed] = model
        curve_rows.extend(_curve_rows("raw_three_seed", seed, scores))
        print(
            f"phase=rc1_three_seed seed={seed} recall={metrics['membership_fidelity']['overall']['active_mode_recall']:.4f} "
            f"tpr={metrics['edge']['tpr_at_fpr_0_05']:.4f} false_bridge={metrics['mechanism']['parallel_fault_false_bridge']:.4f}",
            flush=True,
        )
    stable = three_seed_gate_pass(three_seed_metrics)
    _write_json(output_root / "membership_fidelity.json", {
        "selection": {row["config"]: row["membership_fidelity"] for row in selection_rows},
        "three_seed": {str(row["seed"]): row["membership_fidelity"] for row in three_seed_metrics},
    })
    _write_json(output_root / "raw_anza_metrics.json", {
        "selection": selection_rows, "three_seed": three_seed_metrics,
        "three_seed_gate_pass": stable,
    })
    if not stable:
        result = {
            "status": "RC1_REPAIR_UNSTABLE", "protocol_sha256": protocol_hash,
            "selected_config": selected, "selection": selection,
            "three_seed_metrics": three_seed_metrics,
            "answers": {
                "membership_fixed": "UNSTABLE across seeds", "raw_selectivity_restored": "UNSTABLE",
                "false_bridge_controlled": "See per-seed metrics", "beta_incremental_gain": "NOT_REACHED",
                "practical_gate_passed": "NO", "confirm_allowed": "NO",
            },
            "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
        }
        _finalize_artifacts(output_root, result, curve_rows, protocol_hash)
        return result

    beta_rows = []; development_seed_metrics: dict[int, dict[str, Any]] = {}
    generic_by_seed: dict[int, list[dict[str, Any]]] = {}; fused_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed, model in repaired_models.items():
        beta = fit_beta_train_only(model, seed=seed, device=device_obj)
        beta_rows.append(beta)
        records = _fixed_edge_records(model, split="validation", count=DEVELOPMENT_SAMPLES, device=device_obj)
        off, on, generic_rows, fused_rows = _fusion_scores(records, beta["beta_after"])
        development_seed_metrics[seed] = {"off": off, "on": on, "tpr_delta": on["tpr_at_fpr_0_05"] - off["tpr_at_fpr_0_05"], "pauc_delta": on["low_fpr_pauc_normalized"] - off["low_fpr_pauc_normalized"]}
        generic_by_seed[seed], fused_by_seed[seed] = generic_rows, fused_rows
        curve_rows.extend(_curve_rows("generic_off", seed, generic_rows)); curve_rows.extend(_curve_rows("anza_on", seed, fused_rows))
        checkpoint_dir = output_root / "three_seed_runs" / f"{selected.lower().replace('-', '')}_s{seed}"
        torch.save({
            "model_state": model.state_dict(), "config": selected, "seed": seed,
            "protocol_sha256": protocol_hash, "beta_fit": beta,
            "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
        }, checkpoint_dir / "checkpoint-beta-fitted.pt")
    delta, ci = _bootstrap_delta(generic_by_seed, fused_by_seed, development_seed_metrics)
    pauc_delta = float(np.mean([row["pauc_delta"] for row in development_seed_metrics.values()]))
    development_gate = bool(delta >= MINIMUM_TPR_DELTA and ci[0] > 0.0 and pauc_delta > 0.0)
    if development_gate:
        status = "RC1_DEVELOPMENT_GATE_PASS_CONFIRM_FREEZE_PREPARED"
    elif delta > 0.0:
        status = "RC1_REPAIR_POSITIVE_BUT_PRACTICALLY_INSUFFICIENT"
    else:
        status = "STOP_RC1_NO_INCREMENTAL_VALUE"
    development = {
        "status": status,
        "seed_metrics": {str(key): value for key, value in development_seed_metrics.items()},
        "three_seed_tpr_delta": delta,
        "three_seed_tpr_delta_ci95": ci,
        "mean_pauc_delta": pauc_delta,
        "minimum_tpr_delta": MINIMUM_TPR_DELTA,
        "gate_pass": development_gate,
        "confirm_opened": False,
    }
    _write_json(output_root / "beta_fit.json", {"status": "COMPLETE", "fits": beta_rows})
    _write_json(output_root / "development_metrics.json", development)
    result = {
        "status": status, "protocol_sha256": protocol_hash,
        "selected_config": selected, "selection": selection,
        "three_seed_metrics": three_seed_metrics, "beta_fit": beta_rows,
        "development": development,
        "answers": {
            "membership_fixed": "YES: the frozen three-seed membership gate passed.",
            "raw_selectivity_restored": "YES: the frozen raw-ANZA three-seed gate passed.",
            "false_bridge_controlled": "YES across all three seeds.",
            "beta_incremental_gain": f"Delta TPR={delta:+.6f}, CI={ci}, mean pAUC delta={pauc_delta:+.6f}.",
            "practical_gate_passed": "YES" if development_gate else "NO",
            "confirm_allowed": "YES, freeze prepared but confirm remains unopened." if development_gate else "NO",
        },
        "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    if development_gate:
        _write_json(output_root / "RC1_CONFIRM_FREEZE.json", {
            "protocol_sha256": protocol_hash,
            "selected_config": selected,
            "membership_weights": {"lambda_bg": CONFIGS[selected], "lambda_count": LAMBDA_COUNT, "gamma": 2.0},
            "checkpoint_hashes": {
                str(seed): _digest(output_root / "three_seed_runs" / f"{selected.lower().replace('-', '')}_s{seed}" / "checkpoint-beta-fitted.pt")
                for seed in SEEDS
            },
            "beta_values": {str(row["seed"]): row["beta_after"] for row in beta_rows},
            "threshold_rule": "lowest observed threshold with inclusive FPR<=0.05, separately OFF/ON",
            "fpr_budget": LOW_FPR_BUDGET,
            "primary_metric": protocol["primary_metric"],
            "required_delta": MINIMUM_TPR_DELTA,
            "bootstrap_code_sha256": _digest(Path(__file__)),
            "confirm_seed_range": "CrossingTraceBench-v4 confirm[0:512], still unopened",
            "cracks_data_accessed": False, "expert_data_accessed": False, "confirm_opened": False,
        })
    _finalize_artifacts(output_root, result, curve_rows, protocol_hash, bootstrap=development)
    return result


def _finalize_artifacts(
    output_root: Path,
    result: dict[str, Any],
    curve_rows: list[dict[str, Any]],
    protocol_hash: str,
    *,
    bootstrap: dict[str, Any] | None = None,
) -> None:
    if not (output_root / "beta_fit.json").exists():
        _write_json(output_root / "beta_fit.json", {"status": "NOT_RUN", "reason": result["status"]})
    if not (output_root / "development_metrics.json").exists():
        _write_json(output_root / "development_metrics.json", {"status": "NOT_RUN", "reason": result["status"], "confirm_opened": False})
    _write_csv(output_root / "operating_curve.csv", curve_rows or [{"status": "NO_CURVE"}])
    _write_json(output_root / "bootstrap.json", bootstrap or {"status": "NOT_RUN", "reason": result["status"], "resamples": 0})
    _write_json(output_root / "metrics.json", result)
    (output_root / "PHASE3C_B_RC1_REPORT.md").write_text(_phase_report(result))
    _write_json(output_root / "TASK_STATE.json", {
        "status": result["status"],
        "next_action": "validate and stop" if not result.get("development", {}).get("gate_pass") else "prepare a separate confirm command; do not open in this run",
        "confirm_opened": False,
    })
    _write_json(output_root / "EVIDENCE.json", {
        "status": result["status"], "protocol_sha256": protocol_hash,
        "selected_config": result.get("selected_config"), "answers": result.get("answers"),
        "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    })
    _write_json(output_root / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    _write_json(output_root / "code_state.json", {
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "commit_created": False,
    })
