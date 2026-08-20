"""Single bounded Phase-3B repair after the frozen Phase-3 development failure.

The repair changes only the diagnosed causes: it couples axial supervision to
mode membership and evaluates ANZA ON/OFF on one frozen generic checkpoint.
The confirm stream remains locked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.anza2.losses import active_mode_count_loss, membership_weighted_axis_set_coverage_loss
from synthetic.affinity_losses import balanced_affinity_bce
from .learned_affinity import (
    PROJECT_ROOT, SEEDS, LearnedAffinityModel, _batch, _metrics, _paired_bootstrap,
    _score, _threshold, canonical_hash, set_seed,
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3b"
GENERIC_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3" / "development" / "runs"
PHASE3_V1_PROTOCOL_SHA256 = "3f566a34db6332e029817f2242f455c6a9d40730120d5baca8fd1289c0279517"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generic_checkpoint_paths() -> dict[int, Path]:
    return {seed: GENERIC_ROOT / f"generic_s{seed}" / "checkpoint-last.pt" for seed in SEEDS}


def protocol_payload() -> dict[str, Any]:
    checkpoints = generic_checkpoint_paths()
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen generic checkpoints: {missing}")
    return {
        "version": "anza2_phase3b_membership_coupled_causal_v1",
        "parent_phase3_protocol_sha256": PHASE3_V1_PROTOCOL_SHA256,
        "parent_status": "DEVELOPMENT_COMPLETE_GATE_FAIL",
        "root_cause": "orientation set coverage was not coupled to active fuzzy membership; separately trained backbones confounded the causal comparison",
        "frozen_generic_checkpoints": {str(seed): _digest(path) for seed, path in checkpoints.items()},
        "seeds": list(SEEDS), "train_samples": 256, "development_samples": 256,
        "epochs": 5, "batch_size": 8, "learning_rate": 0.001, "image_size": 64,
        "trainable_parameters": "ANZA field heads and beta only; encoder and generic edge head frozen",
        "membership_loss_weight": 0.25, "membership_weighted_axis_loss_weight": 0.10,
        "mode_count_loss_weight": 0.05,
        "primary_comparison": "same checkpoint generic affinity OFF versus ANZA affinity ON",
        "primary_metric": "local-edge TPR at FPR <= 0.05",
        "minimum_tpr_delta": 0.08, "ci_lower_required": 0.0,
        "bootstrap_unit": "synthetic sample after averaging three seeds",
        "bootstrap_resamples": 10000,
        "confirm_open_rule": "Phase-3B development gate passes and threshold_freeze.json exists",
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }


def _load_generic(seed: int, device: torch.device) -> LearnedAffinityModel:
    path = generic_checkpoint_paths()[seed]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "generic" or payload.get("seed") != seed:
        raise ValueError("frozen generic checkpoint identity mismatch")
    if payload.get("cracks_data_accessed") is not False or payload.get("expert_data_accessed") is not False:
        raise ValueError("frozen generic checkpoint data-lock violation")
    model = LearnedAffinityModel(initial_beta=0.05)
    model.load_state_dict(payload["model_state"])
    return model.to(device)


def _train_repair(seed: int, output_root: Path, device: torch.device) -> LearnedAffinityModel:
    set_seed(seed)
    model = _load_generic(seed, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.field.parameters():
        parameter.requires_grad_(True)
    for parameter in model.combiner.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    history = []
    for epoch in range(5):
        model.train(); losses = []
        order = np.random.default_rng(seed * 20_000 + epoch).permutation(256)
        for start in range(0, 256, 8):
            images, targets = _batch("train", order[start:start + 8].tolist(), 64, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, use_anza=True); field = output["field"]
            edge = balanced_affinity_bce(output["logits"], targets["positive"], targets["negative"])
            fuzzy_union = 1.0 - torch.prod(1.0 - field.membership, dim=1)
            membership = F.binary_cross_entropy(fuzzy_union.clamp(1e-6, 1 - 1e-6), targets["visible"].float())
            theta = targets["theta"].float()
            target_orientation = torch.stack((torch.cos(2 * theta), torch.sin(2 * theta)), dim=2)
            orientation = membership_weighted_axis_set_coverage_loss(
                field.orientation, field.membership, target_orientation, targets["theta_valid"].bool()
            )
            count = active_mode_count_loss(
                field.membership, targets["mode_count"].float(),
                torch.ones_like(targets["visible"], dtype=torch.bool),
            )
            loss = edge + 0.25 * membership + 0.10 * orientation + 0.05 * count
            if not torch.isfinite(loss):
                raise ValueError("non-finite Phase-3B loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite Phase-3B gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "beta": float(model.combiner.beta.detach())}
        history.append(row)
        print(
            f"phase=anza2_phase3b seed={seed} epoch={epoch + 1}/5 "
            f"loss={row['loss']:.5f} beta={row['beta']:.5f}", flush=True,
        )
    run_dir = output_root / "runs" / f"causal_s{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "seed": seed, "history": history,
        "protocol_sha256": canonical_hash(protocol_payload()),
        "generic_checkpoint_sha256": _digest(generic_checkpoint_paths()[seed]),
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, run_dir / "checkpoint-last.pt")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "seed": seed, "history": history,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return model


def _aggregate_delta(
    off_by_seed: dict[int, list[dict[str, Any]]], on_by_seed: dict[int, list[dict[str, Any]]],
    thresholds: dict[int, dict[str, float]],
) -> tuple[float, list[float]]:
    sample_deltas = []
    for index in range(256):
        seed_values = []
        for seed in SEEDS:
            off = off_by_seed[seed][index]["positive_scores"]
            on = on_by_seed[seed][index]["positive_scores"]
            if len(off):
                seed_values.append(float(
                    np.mean(on >= thresholds[seed]["on"]) - np.mean(off >= thresholds[seed]["off"])
                ))
        if seed_values:
            sample_deltas.append(float(np.mean(seed_values)))
    rng = np.random.default_rng(20260818)
    boot = [float(np.mean(rng.choice(sample_deltas, len(sample_deltas), replace=True))) for _ in range(10_000)]
    return float(np.mean(sample_deltas)), [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def run_phase3b(output_root: Path = OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, Any]:
    protocol = protocol_payload(); protocol_hash = canonical_hash(protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("Phase-3B protocol drift")
    protocol_path.write_text(encoded); (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    torch.set_num_threads(min(2, torch.get_num_threads())); device_obj = torch.device(device)
    seed_rows = []; off_by_seed = {}; on_by_seed = {}; thresholds = {}
    for seed in SEEDS:
        model = _train_repair(seed, output_root, device_obj)
        off = _score(model, "generic", "validation", 256, 64, device_obj)
        on = _score(model, "generic_plus_anza", "validation", 256, 64, device_obj)
        off_threshold, on_threshold = _threshold(off), _threshold(on)
        off_by_seed[seed], on_by_seed[seed] = off, on
        thresholds[seed] = {"off": off_threshold, "on": on_threshold}
        off_metrics, on_metrics = _metrics(off, off_threshold), _metrics(on, on_threshold)
        delta, ci = _paired_bootstrap(off, on, {"generic": off_threshold, "generic_plus_anza": on_threshold}, resamples=2000)
        seed_rows.append({
            "seed": seed, "generic_off": off_metrics, "anza_on": on_metrics,
            "tpr_delta": delta, "tpr_delta_ci95": ci, "beta": float(model.combiner.beta.detach()),
        })
    delta, ci = _aggregate_delta(off_by_seed, on_by_seed, thresholds)
    gate = bool(delta >= protocol["minimum_tpr_delta"] and ci[0] > protocol["ci_lower_required"])
    result = {
        "status": "PHASE3B_DEVELOPMENT_GATE_PASS" if gate else "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN",
        "protocol_sha256": protocol_hash, "seed_metrics": seed_rows,
        "three_seed_tpr_delta": delta, "three_seed_tpr_delta_ci95": ci,
        "gate_pass": gate, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_root / "threshold_freeze.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash, "thresholds": thresholds,
        "confirm_opened": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return result


def evaluate_saved_phase3b(output_root: Path = OUTPUT_ROOT, *, device: str = "cpu") -> dict[str, Any]:
    """Re-evaluate frozen Phase-3B checkpoints after the inclusive-FPR audit."""

    protocol = protocol_payload(); protocol_hash = canonical_hash(protocol)
    if (output_root / "protocol_hash.txt").read_text().strip() != protocol_hash:
        raise ValueError("Phase-3B protocol/checkpoint audit mismatch")
    torch.set_num_threads(min(2, torch.get_num_threads())); device_obj = torch.device(device)
    seed_rows = []; off_by_seed = {}; on_by_seed = {}; thresholds = {}
    for seed in SEEDS:
        checkpoint = output_root / "runs" / f"causal_s{seed}" / "checkpoint-last.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != protocol_hash or payload.get("seed") != seed:
            raise ValueError("Phase-3B checkpoint identity mismatch")
        model = LearnedAffinityModel(initial_beta=0.05).to(device_obj)
        model.load_state_dict(payload["model_state"])
        off = _score(model, "generic", "validation", 256, 64, device_obj)
        on = _score(model, "generic_plus_anza", "validation", 256, 64, device_obj)
        off_threshold, on_threshold = _threshold(off), _threshold(on)
        off_by_seed[seed], on_by_seed[seed] = off, on
        thresholds[seed] = {"off": off_threshold, "on": on_threshold}
        off_metrics, on_metrics = _metrics(off, off_threshold), _metrics(on, on_threshold)
        delta, ci = _paired_bootstrap(
            off, on, {"generic": off_threshold, "generic_plus_anza": on_threshold}, resamples=2000
        )
        seed_rows.append({
            "seed": seed, "generic_off": off_metrics, "anza_on": on_metrics,
            "tpr_delta": delta, "tpr_delta_ci95": ci, "beta": float(model.combiner.beta.detach()),
        })
    delta, ci = _aggregate_delta(off_by_seed, on_by_seed, thresholds)
    gate = bool(delta >= protocol["minimum_tpr_delta"] and ci[0] > protocol["ci_lower_required"])
    result = {
        "status": "PHASE3B_DEVELOPMENT_GATE_PASS" if gate else "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN",
        "evaluation_audit": "inclusive threshold now satisfies FPR <= 0.05 under ties",
        "checkpoints_retrained": False, "protocol_sha256": protocol_hash,
        "seed_metrics": seed_rows, "three_seed_tpr_delta": delta,
        "three_seed_tpr_delta_ci95": ci, "minimum_tpr_delta": protocol["minimum_tpr_delta"],
        "gate_pass": gate, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    (output_root / "metrics_reaudited.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_root / "threshold_freeze_reaudited.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash, "thresholds": thresholds,
        "confirm_opened": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return result
