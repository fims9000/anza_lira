"""Zero-training structural-selectivity experiment for ANZA-2 Phase 2.

This experiment uses controlled oracle fields with weak but observable support.
It tests the algebraic value of displacement-aware multimodal geometry; it is
not a learned-image or CRACKS result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_dilation
import torch

from models.anza2.affinity import ANZA2StructuralAffinity, LOCAL8_OFFSETS, _shift_neighbor, shift_field
from models.anza2.field import ANZA2FieldOutput
from models.anza2.geometry import directed_geometry
from structural.widest_path import domain_restricted_widest_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase2"
METHODS = ("probability_only", "simple_axis_similarity", "legacy_global_normalized", "anza2_absolute")
SPLIT_SEED_BASE = {"development": 610_000_000, "confirm": 620_000_000}
SAMPLES_PER_STRATUM = 64
PATH_STRATA = ("straight_gap", "curved_gap", "parallel_false_bridge")
BRANCH_STRATA = ("x_crossing", "t_junction", "y_junction")
TARGET_FPR = 0.05
BOOTSTRAP_RESAMPLES = 10_000


def protocol_payload() -> dict[str, Any]:
    payload = {
        "version": "anza2_phase2_zero_train_mechanism_v1",
        "split_seed_base": SPLIT_SEED_BASE,
        "samples_per_stratum": SAMPLES_PER_STRATUM,
        "path_strata": PATH_STRATA,
        "branch_strata": BRANCH_STRATA,
        "methods": METHODS,
        "primary_metric": "positive path TPR at false-bridge FPR <= 0.05",
        "target_fpr": TARGET_FPR,
        "practical_delta_vs_strongest_control": 0.15,
        "parallel_false_bridge_max": 0.05,
        "crossing_junction_branch_recall_min": 0.95,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "field_status": "oracle weak-support field; no learned image inference",
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "training_performed": False,
        "confirm_open_rule": "protocol and development thresholds frozen first",
    }
    return payload


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _field_from_paths(
    size: int,
    paths: list[tuple[list[tuple[int, int]], float, bool]],
    *,
    scale: float,
    hyper: float,
) -> tuple[ANZA2FieldOutput, np.ndarray]:
    modes = 4
    membership = torch.full((1, modes, size, size), 0.005, dtype=torch.float64)
    angles = torch.zeros_like(membership)
    for mode in range(modes):
        angles[:, mode] = mode * math.pi / modes
    probability = np.full((size, size), 0.005, dtype=np.float32)
    for path_index, (path, strength, alternate_modes) in enumerate(paths):
        for index, (y, x) in enumerate(path):
            before = path[max(0, index - 1)]
            after = path[min(len(path) - 1, index + 1)]
            theta = math.atan2(after[0] - before[0], after[1] - before[1])
            mode = (index % 2) if alternate_modes else min(path_index, modes - 1)
            membership[0, mode, y, x] = float(strength)
            angles[0, mode, y, x] = theta
            probability[y, x] = max(probability[y, x], float(strength))
    orientation = torch.stack((torch.cos(2 * angles), torch.sin(2 * angles)), dim=2)
    base = torch.full_like(membership, float(scale))
    h = torch.full_like(membership, float(hyper))
    return ANZA2FieldOutput(membership, orientation, base, h, base * torch.exp(h), base * torch.exp(-h)), probability


def _path_fixture(case: str, seed: int, size: int = 33) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    center = size // 2
    weak = float(rng.uniform(0.28, 0.42))
    visible = float(rng.uniform(0.88, 0.98))
    scale = float(rng.uniform(0.85, 1.05))
    hyper = float(rng.uniform(0.85, 1.20))
    if case == "straight_gap":
        y = center + int(rng.integers(-2, 3))
        path = [(y, x) for x in range(4, size - 4)]
        strengths = []
        for _index, point in enumerate(path):
            strength = weak if center - 2 <= point[1] <= center + 2 else visible
            strengths.append((point, strength))
        field, probability = _field_from_paths(size, [(path, visible, False)], scale=scale, hyper=hyper)
        for (py, px), strength in strengths:
            field.membership[0, 0, py, px] = strength
            probability[py, px] = strength
        domain = binary_dilation(np.eye(1, size, y, dtype=bool).repeat(size, axis=0), iterations=0)
        domain = np.zeros((size, size), dtype=bool)
        domain[max(0, y - 1) : y + 2, 3 : size - 3] = True
        return {"field": field, "probability": probability, "start": path[0], "goal": path[-1], "domain": domain, "label": 1}
    if case == "curved_gap":
        amplitude = float(rng.uniform(2.5, 4.5))
        phase = float(rng.uniform(-0.3, 0.3))
        path = []
        for x in range(4, size - 4):
            t = (x - 4) / (size - 9)
            y = int(round(center + amplitude * math.sin(math.pi * (t + phase))))
            path.append((y, x))
        field, probability = _field_from_paths(size, [(path, visible, True)], scale=max(scale, 1.0), hyper=min(hyper, 0.9))
        for index, (py, px) in enumerate(path):
            if len(path) // 2 - 2 <= index <= len(path) // 2 + 2:
                active = index % 2
                field.membership[0, active, py, px] = weak
                probability[py, px] = weak
        centerline = np.zeros((size, size), dtype=bool)
        for point in path:
            centerline[point] = True
        domain = binary_dilation(centerline, iterations=1)
        return {"field": field, "probability": probability, "start": path[0], "goal": path[-1], "domain": domain, "label": 1}
    if case == "parallel_false_bridge":
        top, bottom = center - 4, center + 4
        upper = [(top, x) for x in range(4, size - 4)]
        lower = [(bottom, x) for x in range(4, size - 4)]
        vertical = [(y, center) for y in range(top, bottom + 1)]
        field, probability = _field_from_paths(
            size, [(upper, visible, False), (lower, visible, False), (vertical, weak, False)],
            scale=scale, hyper=hyper,
        )
        # The confusing ridge is explicitly horizontal despite its vertical
        # displacement: axis similarity alone accepts it, step support rejects it.
        field.orientation[0, 2, 0, top : bottom + 1, center] = 1.0
        field.orientation[0, 2, 1, top : bottom + 1, center] = 0.0
        domain = np.zeros((size, size), dtype=bool)
        domain[top : bottom + 1, center - 1 : center + 2] = True
        return {"field": field, "probability": probability, "start": (top, center), "goal": (bottom, center), "domain": domain, "label": 0}
    raise ValueError(case)


def _branch_fixture(case: str, seed: int, size: int = 33) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    center = size // 2
    jitter = int(rng.integers(-1, 2))
    center_point = (center + jitter, center)
    if case == "x_crossing":
        endpoints = [(center_point[0], 4), (center_point[0], size - 5), (4, center), (size - 5, center)]
    elif case == "t_junction":
        endpoints = [(center_point[0], 4), (center_point[0], size - 5), (4, center)]
    elif case == "y_junction":
        endpoints = [(size - 5, center), (4, 5), (4, size - 6)]
    else:
        raise ValueError(case)
    paths = []
    for branch_index, endpoint in enumerate(endpoints):
        length = max(abs(endpoint[0] - center_point[0]), abs(endpoint[1] - center_point[1]))
        path = []
        for step in range(length + 1):
            fraction = step / max(length, 1)
            y = int(round(center_point[0] + fraction * (endpoint[0] - center_point[0])))
            x = int(round(center_point[1] + fraction * (endpoint[1] - center_point[1])))
            if not path or path[-1] != (y, x):
                path.append((y, x))
        paths.append((path, 0.95, bool(branch_index % 2)))
    field, probability = _field_from_paths(size, paths, scale=0.95, hyper=0.85)
    # Explicitly install all incident axes at the junction.
    for mode, endpoint in enumerate(endpoints):
        theta = math.atan2(endpoint[0] - center_point[0], endpoint[1] - center_point[1])
        field.membership[0, mode, center_point[0], center_point[1]] = 0.95
        field.orientation[0, mode, 0, center_point[0], center_point[1]] = math.cos(2 * theta)
        field.orientation[0, mode, 1, center_point[0], center_point[1]] = math.sin(2 * theta)
    expected = []
    for endpoint in endpoints:
        dy = int(np.sign(endpoint[0] - center_point[0]))
        dx = int(np.sign(endpoint[1] - center_point[1]))
        expected.append((LOCAL8_OFFSETS.index((dx, dy)), center_point))
    return {"field": field, "probability": probability, "expected_edges": expected, "case": case}


def _probability_relation(probability: np.ndarray) -> np.ndarray:
    source = torch.from_numpy(probability).to(torch.float64).view(1, 1, *probability.shape)
    rows = []
    for dx, dy in LOCAL8_OFFSETS:
        neighbor, valid = _shift_neighbor(source, dx, dy)
        rows.append(torch.sqrt((source * neighbor).clamp_min(0.0)) * valid)
    return torch.cat(rows, dim=1)[0].numpy().astype(np.float32)


def _simple_axis_relation(field: ANZA2FieldOutput, probability: np.ndarray) -> np.ndarray:
    base = torch.from_numpy(_probability_relation(probability)).to(torch.float64).unsqueeze(0)
    active = field.membership.argmax(dim=1, keepdim=True)
    gather = active.unsqueeze(2).expand(-1, -1, 2, -1, -1)
    principal = torch.gather(field.orientation, 1, gather).squeeze(1)
    rows = []
    for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
        neighbor, valid = _shift_neighbor(principal, dx, dy)
        similarity = ((principal * neighbor).sum(dim=1) + 1.0) / 2.0
        rows.append(base[:, channel] * similarity.clamp(0.0, 1.0) * valid.squeeze(1))
    return torch.stack(rows, dim=1)[0].numpy().astype(np.float32)


def _legacy_relation(field: ANZA2FieldOutput) -> np.ndarray:
    normalized_mu = field.membership / field.membership.sum(dim=1, keepdim=True).clamp_min(1e-8)
    raw_rows = []
    for dx, dy in LOCAL8_OFFSETS:
        neighbor, valid = shift_field(field, dx, dy)
        neighbor_mu = neighbor.membership / neighbor.membership.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gp = directed_geometry(field, (dx, dy))
        gq = directed_geometry(neighbor, (-dx, -dy))
        raw = (normalized_mu * neighbor_mu * torch.sqrt(gp * gq)).sum(dim=1) * valid.squeeze(1)
        raw_rows.append(raw)
    raw = torch.stack(raw_rows, dim=1)
    return (raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8))[0].numpy().astype(np.float32)


def _relations(fixture: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "probability_only": _probability_relation(fixture["probability"]),
        "simple_axis_similarity": _simple_axis_relation(fixture["field"], fixture["probability"]),
        "legacy_global_normalized": _legacy_relation(fixture["field"]),
        "anza2_absolute": ANZA2StructuralAffinity()(fixture["field"])[0].detach().cpu().numpy().astype(np.float32),
    }


def _choose_threshold(rows: list[dict[str, Any]], method: str) -> tuple[float, dict[str, float | int]]:
    scores = np.array([row[method] for row in rows], dtype=np.float64)
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    best = None
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        tpr = tp / max(positives, 1)
        fpr = fp / max(negatives, 1)
        if fpr <= TARGET_FPR:
            candidate = (tpr, -fpr, float(threshold), tp, fp, positives, negatives)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise AssertionError("no valid operating threshold")
    return best[2], {
        "tpr": best[0], "fpr": -best[1], "tp": best[3], "fp": best[4],
        "positives": best[5], "negatives": best[6],
    }


def _path_rows(split: str, *, seed_base: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for stratum_index, case in enumerate(PATH_STRATA):
        for index in range(SAMPLES_PER_STRATUM):
            seed = (SPLIT_SEED_BASE[split] if seed_base is None else int(seed_base)) + stratum_index * 10_000 + index
            fixture = _path_fixture(case, seed)
            row: dict[str, Any] = {"split": split, "case": case, "index": index, "seed": seed, "label": fixture["label"]}
            for method, relation in _relations(fixture).items():
                score, _path = domain_restricted_widest_path(
                    relation, fixture["start"], fixture["goal"], domain=fixture["domain"], offsets=LOCAL8_OFFSETS
                )
                row[method] = score
            rows.append(row)
    return rows


def _branch_rows(split: str, *, seed_base: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for stratum_index, case in enumerate(BRANCH_STRATA):
        for index in range(SAMPLES_PER_STRATUM):
            seed = (SPLIT_SEED_BASE[split] if seed_base is None else int(seed_base)) + 100_000 + stratum_index * 10_000 + index
            fixture = _branch_fixture(case, seed)
            relations = _relations(fixture)
            for branch, (channel, point) in enumerate(fixture["expected_edges"]):
                row = {"split": split, "case": case, "index": index, "seed": seed, "branch": branch}
                for method, relation in relations.items():
                    row[method] = float(relation[channel, point[0], point[1]])
                rows.append(row)
    return rows


def _bootstrap_delta(confirm_rows: list[dict[str, Any]], threshold: dict[str, float], control: str) -> list[float]:
    positives = [row for row in confirm_rows if row["label"] == 1]
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in positives:
        by_seed.setdefault(int(row["index"]), []).append(row)
    keys = sorted(by_seed)
    rng = np.random.default_rng(20260818)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        anza_hits = []
        control_hits = []
        for key in sample_keys:
            for row in by_seed[int(key)]:
                anza_hits.append(row["anza2_absolute"] >= threshold["anza2_absolute"])
                control_hits.append(row[control] >= threshold[control])
        deltas.append(float(np.mean(anza_hits) - np.mean(control_hits)))
    return deltas


def run_phase2(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    protocol_hash = _canonical_hash(protocol)
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    development = _path_rows("development")
    development_branches = _branch_rows("development")
    thresholds = {}
    development_metrics = {}
    for method in METHODS:
        threshold, metric = _choose_threshold(development, method)
        thresholds[method] = threshold
        development_metrics[method] = metric
    freeze = {
        "status": "PHASE2_THRESHOLDS_FROZEN",
        "protocol_sha256": protocol_hash,
        "thresholds": thresholds,
        "selection": "maximize development TPR subject to development FPR <= 0.05",
        "confirm_rows_opened": 0,
        "expert_data_accessed": False,
    }
    freeze["freeze_sha256"] = _canonical_hash(freeze)
    (output_root / "threshold_freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    confirm = _path_rows("confirm")
    confirm_branches = _branch_rows("confirm")
    per_method = {}
    for method in METHODS:
        scores = np.array([row[method] for row in confirm])
        labels = np.array([row["label"] for row in confirm])
        predicted = scores >= thresholds[method]
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        branch_scores = np.array([row[method] for row in confirm_branches])
        branch_recall = float(np.mean(branch_scores >= thresholds[method]))
        per_method[method] = {
            "threshold": thresholds[method],
            "positive_path_tpr": tp / positives,
            "parallel_false_bridge_fpr": fp / negatives,
            "tp": tp, "fp": fp, "positives": positives, "negatives": negatives,
            "junction_branch_recall": branch_recall,
            "branch_hits": int(np.sum(branch_scores >= thresholds[method])),
            "branch_total": int(branch_scores.size),
        }
    controls = [name for name in METHODS if name != "anza2_absolute"]
    strongest = max(controls, key=lambda name: per_method[name]["positive_path_tpr"])
    deltas = _bootstrap_delta(confirm, thresholds, strongest)
    point_delta = per_method["anza2_absolute"]["positive_path_tpr"] - per_method[strongest]["positive_path_tpr"]
    ci = [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
    anza = per_method["anza2_absolute"]
    gate = bool(
        anza["positive_path_tpr"] >= 0.90
        and anza["parallel_false_bridge_fpr"] <= 0.05
        and anza["junction_branch_recall"] >= 0.95
        and point_delta >= protocol["practical_delta_vs_strongest_control"]
        and ci[0] > 0.0
    )
    metrics = {
        "status": "PHASE2_ZERO_TRAIN_MECHANISM_PASS" if gate else "STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE",
        "protocol_sha256": protocol_hash,
        "methods": per_method,
        "strongest_control": strongest,
        "anza2_tpr_delta": point_delta,
        "anza2_tpr_delta_ci95": ci,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "phase2_gate_pass": gate,
        "training_performed": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "claim_boundary": "Oracle weak-support field mechanism only; not learned image inference and not a CRACKS result.",
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (output_root / "bootstrap.json").write_text(json.dumps({
        "unit": "paired synthetic index across positive strata",
        "resamples": BOOTSTRAP_RESAMPLES,
        "strongest_control": strongest,
        "delta": point_delta,
        "ci95": ci,
    }, indent=2, sort_keys=True) + "\n")
    with (output_root / "per_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confirm[0]))
        writer.writeheader(); writer.writerows(confirm)
    with (output_root / "per_branch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confirm_branches[0]))
        writer.writeheader(); writer.writerows(confirm_branches)
    _write_curve(output_root, confirm, per_method)
    return metrics


def _write_curve(output_root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_rows = []
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    labels = np.array([row["label"] for row in rows])
    for method in METHODS:
        scores = np.array([row[method] for row in rows])
        points = []
        for threshold in np.r_[np.inf, np.unique(scores)[::-1], -np.inf]:
            predicted = scores >= threshold
            tpr = float(np.sum(predicted & (labels == 1)) / np.sum(labels == 1))
            fpr = float(np.sum(predicted & (labels == 0)) / np.sum(labels == 0))
            curve_rows.append({"method": method, "threshold": threshold, "tpr": tpr, "fpr": fpr})
            if fpr <= 0.10:
                points.append((fpr, tpr))
        points.sort()
        axis.plot([p[0] for p in points], [p[1] for p in points], label=method)
    axis.axvline(TARGET_FPR, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 0.10); axis.set_ylim(0, 1.02)
    axis.set_xlabel("False bridge rate"); axis.set_ylabel("Positive path recovery")
    axis.legend(fontsize=7, loc="lower right")
    fig.savefig(output_root / "low_false_bridge_frontier.png", dpi=300)
    fig.savefig(output_root / "low_false_bridge_frontier.svg")
    plt.close(fig)
    with (output_root / "operating_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "threshold", "tpr", "fpr"))
        writer.writeheader(); writer.writerows(curve_rows)
