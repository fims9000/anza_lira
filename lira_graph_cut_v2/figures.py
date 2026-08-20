"""Frozen benchmark figures, including explicit unavailable panels after STOP."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from lira_final.dense.ensemble import load_probability
from lira_graph_cut_v2.benchmark import _interval, recover_split, split_manifest
from lira_graph_cut_v2.graph_cut import rasterize
from lira_graph_cut_v2.protocol import DENSE_CACHE, RESULT_ROOT
from lira_intervention.candidate import masked_probability as v1_masked_probability
from lira_intervention.data import load_jsonl as load_v1_cases


def _crop(points: np.ndarray, shape: tuple[int, int], padding: int = 24) -> tuple[slice, slice]:
    low = np.floor(points.min(axis=0)).astype(int) - padding
    high = np.ceil(points.max(axis=0)).astype(int) + padding + 1
    return slice(max(0, low[0]), min(shape[0], high[0])), slice(max(0, low[1]), min(shape[1], high[1]))


def _save_support(path: Path, original: np.ndarray, cut: np.ndarray, tube: np.ndarray, points: np.ndarray, title: str) -> None:
    roi = _crop(points, original.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, value, name in zip(axes, (original, cut, tube), ("P >= 0.12 before", "P >= 0.12 after", "erased tube")):
        ax.imshow(value[roi], cmap="gray", interpolation="nearest")
        ax.set_title(name); ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off"); ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def generate() -> dict[str, object]:
    output = RESULT_ROOT / "benchmark/figures"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []

    v1_cases = {case.case_id: case for case in load_v1_cases(Path("results/lira_intervention_final/i1_benchmark/ig_development.jsonl"))}
    with Path("results/lira_intervention_final/i2_candidate/development_diagnostics.csv").open() as handle:
        v1_row = next(row for row in csv.DictReader(handle) if row["contexts_still_hard_connected_after_intervention"] == "1")
    v1 = v1_cases[v1_row["case_id"]]
    probability = load_probability(DENSE_CACHE, v1.section_id)
    v1_cut, v1_tube = v1_masked_probability(probability, v1)
    _save_support(output / "fig01_v1_failed_disconnect.png", probability >= 0.12, v1_cut >= 0.12, v1_tube, np.asarray(v1.hidden_yx), "V1: fixed 3-px treatment remains connected")
    artifacts.append({"figure": 1, "status": "GENERATED", "path": "fig01_v1_failed_disconnect.png"})

    manifest = split_manifest()
    recovered = recover_split("gc_development", manifest)
    traces = {trace.trace_id: trace for local in recovered.values() for trace in local}
    eligibility_path = RESULT_ROOT / "benchmark/gc_development_eligibility.csv"
    with eligibility_path.open() as handle:
        eligibility = list(csv.DictReader(handle))
    context_row = next(row for row in eligibility if row["status"] == "INVALID_CONTEXT_DESTROYED")
    trace = traces[context_row["trace_id"]]
    points, start, end = _interval(trace, "gc_development", int(context_row["gap_length_px"]))
    probability = load_probability(DENSE_CACHE, int(context_row["section_id"]))
    hidden = points[start : end + 1]
    seed = rasterize(hidden, probability.shape, 0)
    tube = distance_transform_edt(~seed) <= float(context_row["radius_px"])
    cut = (probability >= 0.12) & ~tube
    _save_support(output / "fig02_minimal_disconnect_context_loss.png", probability >= 0.12, cut, tube, hidden, f"V2: minimal disconnect r={context_row['radius_px']} destroys required context")
    artifacts.append({"figure": 2, "status": "GENERATED_INVALID_CONTEXT_EXAMPLE", "path": "fig02_minimal_disconnect_context_loss.png"})

    radius_counts = {}
    for row in eligibility:
        if row["radius_px"]:
            radius_counts[int(row["radius_px"])] = radius_counts.get(int(row["radius_px"]), 0) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    radii = [3, 5, 7, 9, 11, 13, 15]
    ax.bar([str(value) for value in radii], [radius_counts.get(value, 0) for value in radii], color="#355C7D")
    ax.set_xlabel("minimal disconnect radius (px)"); ax.set_ylabel("pre-treatment eligible cases")
    ax.set_title("Development minimal-radius distribution before exclusions")
    fig.tight_layout(); fig.savefig(output / "fig03_radius_distribution.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    artifacts.append({"figure": 3, "status": "GENERATED", "path": "fig03_radius_distribution.png"})

    collateral_row = next(row for row in eligibility if row["status"] == "INVALID_COLLATERAL_TRACE")
    trace = traces[collateral_row["trace_id"]]
    points, start, end = _interval(trace, "gc_development", int(collateral_row["gap_length_px"]))
    probability = load_probability(DENSE_CACHE, int(collateral_row["section_id"]))
    hidden = points[start : end + 1]; seed = rasterize(hidden, probability.shape, 0)
    tube = distance_transform_edt(~seed) <= float(collateral_row["radius_px"])
    cut = (probability >= 0.12) & ~tube
    _save_support(output / "fig04_collateral_exclusion.png", probability >= 0.12, cut, tube, hidden, f"Excluded collateral trace fraction={float(collateral_row['collateral_fraction']):.3f}")
    artifacts.append({"figure": 4, "status": "GENERATED", "path": "fig04_collateral_exclusion.png"})

    _placeholder(output / "fig05_valid_ports_unavailable.png", "Valid ports unavailable", "No intervention survived the frozen retention/context rules; creating a valid-port example would fabricate evidence.")
    artifacts.append({"figure": 5, "status": "UNAVAILABLE_AFTER_RETENTION_STOP", "path": "fig05_valid_ports_unavailable.png"})
    _placeholder(output / "fig06_candidate_recall_unavailable.png", "Candidate recall unavailable", "Frozen SBPP was correctly kept locked after STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE.")
    artifacts.append({"figure": 6, "status": "UNAVAILABLE_CANDIDATE_NOT_OPENED", "path": "fig06_candidate_recall_unavailable.png"})
    result = {"status": "COMPLETE_WITH_LOCKED_PANELS", "artifacts": artifacts, "candidate_opened": False, "confirm_opened": False, "expert_accessed": False}
    (output / "figure_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

