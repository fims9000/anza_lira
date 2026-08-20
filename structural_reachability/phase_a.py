"""Frozen, zero-training CRACKS Phase-A causal geometry probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cracks_experiment.partial_label_training import _model, load_t1_checkpoint, t1_matrix
from cracks_experiment.training import NORMALIZATION
from cracks_experiment.validation import _sha256, tiled_probability
from datasets.cracks import load_section_image
from structural_reachability.geometry import (
    axial_mean,
    compute_axial_consistency,
    compute_directed_anisotropic_factor,
    compute_fuzzy_compatibility,
    compute_scale_compatibility,
    log_geometric_mean,
    symmetrize_affinity,
)
from structural_reachability.metrics import evaluate_low_fpr_curve, section_paired_bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "structural_reachability" / "phase_a"
PAIR_ROOT = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_pairs"
T1_ROOT = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_t1"
SEEDS = (41, 42, 43)
RELATIONS = (
    "A0_probability_only",
    "A1_rgb_similarity",
    "A2_geometry_G_theta",
    "A3_geometry_plus_fuzzy",
    "A4_full_geometry",
)
FPR_MAX = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260818
RGB_SIMILARITY_SCALE = 0.10
MODEL_CONFUSER_COUNT = 20


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _line_pixels(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    y0, x0 = first
    y1, x1 = second
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    result = []
    while True:
        result.append((y0, x0))
        if (y0, x0) == (y1, x1):
            return tuple(result)
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _frozen_run(seed: int) -> tuple[Any, Path]:
    spec = next(row for row in t1_matrix() if row.model == "anza_v1" and row.seed == int(seed))
    run_dir = T1_ROOT / f"{spec.run_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if (
        status.get("status") != "COMPLETE"
        or status.get("expert_data_accessed") is not False
        or status.get("expert_scores_used") is not False
    ):
        raise PermissionError(f"T1 ANZA seed {seed} is not complete and expert-locked")
    return spec, run_dir / "checkpoint-last.pt"


def phase_a_protocol() -> dict[str, Any]:
    pair_manifest = PAIR_ROOT / "manifest.json"
    specification = Path("/home/lebedeffson/Downloads/ANZA_STRUCTURAL_REACHABILITY_TZ_20260818.md")
    checkpoints = {str(seed): _sha256(_frozen_run(seed)[1]) for seed in SEEDS}
    sources = [
        PROJECT_ROOT / "structural_reachability" / "geometry.py",
        PROJECT_ROOT / "structural_reachability" / "metrics.py",
        Path(__file__),
    ]
    return {
        "version": "anza_structural_reachability_phase_a_v1",
        "phase": "A_ZERO_TRAINING_GEOMETRY_PROBE",
        "research_question": "Does frozen ANZA axial/fuzzy geometry add low-FPR structural discrimination beyond foreground probability?",
        "data": "existing balanced descriptor-matched CRACKS crowd validation pairs; expert forbidden",
        "candidate_source_sha256": _sha256(pair_manifest),
        "specification_sha256": _sha256(specification),
        "segmentation_checkpoint_sha256": checkpoints,
        "seeds": list(SEEDS),
        "relations": list(RELATIONS),
        "path_reduction": "minimum local edge score along the same Bresenham candidate corridor",
        "component_fusion": "equal-weight geometric mean in log space",
        "symmetrization": "geometric_mean selected before scores",
        "rgb_similarity_scale": RGB_SIMILARITY_SCALE,
        "fuzzy_membership_semantics": "legacy frozen checkpoint categorical softmax; not relabeled as independent fuzzy degrees",
        "primary_metrics": ["TPR@FPR<=0.05", "normalized partial AUC FPR[0,0.05]", "matched ranking probability"],
        "secondary_metric": "AUROC",
        "bootstrap": {
            "unit": "section_id",
            "resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": BOOTSTRAP_SEED,
            "seed_aggregation": "metric mean within bootstrap draw",
        },
        "meaningful_effect": "delta_A=max(0.05,2*sample_SD(A0 seed-specific TPR@FPR<=0.05))",
        "gate": "A4-A0 TPR delta >= delta_A and paired section-bootstrap CI95 lower bound > 0",
        "model_generated_confusers": f"secondary fixed top-{MODEL_CONFUSER_COUNT} validation negatives by the already frozen prior classifier; never used for primary gate",
        "training_performed": False,
        "expert_data_accessed": False,
        "expert_scores_used": False,
        "phase_b_locked_until_gate_a_pass": True,
        "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in sources},
    }


@torch.inference_mode()
def extract_anza_geometry(model: torch.nn.Module, image_hwc: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the actual frozen legacy ANZA fields without changing the checkpoint."""

    raw = torch.from_numpy(np.asarray(image_hwc, dtype=np.float32).transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1)
    normalized = F.pad((raw - mean) / std, (0, 3, 0, 1))
    probability = tiled_probability(model, normalized).numpy()[:255, :701]
    device = next(model.parameters()).device
    spatial = model.enc1.spatial
    if spatial.geometry_conv is None or spatial.cfg.geometry_mode != "local_hyperbolic":
        raise TypeError("Phase A requires frozen local-hyperbolic legacy ANZA geometry")
    inputs = normalized.unsqueeze(0).to(device)
    membership = torch.softmax(
        spatial.gate_conv(inputs) / float(spatial.cfg.fuzzy_temperature), dim=1
    )
    theta, raw_base, raw_hyper = torch.chunk(spatial.geometry_conv(inputs), 3, dim=1)
    base = F.softplus(raw_base) + 1e-4
    hyper = F.softplus(raw_hyper).clamp_max(float(spatial.cfg.max_hyperbolicity))
    sigma_parallel = base * torch.exp(hyper)
    sigma_perpendicular = base * torch.exp(-hyper)
    fields = {
        "image": np.asarray(image_hwc, dtype=np.float32).transpose(2, 0, 1),
        "probability": probability.astype(np.float32),
        "membership": membership[0, :, :255, :701].cpu().numpy().astype(np.float32),
        "theta": theta[0, :, :255, :701].cpu().numpy().astype(np.float32),
        "sigma_parallel": sigma_parallel[0, :, :255, :701].cpu().numpy().astype(np.float32),
        "sigma_perpendicular": sigma_perpendicular[0, :, :255, :701].cpu().numpy().astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError("non-finite frozen ANZA field")
    return fields


def score_candidate_path(fields: dict[str, np.ndarray], path: tuple[tuple[int, int], ...]) -> dict[str, float]:
    """Score one fixed corridor; every relation sees exactly the same local edges."""

    if len(path) < 2:
        raise ValueError("candidate path must contain at least one edge")
    edge_rows: list[dict[str, float]] = []
    for (py, px), (qy, qx) in zip(path[:-1], path[1:]):
        mu_p = fields["membership"][:, py, px]
        mu_q = fields["membership"][:, qy, qx]
        theta_p = axial_mean(fields["theta"][:, py, px], mu_p)
        theta_q = axial_mean(fields["theta"][:, qy, qx], mu_q)
        sp_p = float(np.average(fields["sigma_parallel"][:, py, px], weights=mu_p))
        sp_q = float(np.average(fields["sigma_parallel"][:, qy, qx], weights=mu_q))
        st_p = float(np.average(fields["sigma_perpendicular"][:, py, px], weights=mu_p))
        st_q = float(np.average(fields["sigma_perpendicular"][:, qy, qx], weights=mu_q))
        dy, dx = qy - py, qx - px
        g_forward = float(compute_directed_anisotropic_factor(theta_p, sp_p, st_p, dy, dx))
        g_reverse = float(compute_directed_anisotropic_factor(theta_q, sp_q, st_q, -dy, -dx))
        geometry = float(symmetrize_affinity(g_forward, g_reverse))
        axial = float(compute_axial_consistency(theta_p, theta_q))
        fuzzy = float(compute_fuzzy_compatibility(mu_p, mu_q))
        scale = float(compute_scale_compatibility(sp_p, sp_q, st_p, st_q))
        probability = math.sqrt(
            float(np.clip(fields["probability"][py, px], 0.0, 1.0))
            * float(np.clip(fields["probability"][qy, qx], 0.0, 1.0))
        )
        rgb_delta = fields["image"][:, py, px] - fields["image"][:, qy, qx]
        image_similarity = math.exp(
            -float(np.mean(rgb_delta * rgb_delta)) / (2.0 * RGB_SIMILARITY_SCALE**2)
        )
        edge_rows.append({
            "probability": probability,
            "image_similarity": image_similarity,
            "G": geometry,
            "c_theta": axial,
            "c_mu": fuzzy,
            "c_sigma": scale,
            "A0_probability_only": probability,
            "A1_rgb_similarity": image_similarity,
            "A2_geometry_G_theta": float(log_geometric_mean(np.asarray([geometry, axial]))),
            "A3_geometry_plus_fuzzy": float(log_geometric_mean(np.asarray([geometry, axial, fuzzy]))),
            "A4_full_geometry": float(log_geometric_mean(np.asarray([geometry, axial, fuzzy, scale]))),
        })
    result = {key: float(min(row[key] for row in edge_rows)) for key in edge_rows[0]}
    result["edge_count"] = float(len(edge_rows))
    if not all(np.isfinite(value) and 0.0 <= value <= max(1.0, len(edge_rows)) for value in result.values()):
        raise AssertionError("invalid candidate relation score")
    return result


def _load_model(seed: int, device: str) -> tuple[torch.nn.Module, Path]:
    spec, checkpoint = _frozen_run(seed)
    model = _model(spec).to(torch.device(device))
    load_t1_checkpoint(checkpoint, spec, model)
    model.eval()
    return model, checkpoint


def _confuser_pair_ids() -> set[int]:
    rows = list(csv.DictReader((PAIR_ROOT / "scores.csv").open()))
    validation = [row for row in rows if row["split"] == "validation"]
    selected = sorted(validation, key=lambda row: float(row["negative_score"]), reverse=True)
    return {int(row["pair_id"]) for row in selected[:MODEL_CONFUSER_COUNT]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_rows(device: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = [row for row in manifest["rows"] if row["split"] == "validation"]
    confusers = _confuser_pair_ids()
    image_root = PROJECT_ROOT / "data" / "cracks" / "images"
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, _checkpoint = _load_model(seed, device)
        by_section: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for pair_id, row in enumerate(source_rows):
            by_section.setdefault(int(row["section_id"]), []).append((pair_id, row))
        for position, (section_id, section_rows) in enumerate(sorted(by_section.items()), start=1):
            image = load_section_image(image_root / f"section_{section_id:03d}.png")
            fields = extract_anza_geometry(model, image)
            for pair_id, row in section_rows:
                for label, prefix in ((1, "positive"), (0, "negative")):
                    first = tuple(int(value) for value in row[f"{prefix}_first"])
                    second = tuple(int(value) for value in row[f"{prefix}_second"])
                    components = score_candidate_path(fields, _line_pixels(first, second))
                    common = {
                        "split": "validation",
                        "section_id": section_id,
                        "annotator": row["annotator"],
                        "pair_id": pair_id,
                        "seed": seed,
                        "label": label,
                        "source_kind": "same_trace_internal_gap" if label else "different_connected_traces",
                        "model_generated_confuser": bool(not label and pair_id in confusers),
                        "first_y": first[0], "first_x": first[1],
                        "second_y": second[0], "second_x": second[1],
                        **{key: value for key, value in components.items() if key not in RELATIONS},
                    }
                    for relation in RELATIONS:
                        output.append({**common, "relation": relation, "score": components[relation]})
            if position % 20 == 0 or position == len(by_section):
                print(
                    f"phase=reachability_A seed={seed} section={position}/{len(by_section)} "
                    "training=NO expert=LOCKED status=RUNNING",
                    flush=True,
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def _metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    curves: list[dict[str, Any]] = []
    for relation in RELATIONS:
        seed_metrics = []
        for seed in SEEDS:
            selected = [row for row in rows if row["relation"] == relation and int(row["seed"]) == seed]
            result = evaluate_low_fpr_curve(
                np.asarray([row["label"] for row in selected]),
                np.asarray([row["score"] for row in selected]),
                pair_ids=np.asarray([row["pair_id"] for row in selected]),
                fpr_max=FPR_MAX,
            )
            seed_metrics.append({key: value for key, value in result.items() if key != "curve"})
            curves.extend({"relation": relation, "seed": seed, **point} for point in result["curve"])
        summaries[relation] = {
            "per_seed": {str(seed): metric for seed, metric in zip(SEEDS, seed_metrics)},
            "seed_mean": {
                key: float(np.mean([metric[key] for metric in seed_metrics]))
                for key in (
                    "tpr_at_fpr_max", "achieved_fpr", "low_fpr_partial_auc_normalized",
                    "auroc_secondary", "matched_ranking_probability",
                )
            },
        }
    return summaries, curves


def _figure(curves: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    colors = {RELATIONS[0]: "#4C78A8", RELATIONS[1]: "#A0A0A0", RELATIONS[2]: "#F58518", RELATIONS[3]: "#54A24B", RELATIONS[4]: "#B279A2"}
    grid = np.linspace(0.0, FPR_MAX, 101)
    for relation in RELATIONS:
        interpolated = []
        for seed in SEEDS:
            selected = [row for row in curves if row["relation"] == relation and int(row["seed"]) == seed]
            x = np.asarray([row["fpr"] for row in selected])
            y = np.asarray([row["tpr"] for row in selected])
            order = np.argsort(x)
            interpolated.append(np.interp(grid, x[order], y[order]))
        mean = np.mean(interpolated, axis=0)
        ax.plot(grid, mean, label=relation.split("_", 1)[0], color=colors[relation], linewidth=2)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, FPR_MAX), ylim=(0, 1), title="Frozen CRACKS Phase A: low-FPR structural discrimination")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(output / "fig_low_fpr_geometry.png", dpi=220)
    fig.savefig(output / "fig_low_fpr_geometry.svg")
    plt.close(fig)


def run_phase_a(*, device: str | None = None) -> dict[str, Any]:
    """Run the single predeclared Phase-A probe; never train or access expert data."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_ROOT / "metrics.json"
    protocol = phase_a_protocol()
    protocol_hash = _canonical_hash(protocol)
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("protocol_sha256") == protocol_hash and existing.get("expert_data_accessed") is False:
            return {**existing, "action": "SKIP"}
        raise ValueError("existing Phase-A artifact has provenance drift")
    manifest = json.loads((PAIR_ROOT / "manifest.json").read_text())
    if manifest.get("expert_data_accessed") is not False or not manifest.get("section_disjoint"):
        raise PermissionError("source pair manifest is not expert-locked and section-disjoint")
    validation_rows = [row for row in manifest["rows"] if row["split"] == "validation"]
    if len(validation_rows) != 120 or len({row["section_id"] for row in validation_rows}) != 73:
        raise ValueError("frozen validation pair population drift")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _build_rows(selected_device, manifest)
    expected = len(validation_rows) * 2 * len(SEEDS) * len(RELATIONS)
    if len(rows) != expected:
        raise AssertionError(f"Phase-A row count drift: {len(rows)} != {expected}")
    summaries, curves = _metrics(rows)
    baseline_seed_tpr = [
        summaries[RELATIONS[0]]["per_seed"][str(seed)]["tpr_at_fpr_max"] for seed in SEEDS
    ]
    baseline_seed_sd = float(np.std(baseline_seed_tpr, ddof=1))
    delta_a = max(0.05, 2.0 * baseline_seed_sd)
    bootstrap_tpr = section_paired_bootstrap(
        rows,
        candidate_relation=RELATIONS[-1],
        baseline_relation=RELATIONS[0],
        seeds=SEEDS,
        metric="tpr_at_fpr_max",
        resamples=BOOTSTRAP_RESAMPLES,
        random_seed=BOOTSTRAP_SEED,
        fpr_max=FPR_MAX,
    )
    bootstrap_pauc = section_paired_bootstrap(
        rows,
        candidate_relation=RELATIONS[-1],
        baseline_relation=RELATIONS[0],
        seeds=SEEDS,
        metric="low_fpr_partial_auc_normalized",
        resamples=BOOTSTRAP_RESAMPLES,
        random_seed=BOOTSTRAP_SEED,
        fpr_max=FPR_MAX,
    )
    gate_checks = {
        "tpr_delta_meaningful": bootstrap_tpr["point_delta"] >= delta_a,
        "tpr_delta_ci_low_positive": bootstrap_tpr["ci95"][0] > 0.0,
        "low_fpr_partial_auc_improved": bootstrap_pauc["point_delta"] > 0.0,
    }
    passed = all(gate_checks.values())
    status = "PHASE_A_PASS" if passed else "STOP_ARCHITECTURAL_ANZA_NO_CAUSAL_GEOMETRY_GAIN"
    protocol_path = OUTPUT_ROOT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "protocol_hash.txt").write_text(protocol_hash + "\n")
    _write_csv(OUTPUT_ROOT / "per_candidate.csv", rows)
    _write_csv(OUTPUT_ROOT / "edge_features.csv", rows)
    _write_csv(OUTPUT_ROOT / "operating_curve.csv", curves)
    split_manifest = {
        "source_manifest_sha256": _sha256(PAIR_ROOT / "manifest.json"),
        "train_section_ids": manifest["train_section_ids"],
        "validation_section_ids": manifest["validation_section_ids"],
        "validation_pair_count": len(validation_rows),
        "validation_section_count": len(set(row["section_id"] for row in validation_rows)),
        "section_disjoint": not bool(set(manifest["train_section_ids"]) & set(manifest["validation_section_ids"])),
        "expert_section_ids": [],
    }
    (OUTPUT_ROOT / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    checkpoint_manifest = {
        str(seed): {"path": str(_frozen_run(seed)[1]), "sha256": _sha256(_frozen_run(seed)[1])}
        for seed in SEEDS
    }
    (OUTPUT_ROOT / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n")
    access = {
        "images": [f"data/cracks/images/section_{value:03d}.png" for value in sorted(set(row["section_id"] for row in validation_rows))],
        "candidate_manifest": str(PAIR_ROOT / "manifest.json"),
        "crowd_annotations_opened_during_phase_a": [],
        "expert_paths": [],
        "expert_data_accessed": False,
    }
    (OUTPUT_ROOT / "data_access_log.json").write_text(json.dumps(access, indent=2, sort_keys=True) + "\n")
    bootstrap = {"tpr": bootstrap_tpr, "low_fpr_partial_auc": bootstrap_pauc}
    (OUTPUT_ROOT / "bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "calibration.json").write_text(json.dumps({
        "status": "NOT_APPLICABLE_ZERO_TRAINING_SCORE_PROBE",
        "threshold_selection": "metric-derived operating point at fixed FPR budget; not used to calibrate a deployable classifier",
    }, indent=2, sort_keys=True) + "\n")
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout
    (OUTPUT_ROOT / "code_state.json").write_text(json.dumps({
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "dirty": bool(git_status.strip()),
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "commit_created": False,
    }, indent=2, sort_keys=True) + "\n")
    (OUTPUT_ROOT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "torch": torch.__version__, "device": selected_device,
        "cuda_device": torch.cuda.get_device_name(0) if selected_device == "cuda" else None,
    }, indent=2, sort_keys=True) + "\n")
    result = {
        "status": status,
        "phase": "A",
        "protocol_sha256": protocol_hash,
        "data_used": "CRACKS crowd validation matched pairs only",
        "validation_pair_count": len(validation_rows),
        "validation_section_count": len(set(row["section_id"] for row in validation_rows)),
        "expert_data_accessed": False,
        "expert_scores_used": False,
        "training_performed": False,
        "relations": summaries,
        "baseline_seed_tpr": baseline_seed_tpr,
        "baseline_seed_sd": baseline_seed_sd,
        "delta_A": delta_a,
        "primary_comparison": bootstrap_tpr,
        "partial_auc_comparison": bootstrap_pauc,
        "gate_checks": gate_checks,
        "phase_b_authorized": passed,
        "root_cause": None if passed else "Frozen ANZA geometry did not satisfy the predeclared practical and statistical low-FPR gain gate over the same checkpoint foreground probability.",
        "action": "RUN",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    evidence = {
        "protocol_sha256": protocol_hash,
        "metrics_sha256": _sha256(result_path),
        "per_candidate_sha256": _sha256(OUTPUT_ROOT / "per_candidate.csv"),
        "source_pair_manifest_sha256": _sha256(PAIR_ROOT / "manifest.json"),
        "checkpoint_sha256": {str(seed): _sha256(_frozen_run(seed)[1]) for seed in SEEDS},
        "expert_data_accessed": False,
        "training_performed": False,
        "status": status,
    }
    (OUTPUT_ROOT / "EVIDENCE.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    report = "\n".join([
        "# PHASE A — ANZA Structural Reachability geometry probe",
        "",
        "```text",
        "PHASE: A — zero-training frozen geometry probe",
        f"PROTOCOL HASH: {protocol_hash}",
        "DATA USED: CRACKS non-expert frozen validation matched pairs (120 pairs, 73 sections)",
        "EXPERT ACCESSED: no",
        "TRAINING PERFORMED: no",
        "PRIMARY QUESTION: Does frozen ANZA geometry add low-FPR structural discrimination beyond foreground probability?",
        "PRIMARY METRIC: TPR at FPR <= 0.05",
        f"BASELINE: {summaries[RELATIONS[0]]['seed_mean']['tpr_at_fpr_max']:.6f}",
        f"ANZA RESULT: {summaries[RELATIONS[-1]]['seed_mean']['tpr_at_fpr_max']:.6f}",
        f"DELTA: {bootstrap_tpr['point_delta']:.6f}",
        f"95% CI: [{bootstrap_tpr['ci95'][0]:.6f}, {bootstrap_tpr['ci95'][1]:.6f}]",
        f"PRACTICAL GATE: delta >= {delta_a:.6f} and CI lower bound > 0",
        f"STATUS: {'PASS' if passed else 'FAIL'} — {status}",
        f"ROOT CAUSE: {result['root_cause'] or 'Gate A passed; causal geometry signal warrants Phase B.'}",
        f"NEXT ACTION: {'Phase B may be prepared under its frozen capacity-matched protocol.' if passed else 'STOP architecture work; do not run Phase B or any large training matrix.'}",
        "FILES: metrics.json, per_candidate.csv, operating_curve.csv, bootstrap_summary.json, EVIDENCE.json",
        "```",
        "",
        "The max–min/widest-path operator is treated as prior art (WPRF, arXiv:2607.07123), not as ANZA novelty.",
        "The legacy memberships are reported as the frozen checkpoint softmax outputs; this probe does not relabel them as independent fuzzy degrees.",
    ]) + "\n"
    (OUTPUT_ROOT / "PHASE_A_GEOMETRY_PROBE_REPORT.md").write_text(report)
    if not passed:
        (OUTPUT_ROOT / "FAILURE_ANALYSIS.md").write_text(
            "# Phase A gate failure\n\n" + result["root_cause"] + "\n\n"
            "Phase B, bounded CRACKS training, completion, and expert evaluation remain locked. "
            "No threshold, score fusion, symmetrization, candidate subset, or primary metric may be changed post hoc.\n"
        )
    _figure(curves, OUTPUT_ROOT)
    return result
