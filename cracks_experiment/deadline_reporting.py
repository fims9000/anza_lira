"""Claim-safe deadline tables, figures, and evidence from frozen result rows.

This module deliberately has no dataset or checkpoint imports.  It consumes
post-freeze machine-readable rows supplied by the caller, so importing it
cannot open expert annotations or trigger training/evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


MAIN_MODELS = ("unet", "deformable_unet", "anza_v1", "anza_v2b")
MAIN_SEEDS = (41, 42, 43)
MAIN_METRICS = (
    "dice",
    "cldice",
    "skeleton_f1_at_2px",
    "fragmentation",
    "trace_orientation_error_median_deg",
)
COMPARISON_METRICS = ("dice", "cldice", "skeleton_f1_at_2px")
ABLATION_RUNS = (
    "v2_full_s42",
    "v2_no_replay_s42",
    "v2_no_fuzzy_s42",
    "v2_no_directional_s42",
)
QUALITATIVE_SALT = "anza-v2-qual-20260817"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260817

DEADLINE_SCOPE_TEXT = """# ANZA-LIRA deadline scope

Setting A and the corrected synthetic evaluator are included. Settings B/C are
`NOT_RUN_DEADLINE_SCOPE` and are not used in submitted claims.

Main models use seeds 41/42/43 averaged within section before section-level
aggregation. Ablations are seed-42-only. Qualitative sections are selected by
SHA-256 rank with salt `anza-v2-qual-20260817` before model-error inspection.
"""


def _finite(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_row(row: Mapping[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            output[key] = None
        elif value in {"True", "False"}:
            output[key] = value == "True"
        else:
            try:
                output[key] = int(value)
            except ValueError:
                try:
                    output[key] = float(value)
                except ValueError:
                    output[key] = value
    return output


def deterministic_qualitative_sections(
    section_ids: Iterable[int], *, count: int = 3, salt: str = QUALITATIVE_SALT
) -> list[int]:
    """Select examples without using model errors or metric values."""

    unique = sorted({int(section_id) for section_id in section_ids})
    if len(unique) < count:
        raise ValueError(f"Need at least {count} distinct expert section IDs")
    ranked = sorted(
        unique,
        key=lambda section_id: (
            hashlib.sha256(f"{salt}:{section_id}".encode()).hexdigest(),
            section_id,
        ),
    )
    return ranked[:count]


def _bootstrap(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Section bootstrap needs a finite non-empty vector")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_sections": int(len(array)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "seismic_section",
    }


def _index_main_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        model = str(row["model"])
        if model not in MAIN_MODELS:
            continue
        seed = int(row["seed"])
        if seed not in MAIN_SEEDS:
            raise ValueError(f"Unexpected main-model seed: {model} seed {seed}")
        if "policy" in row and row["policy"] != "paper_like":
            raise ValueError("Deadline primary tables accept paper_like expert rows only")
        key = (model, seed, int(row["section_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate main result row: {key}")
        for metric in MAIN_METRICS:
            _finite(row[metric], field=f"{key}.{metric}")
        indexed[key] = row
    section_sets: dict[tuple[str, int], set[int]] = {
        (model, seed): {section for candidate, item_seed, section in indexed if candidate == model and item_seed == seed}
        for model in MAIN_MODELS
        for seed in MAIN_SEEDS
    }
    reference = section_sets[(MAIN_MODELS[0], MAIN_SEEDS[0])]
    if not reference:
        raise ValueError("Main Setting A rows are empty")
    for key, sections in section_sets.items():
        if sections != reference:
            raise ValueError(f"Main model/seed section mismatch for {key}")
    return indexed


def aggregate_main(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, float]]]]:
    """Average three seeds within section before macro aggregation."""

    indexed = _index_main_rows(rows)
    sections = sorted({key[2] for key in indexed})
    section_values: dict[str, dict[int, dict[str, float]]] = {}
    table: list[dict[str, Any]] = []
    for model_index, model in enumerate(MAIN_MODELS):
        model_sections: dict[int, dict[str, float]] = {}
        output: dict[str, Any] = {
            "model": model,
            "seed_count": 3,
            "seeds": "41|42|43",
            "n_sections": len(sections),
            "aggregation": "seed_mean_within_section_then_section_macro",
        }
        for metric_index, metric in enumerate(MAIN_METRICS):
            values = {
                section: float(
                    np.mean([
                        _finite(indexed[(model, seed, section)][metric], field=metric)
                        for seed in MAIN_SEEDS
                    ])
                )
                for section in sections
            }
            for section, value in values.items():
                model_sections.setdefault(section, {})[metric] = value
            summary = _bootstrap(list(values.values()), seed=BOOTSTRAP_SEED + model_index * 10 + metric_index)
            output[f"{metric}_mean"] = summary["mean"]
            output[f"{metric}_ci95_low"] = summary["ci95_low"]
            output[f"{metric}_ci95_high"] = summary["ci95_high"]
        section_values[model] = model_sections
        table.append(output)
    return table, section_values


def paired_main_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair on section and seed, then average the three deltas within section."""

    indexed = _index_main_rows(rows)
    sections = sorted({key[2] for key in indexed})
    output: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(MAIN_MODELS[:-1]):
        for metric_index, metric in enumerate(COMPARISON_METRICS):
            section_deltas = [
                float(np.mean([
                    _finite(indexed[("anza_v2b", seed, section)][metric], field=metric)
                    - _finite(indexed[(baseline, seed, section)][metric], field=metric)
                    for seed in MAIN_SEEDS
                ]))
                for section in sections
            ]
            summary = _bootstrap(
                section_deltas,
                seed=BOOTSTRAP_SEED + 100 + baseline_index * 10 + metric_index,
            )
            output.append(
                {
                    "comparison": f"anza_v2b_minus_{baseline}",
                    "first": "anza_v2b",
                    "second": baseline,
                    "metric": metric,
                    "delta_first_minus_second": summary["mean"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "ci_excludes_zero": bool(summary["ci95_low"] > 0 or summary["ci95_high"] < 0),
                    "n_sections": summary["n_sections"],
                    "seed_count": 3,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_unit": "seismic_section",
                    "pairing": "section+seed_delta_then_seed_mean_within_section",
                }
            )
    return output


def aggregate_ablations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize frozen seed-42 ablations and pair each against full V2 seed 42."""

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        run_id = "v2_full_s42" if str(row["run_id"]) == "v2_s42" else str(row["run_id"])
        if run_id not in ABLATION_RUNS:
            continue
        if int(row["seed"]) != 42:
            raise ValueError(f"Deadline ablation must be seed 42: {run_id}")
        if "policy" in row and row["policy"] != "paper_like":
            raise ValueError("Deadline ablation tables accept paper_like expert rows only")
        key = (run_id, int(row["section_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate ablation result row: {key}")
        for metric in MAIN_METRICS:
            _finite(row[metric], field=f"{key}.{metric}")
        indexed[key] = row
    sections_by_run = {
        run_id: {section for candidate, section in indexed if candidate == run_id}
        for run_id in ABLATION_RUNS
    }
    reference = sections_by_run[ABLATION_RUNS[0]]
    if not reference or any(sections != reference for sections in sections_by_run.values()):
        raise ValueError("Ablations require identical non-empty seed-42 section IDs")
    output: list[dict[str, Any]] = []
    for run_index, run_id in enumerate(ABLATION_RUNS):
        row: dict[str, Any] = {
            "run_id": run_id,
            "seed": 42,
            "seed_count": 1,
            "scope": "single-seed ablation",
            "reference": "v2_full_s42",
            "n_sections": len(reference),
        }
        for metric_index, metric in enumerate(MAIN_METRICS):
            values = [_finite(indexed[(run_id, section)][metric], field=metric) for section in sorted(reference)]
            summary = _bootstrap(values, seed=BOOTSTRAP_SEED + 200 + run_index * 10 + metric_index)
            row[f"{metric}_mean"] = summary["mean"]
            if run_id == "v2_full_s42":
                delta = [0.0] * len(values)
            else:
                delta = [
                    _finite(indexed[(run_id, section)][metric], field=metric)
                    - _finite(indexed[("v2_full_s42", section)][metric], field=metric)
                    for section in sorted(reference)
                ]
            delta_summary = _bootstrap(delta, seed=BOOTSTRAP_SEED + 300 + run_index * 10 + metric_index)
            row[f"{metric}_delta_vs_full"] = delta_summary["mean"]
            row[f"{metric}_delta_ci95_low"] = delta_summary["ci95_low"]
            row[f"{metric}_delta_ci95_high"] = delta_summary["ci95_high"]
        output.append(row)
    return output


def validate_synthetic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject model-labelled fake routing values in the corrected table."""

    if not rows:
        raise ValueError("synthetic_corrected rows are missing")
    route_fields = (
        "route_top1_hit",
        "route_true_probability_mass",
        "route_mrr",
        "route_average_precision",
        "route_entropy_normalized",
        "route_excess_over_chance",
        "topology_constrained_pairing_score",
    )
    normalized: list[dict[str, Any]] = []
    names = {str(row.get("model", row.get("candidate", ""))) for row in rows}
    if "geometry_only_minimum_angle_heuristic" not in names:
        raise ValueError("Corrected synthetic table needs the separate geometry-only heuristic row")
    for original in rows:
        row = dict(original)
        name = str(row.get("model", row.get("candidate", "")))
        if name in {"unet", "deformable_unet", "anza_v1"}:
            for field in route_fields:
                if row.get(field) not in {None, "", "NA", "N/A"}:
                    raise ValueError(f"Baseline route metric must be NA: {name}.{field}")
                row[field] = "NA"
        if name == "geometry_only_minimum_angle_heuristic":
            if row.get("uses_generator_branch_geometry") not in {True, "true", "True", 1, "1"}:
                raise ValueError("Geometry heuristic must disclose generator-geometry conditioning")
            row["result_family"] = "diagnostic_geometry_only"
        normalized.append(row)
    return normalized


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _main_figure(main: Sequence[Mapping[str, Any]], root: Path) -> None:
    labels = ("U-Net", "Deformable U-Net", "ANZA-LIRA v1", "Mode-resolved transport")
    metrics = ("dice", "cldice", "skeleton_f1_at_2px")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for axis, metric in zip(axes, metrics):
        means = np.asarray([float(row[f"{metric}_mean"]) for row in main])
        lows = np.asarray([float(row[f"{metric}_ci95_low"]) for row in main])
        highs = np.asarray([float(row[f"{metric}_ci95_high"]) for row in main])
        axis.errorbar(np.arange(4), means, yerr=np.vstack((means - lows, highs - means)), fmt="o", capsize=4)
        axis.set_xticks(np.arange(4), labels, rotation=25, ha="right")
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("CRACKS Setting A: section-first mean and 95% section bootstrap CI")
    _save_figure(fig, root, "fig_cracks_main")


def _ablation_figure(ablations: Sequence[Mapping[str, Any]], root: Path) -> None:
    rows = list(ablations)[1:]
    labels = [str(row["run_id"]).replace("v2_", "") for row in rows]
    metrics = ("dice", "cldice", "skeleton_f1_at_2px")
    x = np.arange(len(rows))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for index, metric in enumerate(metrics):
        means = np.asarray([float(row[f"{metric}_delta_vs_full"]) for row in rows])
        lows = np.asarray([float(row[f"{metric}_delta_ci95_low"]) for row in rows])
        highs = np.asarray([float(row[f"{metric}_delta_ci95_high"]) for row in rows])
        axis.bar(x + (index - 1) * width, means, width, label=metric.replace("_", " "))
        axis.errorbar(x + (index - 1) * width, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Ablation minus full V2 (seed 42)")
    axis.set_title("Single-seed ablations; not model-selection evidence")
    axis.legend(frameon=False)
    _save_figure(fig, root, "fig_ablation")


def _synthetic_figure(rows: Sequence[Mapping[str, Any]], root: Path) -> None:
    v2 = next((row for row in rows if str(row.get("model", row.get("candidate"))) in {"anza_v2b", "C3"}), None)
    if v2 is None or v2.get("route_excess_over_chance") in {None, "", "NA", "N/A"}:
        return
    values = [
        _finite(v2["route_excess_over_chance"], field="route_excess_over_chance"),
        _finite(v2["route_entropy_normalized"], field="route_entropy_normalized"),
        _finite(v2["topology_constrained_pairing_score"], field="topology_constrained_pairing_score"),
    ]
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.bar(("excess over chance", "routing entropy", "topology assignment"), values)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Corrected routing audit (descriptive; see confidence intervals)")
    axis.tick_params(axis="x", rotation=18)
    _save_figure(fig, root, "fig_synthetic_mechanism")


def _examples_figure(
    selected: Sequence[int], panels: Mapping[int, Mapping[str, np.ndarray]], root: Path
) -> None:
    columns = ("input", "expert", "unet", "anza_v1", "anza_v2b")
    if any(section not in panels for section in selected):
        raise ValueError("Qualitative panels are missing a hash-selected section")
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(13.0, 6.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(selected), len(columns))
    for row_index, section in enumerate(selected):
        for column_index, column in enumerate(columns):
            data = np.asarray(panels[section][column])
            axes[row_index, column_index].imshow(data, cmap="gray", aspect="auto")
            axes[row_index, column_index].set_axis_off()
            if row_index == 0:
                axes[row_index, column_index].set_title(column.replace("anza_", "ANZA "))
        axes[row_index, 0].set_ylabel(f"Section {section}")
    fig.suptitle(f"Deterministic pre-error selection; salt={QUALITATIVE_SALT}", fontsize=10)
    _save_figure(fig, root, "fig_cracks_examples")


def build_deadline_statistics(
    output_root: Path,
    *,
    expert_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    expected_section_count: int | None = 40,
) -> dict[str, Any]:
    """Write the four frozen deadline tables without generating figures."""

    output_root = Path(output_root)
    tables = output_root / "tables"
    main, _ = aggregate_main(expert_rows)
    comparisons = paired_main_comparisons(expert_rows)
    ablations = aggregate_ablations(ablation_rows)
    synthetic = validate_synthetic_rows(synthetic_rows)
    section_ids = {int(row["section_id"]) for row in expert_rows}
    if expected_section_count is not None and len(section_ids) != expected_section_count:
        raise ValueError(
            f"Deadline Setting A requires {expected_section_count} expert sections; got {len(section_ids)}"
        )
    _write_csv(tables / "main_cracks.csv", main)
    _write_csv(tables / "paired_comparisons.csv", comparisons)
    _write_csv(tables / "ablations.csv", ablations)
    _write_csv(tables / "synthetic_corrected.csv", synthetic)
    _write_csv(
        output_root / "raw_per_section.csv",
        [dict(row) for row in expert_rows] + [dict(row) for row in ablation_rows],
    )
    selected = deterministic_qualitative_sections(section_ids)
    selection = {
        "status": "FROZEN_BEFORE_ERROR_REVIEW",
        "salt": QUALITATIVE_SALT,
        "selected_section_ids": selected,
        "selection_input": "section_id_only",
    }
    (output_root / "QUALITATIVE_SELECTION.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "DEADLINE_SCOPE.md").write_text(DEADLINE_SCOPE_TEXT)
    return {
        "status": "COMPLETE",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "seismic_section",
        "main_model_count": len(main),
        "paired_comparison_count": len(comparisons),
        "ablation_count": len(ablations),
        "qualitative_sections": selected,
    }


def generate_deadline_figures(
    output_root: Path,
    *,
    qualitative_panels: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Render deadline figures independently from training and statistics."""

    output_root = Path(output_root)
    tables = output_root / "tables"
    required = [
        tables / "main_cracks.csv",
        tables / "ablations.csv",
        tables / "synthetic_corrected.csv",
        output_root / "QUALITATIVE_SELECTION.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Deadline figure inputs missing: {missing}")
    main = _read_csv(required[0])
    ablations = _read_csv(required[1])
    synthetic = _read_csv(required[2])
    selected = json.loads(required[3].read_text())["selected_section_ids"]
    figures = output_root / "figures"
    _main_figure(main, figures)
    _ablation_figure(ablations, figures)
    _synthetic_figure(synthetic, figures)
    examples_status = "WAITING_HASH_SELECTED_PANELS"
    if qualitative_panels is not None:
        _examples_figure(selected, qualitative_panels, figures)
        examples_status = "COMPLETE"
    return {
        "status": "COMPLETE" if examples_status == "COMPLETE" else "PARTIAL",
        "examples_status": examples_status,
        "qualitative_sections": selected,
        "formats": ["png_300dpi", "svg", "pdf"],
    }


def build_deadline_evidence(output_root: Path) -> dict[str, Any]:
    """Build machine-linked numbers and claim-safe reports from deadline CSVs."""

    output_root = Path(output_root)
    tables_root = output_root / "tables"
    paths = {
        name: tables_root / name
        for name in (
            "main_cracks.csv",
            "paired_comparisons.csv",
            "ablations.csv",
            "synthetic_corrected.csv",
        )
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    selection_path = output_root / "QUALITATIVE_SELECTION.json"
    gate_path = output_root / "SYNTHETIC_GATE_AUDIT.json"
    if not selection_path.exists():
        missing.append(str(selection_path))
    if not gate_path.exists():
        missing.append(str(gate_path))
    if missing:
        raise FileNotFoundError(f"Deadline evidence inputs missing: {missing}")
    main = [_coerce_row(row) for row in _read_csv(paths["main_cracks.csv"])]
    comparisons = [_coerce_row(row) for row in _read_csv(paths["paired_comparisons.csv"])]
    ablations = [_coerce_row(row) for row in _read_csv(paths["ablations.csv"])]
    synthetic = [_coerce_row(row) for row in _read_csv(paths["synthetic_corrected.csv"])]
    mechanism = next(
        (row for row in synthetic if row.get("model", row.get("candidate")) in {"anza_v2b", "C3"}),
        {},
    )
    gate = json.loads(gate_path.read_text())
    gate_evidence = gate.get("corrected_mechanism_evidence", {})
    gate_verdict = gate_evidence.get("verdict")
    if gate_verdict not in {"SUPPORTED_ABOVE_CHANCE", "NOT_ESTABLISHED", "NEGATIVE"}:
        raise ValueError("Synthetic gate has no valid corrected mechanism verdict")
    mechanism_low = mechanism.get("route_excess_over_chance_ci95_low")
    mechanism_supported = gate_verdict == "SUPPORTED_ABOVE_CHANCE"
    if mechanism_supported and (
        mechanism_low in {None, "", "NA", "N/A"}
        or _finite(mechanism_low, field="mechanism CI") <= 0
    ):
        raise ValueError("Positive synthetic gate conflicts with the corrected C3 CI")
    false_bridge_verdict = gate.get("false_bridge_verdict", {}).get("status")
    if false_bridge_verdict not in {
        "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE",
        "FALSE_BRIDGE_ENDPOINT_RETAINS_DISCRIMINATIVE_RANGE",
    }:
        raise ValueError("Synthetic gate has no valid false-bridge verdict")
    verdict = (
        "DEADLINE_RESULT_READY"
        if mechanism_supported
        else "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM"
    )
    provenance_paths = {
        **paths,
        "raw_per_section.csv": output_root / "raw_per_section.csv",
        "QUALITATIVE_SELECTION.json": selection_path,
        "DEADLINE_SCOPE.md": output_root / "DEADLINE_SCOPE.md",
        "SYNTHETIC_GATE_AUDIT.json": gate_path,
    }
    provenance = {
        name: {"path": str(path.relative_to(output_root)), "sha256": _sha256(path)}
        for name, path in provenance_paths.items()
    }
    numbers = {
        "schema_version": 1,
        "status": verdict,
        "scope": {
            "included": "Setting A and corrected synthetic evaluator",
            "settings_b_c": "NOT_RUN_DEADLINE_SCOPE",
        },
        "main_cracks": main,
        "paired_comparisons": comparisons,
        "ablations": ablations,
        "synthetic_corrected": synthetic,
        "synthetic_mechanism_supported": mechanism_supported,
        "synthetic_mechanism_verdict": gate_verdict,
        "false_bridge_verdict": false_bridge_verdict,
        "qualitative_selection": json.loads(selection_path.read_text()),
        "provenance": provenance,
    }
    (output_root / "THESIS_NUMBERS.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n"
    )
    evidence = [
        "# Thesis evidence",
        "",
        "All numeric results are machine-derived from the frozen tables below.",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
        *[f"| `{record['path']}` | `{record['sha256']}` |" for record in provenance.values()],
        "",
        "Main models use seeds 41/42/43 averaged within section before section aggregation.",
        f"Paired intervals use {BOOTSTRAP_RESAMPLES} section bootstrap resamples.",
        "Ablations are seed-42-only and cannot support post-hoc model promotion.",
    ]
    (output_root / "THESIS_EVIDENCE.md").write_text("\n".join(evidence) + "\n")
    mechanism_sentence = (
        "Transport carries continuation information above chance on the controlled benchmark."
        if mechanism_supported
        else "Corrected evaluation did not establish reliable branch-identity routing."
    )
    report = "\n".join(
        [
            "# Deadline report",
            "",
            "## STATUS",
            "",
            verdict,
            "",
            "## Main numbers",
            "",
            "See `THESIS_NUMBERS.json` and `tables/main_cracks.csv`; no metric is retyped manually.",
            "",
            "## Paired deltas",
            "",
            "See `tables/paired_comparisons.csv`; the independent resampling unit is section.",
            "",
            "## Ablations",
            "",
            "Seed 42 only; no-replay, no-fuzzy, and no-directional results are not primary-model evidence.",
            "",
            "## Synthetic mechanism verdict",
            "",
            mechanism_sentence,
            "",
            "## False bridge verdict",
            "",
            false_bridge_verdict,
            "",
            "## Claims allowed",
            "",
            "Setting A results may be described as crowd-to-expert same-section reconstruction. " + mechanism_sentence,
            "",
            "## Claims forbidden",
            "",
            "- Unseen-section generalization from Setting A.",
            "- Model-specific routing for baselines without transport output.",
            "- Post-hoc promotion of a one-seed ablation.",
            "- A positive mechanism claim when the corrected CI includes zero.",
            "",
            "## Protocol deviations",
            "",
            "Settings B/C are `NOT_RUN_DEADLINE_SCOPE` and absent from submitted claims.",
            "",
            "## Exact artifact paths",
            "",
            "- `tables/main_cracks.csv`",
            "- `tables/paired_comparisons.csv`",
            "- `tables/ablations.csv`",
            "- `tables/synthetic_corrected.csv`",
            "- `figures/fig_cracks_main.png`",
            "- `figures/fig_cracks_examples.png`",
            "- `figures/fig_ablation.png`",
            "- `figures/fig_synthetic_mechanism.png`",
        ]
    ) + "\n"
    (output_root / "DEADLINE_REPORT.md").write_text(report)
    return {
        "status": verdict,
        "mechanism_supported": mechanism_supported,
        "false_bridge_verdict": false_bridge_verdict,
    }


def build_deadline_package(
    output_root: Path,
    *,
    expert_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    qualitative_panels: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    expected_section_count: int | None = 40,
) -> dict[str, Any]:
    """Run only deadline post-processing; never train or access expert masks."""

    statistics = build_deadline_statistics(
        output_root,
        expert_rows=expert_rows,
        ablation_rows=ablation_rows,
        synthetic_rows=synthetic_rows,
        expected_section_count=expected_section_count,
    )
    figures = generate_deadline_figures(
        output_root, qualitative_panels=qualitative_panels
    )
    evidence = build_deadline_evidence(output_root)
    return {"statistics": statistics, "figures": figures, "evidence": evidence}


# Backward-compatible spelling for callers created while the deadline spec was
# being translated into code.
build_deadline_reporting = build_deadline_package
