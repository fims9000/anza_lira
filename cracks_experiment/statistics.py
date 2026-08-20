"""Section-cluster statistics and machine-generated ANZA-LIRA v2 tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cracks_experiment.finetuning import FOLDS, setting_b_sources
from cracks_experiment.matrix import setting_a_matrix
from cracks_experiment.robustness import setting_c_models


PRIMARY_METRICS = (
    "dice",
    "iou",
    "cldice",
    "skeleton_f1_at_2px",
    "fragmentation",
    "trace_orientation_error_median_deg",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(
    values: Iterable[float],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    generator = np.random.default_rng(seed)
    samples = np.asarray(
        [array[generator.integers(0, len(array), len(array))].mean() for _ in range(int(resamples))]
    )
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "n_sections": len(array),
        "bootstrap_resamples": int(resamples),
    }


def paired_section_delta(
    first: dict[int, float],
    second: dict[int, float],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | bool]:
    sections = sorted(set(first) & set(second))
    if set(first) != set(second) or not sections:
        raise ValueError("Paired comparison requires identical non-empty section IDs")
    delta = np.asarray([first[section] - second[section] for section in sections], dtype=np.float64)
    result = bootstrap_mean(delta, resamples=resamples, seed=seed)
    low, high = float(result["ci95_low"]), float(result["ci95_high"])
    return {**result, "ci_excludes_zero": bool(low > 0.0 or high < 0.0)}


def _section_average(rows: list[dict[str, str]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["section_id"]), []).append(float(row[metric]))
    return {section: float(np.mean(values)) for section, values in grouped.items()}


def _model_table(
    rows_by_model: dict[str, list[dict[str, str]]],
    setting: str,
) -> list[dict[str, Any]]:
    output = []
    for model, rows in rows_by_model.items():
        section_metrics = {metric: _section_average(rows, metric) for metric in PRIMARY_METRICS}
        row: dict[str, Any] = {"setting": setting, "model": model, "status": "COMPLETE"}
        for offset, metric in enumerate(PRIMARY_METRICS):
            summary = bootstrap_mean(section_metrics[metric].values(), seed=100 + offset)
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_ci95_low"] = summary["ci95_low"]
            row[f"{metric}_ci95_high"] = summary["ci95_high"]
        row["n_sections"] = len(next(iter(section_metrics.values())))
        output.append(row)
    return output


def build_statistics(study_root: Path) -> dict[str, Any]:
    study_root = Path(study_root)
    cracks_root = study_root / "cracks"
    tables_root = study_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    main_rows: dict[str, list[dict[str, str]]] = {}
    ablation_rows: dict[str, list[dict[str, str]]] = {}
    setting_a_root = cracks_root / "setting_a_expert"
    for spec in setting_a_matrix():
        path = setting_a_root / f"{spec.run_id}-{spec.run_hash}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Setting A expert rows missing: {path}")
        rows = [row for row in _read_csv(path) if row["policy"] == "paper_like"]
        destination = main_rows if spec.comparison_family == "main" else ablation_rows
        destination.setdefault(spec.model if spec.comparison_family == "main" else spec.run_id, []).extend(rows)
    setting_a_table = _model_table(main_rows, "A_crowd_to_expert_same_image")

    setting_b_rows: dict[str, list[dict[str, str]]] = {}
    for spec in setting_b_sources():
        for fold in FOLDS["folds"]:
            prefix = f"{spec.model}_fold{fold['fold']}-"
            matches = sorted((cracks_root / "setting_b").glob(f"{prefix}*/test_sections.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one Setting B artifact for {prefix}, got {len(matches)}")
            setting_b_rows.setdefault(spec.model, []).extend(_read_csv(matches[0]))
    setting_b_table = _model_table(setting_b_rows, "B_limited_expert_fine_tuning")

    setting_c_rows: dict[str, list[dict[str, str]]] = {}
    for spec in setting_c_models():
        for fold in FOLDS["folds"]:
            prefix = f"{spec.model}_fold{fold['fold']}-"
            matches = sorted((cracks_root / "setting_c").glob(f"{prefix}*/test_sections.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one Setting C artifact for {prefix}, got {len(matches)}")
            setting_c_rows.setdefault(spec.model, []).extend(_read_csv(matches[0]))
    setting_c_table = _model_table(setting_c_rows, "C_image_disjoint_robustness")
    _write_csv(tables_root / "main_cracks.csv", setting_a_table + setting_b_table + setting_c_table)

    comparisons = []
    v2_sections = {
        metric: _section_average(main_rows["anza_v2b"], metric) for metric in PRIMARY_METRICS
    }
    for comparator in ("unet", "deformable_unet", "anza_v1"):
        for offset, metric in enumerate(PRIMARY_METRICS):
            other = _section_average(main_rows[comparator], metric)
            delta = paired_section_delta(v2_sections[metric], other, seed=300 + offset)
            comparisons.append(
                {
                    "setting": "A_crowd_to_expert_same_image",
                    "first": "anza_v2b",
                    "second": comparator,
                    "metric": metric,
                    "desirable_direction": "lower" if metric in {"fragmentation", "trace_orientation_error_median_deg"} else "higher",
                    "delta_first_minus_second": delta["mean"],
                    "ci95_low": delta["ci95_low"],
                    "ci95_high": delta["ci95_high"],
                    "ci_excludes_zero": delta["ci_excludes_zero"],
                    "n_sections": delta["n_sections"],
                }
            )
    _write_csv(tables_root / "paired_comparisons.csv", comparisons)

    ablation_input = {"v2_full_s42": [row for row in main_rows["anza_v2b"] if int(row["seed"]) == 42], **ablation_rows}
    ablation_table = _model_table(ablation_input, "A_ablation_seed42")
    empty = {key: "" for key in ablation_table[0]}
    for name in ("v2_no_junction_s42", "v2_no_cone_s42"):
        ablation_table.append(
            {
                **empty,
                "setting": "A_ablation_seed42",
                "model": name,
                "status": "NOT_RUN_FROZEN_SYNTHETIC_MECHANISM_NEGATIVE",
            }
        )
    _write_csv(tables_root / "ablations.csv", ablation_table)

    human_path = cracks_root / "human_comparison" / "annotator_sections.csv"
    disagreement_path = cracks_root / "disagreement" / "summary.json"
    if not human_path.exists() or not disagreement_path.exists():
        raise FileNotFoundError("Human comparison or disagreement analysis is incomplete")
    human_rows = [row for row in _read_csv(human_path) if row["policy"] == "paper_like"]
    human_table = []
    for role in ("novice", "practitioner"):
        selected = [row for row in human_rows if row["role"] == role]
        for metric in ("dice", "cldice", "skeleton_f1_at_2px"):
            values = [float(row[metric]) for row in selected]
            human_table.append(
                {
                    "role": role,
                    "metric": metric,
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "annotation_section_count": len(values),
                }
            )
    _write_csv(tables_root / "human_comparison.csv", human_table)
    disagreement = json.loads(disagreement_path.read_text())
    disagreement_rows = [
        {"human_metric": "mean_human_entropy", "model_metric": metric, **values}
        for metric, values in disagreement["correlations"].items()
    ]
    _write_csv(tables_root / "disagreement_correlations.csv", disagreement_rows)

    synthetic_summary = json.loads((study_root / "synthetic" / "test" / "summary.json").read_text())
    synthetic_rows = []
    for candidate, result in synthetic_summary["models"].items():
        metrics = result["metrics"]
        synthetic_rows.append(
            {
                "candidate": candidate,
                "model": result["model"],
                "visible_dice": metrics["visible_dice"],
                "branch_pairing_accuracy": metrics["branch_pairing_accuracy"],
                "false_merge_rate": metrics["false_merge_rate"],
                "false_split_rate": metrics["false_split_rate"],
                "identity_switch_rate": metrics["identity_switch_rate"],
                "gap_recovery_rate": metrics["gap_recovery_rate"],
                "false_bridge_rate": metrics["false_bridge_rate"],
            }
        )
    _write_csv(tables_root / "structural_benchmark.csv", synthetic_rows)
    payload = {
        "status": "COMPLETE",
        "bootstrap_unit": "seismic_section",
        "bootstrap_resamples": 2000,
        "setting_a_model_count": len(setting_a_table),
        "setting_b_model_count": len(setting_b_table),
        "setting_c_model_count": len(setting_c_table),
        "paired_comparison_count": len(comparisons),
        "all_values_finite": all(
            math.isfinite(float(value))
            for row in setting_a_table + setting_b_table + setting_c_table
            for key, value in row.items()
            if key not in {"setting", "model", "status"}
        ),
    }
    (tables_root / "statistics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
