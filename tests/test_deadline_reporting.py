import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cracks_experiment.deadline_reporting import (
    BOOTSTRAP_RESAMPLES,
    MAIN_MODELS,
    MAIN_SEEDS,
    aggregate_main,
    build_deadline_package,
    deterministic_qualitative_sections,
    paired_main_comparisons,
    validate_synthetic_rows,
)


def _metric_payload(value: float) -> dict[str, float]:
    return {
        "dice": value,
        "cldice": value - 0.02,
        "skeleton_f1_at_2px": value - 0.03,
        "fragmentation": 1.0 - value,
        "trace_orientation_error_median_deg": 20.0 * (1.0 - value),
    }


def _main_rows() -> list[dict[str, object]]:
    offsets = {"unet": 0.0, "deformable_unet": 0.01, "anza_v1": 0.02, "anza_v2b": 0.05}
    rows = []
    for model in MAIN_MODELS:
        for seed in MAIN_SEEDS:
            for section in (101, 102, 103, 104):
                value = 0.5 + offsets[model] + 0.01 * (seed - 41) + 0.001 * (section - 100)
                rows.append({"model": model, "seed": seed, "section_id": section, **_metric_payload(value)})
    return rows


def _ablation_rows() -> list[dict[str, object]]:
    offsets = {
        "v2_full_s42": 0.0,
        "v2_no_replay_s42": 0.02,
        "v2_no_fuzzy_s42": -0.01,
        "v2_no_directional_s42": -0.03,
    }
    return [
        {
            "run_id": run_id,
            "seed": 42,
            "section_id": section,
            **_metric_payload(0.65 + offset + section * 0.0001),
        }
        for run_id, offset in offsets.items()
        for section in (101, 102, 103, 104)
    ]


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in ("unet", "deformable_unet", "anza_v1"):
        rows.append({
            "model": model,
            "visible_dice": 0.8,
            "false_bridge_rate": 1.0,
            "route_top1_hit": None,
        })
    rows.append({
        "model": "anza_v2b",
        "visible_dice": 0.82,
        "false_bridge_rate": 1.0,
        "route_top1_hit": 0.6,
        "route_true_probability_mass": 0.45,
        "route_mrr": 0.7,
        "route_average_precision": 0.68,
        "route_entropy_normalized": 0.75,
        "route_excess_over_chance": 0.05,
        "route_excess_over_chance_ci95_low": -0.01,
        "topology_constrained_pairing_score": 0.55,
    })
    rows.append({
        "model": "geometry_only_minimum_angle_heuristic",
        "uses_generator_branch_geometry": True,
        "visible_dice": "NA",
        "false_bridge_rate": "NA",
    })
    return rows


def test_three_seed_summary_and_paired_delta_are_section_first() -> None:
    rows = _main_rows()
    table, sections = aggregate_main(rows)
    v2 = next(row for row in table if row["model"] == "anza_v2b")
    expected = np.mean([np.mean([row["dice"] for row in rows if row["model"] == "anza_v2b" and row["section_id"] == section]) for section in (101, 102, 103, 104)])
    assert v2["dice_mean"] == pytest.approx(expected)
    assert v2["seed_count"] == 3
    assert len(sections["anza_v2b"]) == 4

    comparisons = paired_main_comparisons(rows)
    dice = next(row for row in comparisons if row["second"] == "unet" and row["metric"] == "dice")
    assert dice["delta_first_minus_second"] == pytest.approx(0.05)
    assert dice["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert dice["pairing"] == "section+seed_delta_then_seed_mean_within_section"


def test_main_aggregation_fails_closed_on_missing_seed_section() -> None:
    rows = _main_rows()
    rows.pop()
    with pytest.raises(ValueError, match="section mismatch"):
        aggregate_main(rows)


def test_qualitative_selection_depends_only_on_ids_and_frozen_salt() -> None:
    first = deterministic_qualitative_sections([104, 101, 103, 102])
    second = deterministic_qualitative_sections([102, 103, 101, 104])
    assert first == second
    assert len(first) == 3


def test_corrected_synthetic_table_rejects_baseline_route_values() -> None:
    rows = _synthetic_rows()
    rows[0]["route_top1_hit"] = 0.5
    with pytest.raises(ValueError, match="must be NA"):
        validate_synthetic_rows(rows)


def test_deadline_package_generates_exact_tables_figures_and_claim_safe_evidence(tmp_path: Path) -> None:
    (tmp_path / "SYNTHETIC_GATE_AUDIT.json").write_text(json.dumps({
        "corrected_mechanism_evidence": {"verdict": "NOT_ESTABLISHED"},
        "false_bridge_verdict": {
            "status": "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE"
        },
    }))
    selected = deterministic_qualitative_sections((101, 102, 103, 104))
    panels = {
        section: {
            name: np.full((8, 20), index + section / 1000.0, dtype=np.float32)
            for index, name in enumerate(("input", "expert", "unet", "anza_v1", "anza_v2b"))
        }
        for section in selected
    }
    result = build_deadline_package(
        tmp_path,
        expert_rows=_main_rows(),
        ablation_rows=_ablation_rows(),
        synthetic_rows=_synthetic_rows(),
        qualitative_panels=panels,
        expected_section_count=4,
    )
    assert result["evidence"]["status"] == "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM"
    for name in ("main_cracks.csv", "paired_comparisons.csv", "ablations.csv", "synthetic_corrected.csv"):
        assert (tmp_path / "tables" / name).is_file()
    for stem in ("fig_cracks_main", "fig_cracks_examples", "fig_ablation", "fig_synthetic_mechanism"):
        for suffix in ("png", "svg", "pdf"):
            assert (tmp_path / "figures" / f"{stem}.{suffix}").stat().st_size > 0
    numbers = json.loads((tmp_path / "THESIS_NUMBERS.json").read_text())
    assert numbers["scope"]["settings_b_c"] == "NOT_RUN_DEADLINE_SCOPE"
    assert numbers["false_bridge_verdict"] == "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE"
    report = (tmp_path / "DEADLINE_REPORT.md").read_text()
    assert "did not establish reliable branch-identity routing" in report
    assert "Post-hoc promotion" in report
    with (tmp_path / "tables" / "ablations.csv").open(newline="") as handle:
        ablations = list(csv.DictReader(handle))
    assert all(row["scope"] == "single-seed ablation" for row in ablations)
