from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.geocrack_study import _pixel_metrics, verify_report_consistency
from scripts.geocrack_synthetic_pipeline import run_synthetic_pipeline


def test_pixel_metrics_exact_and_empty_conventions_are_finite() -> None:
    target = np.zeros((9, 9), dtype=bool)
    target[4, 2:7] = True
    exact = _pixel_metrics(target, target)
    assert exact["dice"] == 1
    assert exact["iou"] == 1
    assert exact["precision"] == 1
    assert exact["recall"] == 1
    empty = _pixel_metrics(np.zeros_like(target), np.zeros_like(target))
    assert all(value == 1 for value in empty.values())
    assert all(np.isfinite(value) for value in [*exact.values(), *empty.values()])


def test_complete_synthetic_vertical_slice(tmp_path: Path) -> None:
    output = tmp_path / "synthetic_pipeline"
    status = run_synthetic_pipeline(output)
    assert status["status"] == "PASS"
    assert status["scientific_result"] is False
    assert set(status["steps"].values()) == {"PASS"}
    assert (output / "figures" / "synthetic_pipeline_overview.png").is_file()
    assert (output / "figures" / "synthetic_pipeline_overview.svg").is_file()
    assert (output / "figures" / "synthetic_pipeline_overview.pdf").is_file()
    verify_report_consistency(
        output / "THESIS_NUMBERS.json",
        output / "FINAL_REPORT.md",
        output / "REPORT_PROVENANCE.json",
    )
    thesis = json.loads((output / "THESIS_NUMBERS.json").read_text(encoding="utf-8"))
    assert thesis["scientific_result"] is False
    with (output / "FINAL_REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write("\nInjected metric: 999\n")
    with pytest.raises(ValueError, match="FINAL_REPORT changed"):
        verify_report_consistency(
            output / "THESIS_NUMBERS.json",
            output / "FINAL_REPORT.md",
            output / "REPORT_PROVENANCE.json",
        )
