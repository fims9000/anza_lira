from __future__ import annotations

import json

from method_repair.reporting import REQUIRED_FINAL_FILES, build_negative_package


def test_negative_package_contains_no_fabricated_real_result(tmp_path) -> None:
    result = build_negative_package(".", tmp_path, device="cpu")
    assert result["status"] == "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"
    assert all((tmp_path / name).is_file() for name in REQUIRED_FINAL_FILES)
    numbers = json.loads((tmp_path / "THESIS_NUMBERS.json").read_text())
    assert numbers["cracks"]["training"] == "NOT_RUN"
    assert numbers["cracks"]["expert_evaluation"] == "NOT_RUN"
    assert numbers["test_access"] == {"old_synthetic_samples": 0, "new_synthetic_samples": 0}
    assert "real-data superiority" in (tmp_path / "THESIS_EVIDENCE.md").read_text()


def test_report_uses_machine_numbers_for_key_a3_values(tmp_path) -> None:
    build_negative_package(".", tmp_path, device="cpu")
    numbers = json.loads((tmp_path / "THESIS_NUMBERS.json").read_text())
    report = (tmp_path / "FINAL_REPORT.md").read_text()
    a3 = numbers["synthetic_candidates"]["A3"]
    for name in (
        "visible_dice",
        "route_average_precision",
        "route_entropy_normalized",
        "false_bridge_rate",
    ):
        assert f"{a3[name]:.6f}" in report
