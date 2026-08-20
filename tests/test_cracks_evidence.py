import pytest

from cracks_experiment.evidence import _number, validate_report_numbers


def test_report_number_marker_resolves_and_detects_tampering() -> None:
    numbers = {"cracks": {"dice": 0.812345}}
    report = f"Dice: {_number(numbers, 'cracks.dice', '.3f')}"
    validate_report_numbers(report, numbers)
    with pytest.raises(ValueError, match="mismatch"):
        validate_report_numbers(report.replace("0.812", "0.999"), numbers)
