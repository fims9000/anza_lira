from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_cracks_split import freeze_or_verify_test_ids, validate_split_manifest


def _feasible_manifest() -> dict:
    assignments = {
        "train": list(range(1, 261)),
        "guard_1": list(range(261, 281)),
        "validation": list(range(281, 321)),
        "guard_2": list(range(321, 341)),
        "test": list(range(341, 401)),
    }
    return {"assignments": assignments, "counts": {name: len(ids) for name, ids in assignments.items()}}


def test_predeclared_coordinate_split_passes_without_overlap() -> None:
    assert validate_split_manifest(_feasible_manifest()) == []


def test_split_does_not_shift_missing_section_by_row_position() -> None:
    manifest = _feasible_manifest()
    manifest["assignments"]["validation"].remove(300)
    manifest["counts"]["validation"] -= 1
    assert 301 in manifest["assignments"]["validation"]
    assert validate_split_manifest(manifest) == []


def test_actual_expert_coverage_shortfall_fails_closed() -> None:
    manifest = _feasible_manifest()
    manifest["assignments"]["train"] = manifest["assignments"]["train"][:33]
    manifest["assignments"]["validation"] = manifest["assignments"]["validation"][:3]
    manifest["assignments"]["test"] = []
    manifest["counts"] = {name: len(ids) for name, ids in manifest["assignments"].items()}
    failures = validate_split_manifest(manifest)
    assert any("train has 33" in failure for failure in failures)
    assert any("validation has 3" in failure for failure in failures)
    assert any("test has 0" in failure for failure in failures)


def test_test_id_hash_fails_after_mutation(tmp_path: Path) -> None:
    checksum = tmp_path / "test_split.sha256"
    freeze_or_verify_test_ids([341, 342, 343], checksum)
    with pytest.raises(ValueError, match="Frozen CRACKS test IDs changed"):
        freeze_or_verify_test_ids([341, 342, 344], checksum)
