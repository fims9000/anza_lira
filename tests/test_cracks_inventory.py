from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_real_inventory_records_actual_verified_archive_structure() -> None:
    path = Path("results/cracks_study/archive_inventory.json")
    if not path.is_file():
        pytest.skip("Generated CRACKS audit is not stored in Git")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    assert inventory["status"] == "PASS"
    assert inventory["images"]["file_count"] == 396
    assert inventory["annotations"]["annotator_directory_count"] == 35
    assert inventory["annotations"]["expert_mask_count"] == 40
    assert not inventory["images"]["corrupt_or_invalid_files"]
    assert not inventory["annotations"]["corrupt_or_invalid_files"]
