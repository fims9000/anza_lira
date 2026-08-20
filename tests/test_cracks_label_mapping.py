from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_cracks_archives import OFFICIAL_COLORS


def test_official_cracks_color_semantics_are_explicit() -> None:
    assert OFFICIAL_COLORS[(255, 127, 14)] == "certain_no_fault"
    assert OFFICIAL_COLORS[(44, 160, 44)] == "uncertain_fault"
    assert OFFICIAL_COLORS[(31, 119, 180)] == "certain_fault"
    assert (255, 255, 255) not in OFFICIAL_COLORS


def test_real_expert_color_audit_keeps_white_unassigned() -> None:
    path = Path("results/cracks_study/expert_color_audit.json")
    if not path.is_file():
        pytest.skip("Generated CRACKS audit is not stored in Git")
    audit = json.loads(path.read_text(encoding="utf-8"))
    mapping = {tuple(row["rgb"]): row["semantic"] for row in audit["colors"]}
    assert mapping[(255, 255, 255)] == "unassigned_ignore"
    assert audit["unknown_exceeds_one_percent"] is True
    assert audit["strict_target"]["positive"] == [31, 119, 180]
    assert audit["strict_target"]["negative"] == [255, 127, 14]
