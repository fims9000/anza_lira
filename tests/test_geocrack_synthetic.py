from __future__ import annotations

from pathlib import Path

from datasets.geocrack import discover_pairs
from tests.fixtures.geocrack_synthetic import CASE_NAMES, generate_synthetic_geocrack


def test_generator_emits_all_controlled_cases_without_committed_images(tmp_path: Path) -> None:
    root = tmp_path / "geocrack_synthetic"
    manifest = generate_synthetic_geocrack(root)
    assert manifest["scientific_result"] is False
    assert set(manifest["case_names"]) == set(CASE_NAMES)
    assert {sample["case"] for sample in manifest["samples"]} == set(CASE_NAMES)
    pairs = discover_pairs(root)
    assert len(pairs) == len(CASE_NAMES)
    assert not list(Path("tests/fixtures/geocrack_synthetic").glob("*.png"))


def test_generator_can_emit_distinct_source_groups_for_integration_split(tmp_path: Path) -> None:
    root = tmp_path / "geocrack_synthetic"
    manifest = generate_synthetic_geocrack(root, variants_per_case=2)
    pairs = discover_pairs(root)
    assert len(pairs) == 2 * len(CASE_NAMES)
    assert len({pair["source_image_id"] for pair in pairs}) == len(pairs)
    assert manifest["sample_count"] == len(pairs)
