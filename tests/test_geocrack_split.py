from __future__ import annotations

import csv
from pathlib import Path

import pytest

from datasets.geocrack import extract_source_image_id
from scripts.audit_geocrack_sites import audit_site_mapping
from scripts.check_geocrack_split import assert_no_source_leakage, freeze_or_verify_test_split, load_sources
from scripts.prepare_geocrack_split import grouped_small_split


def _write_split(path: Path, sources: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patch_id", "source_image_id", "image_path", "mask_path"])
        writer.writeheader()
        for index, source in enumerate(sources):
            writer.writerow(
                {
                    "patch_id": f"{source}_patch{index}",
                    "source_image_id": source,
                    "image_path": f"images/{source}_original_patch{index}.png",
                    "mask_path": f"masks/{source}_binarymask_patch{index}.png",
                }
            )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("DJI_0194_original_patch155.png", "DJI_0194"),
        ("DJI_0194_binarymask_patch155.png", "DJI_0194"),
        ("Oman_01_patch_001.png", "Oman_01"),
        ("Oman_01_patch_001_mask.png", "Oman_01"),
    ],
)
def test_extract_source_image_id(filename: str, expected: str) -> None:
    assert extract_source_image_id(filename) == expected


def test_source_leakage_is_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    _write_split(train, ["source_a", "source_b"])
    _write_split(val, ["source_c"])
    _write_split(test, ["source_a"])

    with pytest.raises(ValueError, match="TRAIN/TEST"):
        assert_no_source_leakage(load_sources(train), load_sources(val), load_sources(test))


def test_grouped_small_split_never_splits_sources() -> None:
    rows = [
        {"patch_id": f"{source}_{index}", "source_image_id": source}
        for source, count in (("a", 8), ("b", 7), ("c", 6), ("d", 5), ("e", 4), ("f", 3))
        for index in range(count)
    ]
    splits = grouped_small_split(rows, targets={"train": 16, "val": 8, "test": 9}, seed=2026)
    source_sets = {name: {row["source_image_id"] for row in part} for name, part in splits.items()}

    assert_no_source_leakage(source_sets["train"], source_sets["val"], source_sets["test"])
    assert sum(len(part) for part in splits.values()) == len(rows)
    assert all(splits[name] for name in ("train", "val", "test"))


def test_grouped_split_can_require_multiple_sources_per_partition() -> None:
    rows = [
        {"patch_id": f"{source}_{index}", "source_image_id": source}
        for source in ("a", "b", "c", "d", "e", "f")
        for index in range(2)
    ]
    splits = grouped_small_split(
        rows,
        targets={"train": 4, "val": 4, "test": 4},
        seed=2026,
        min_sources={"train": 2, "val": 2, "test": 2},
    )
    assert all(len({row["source_image_id"] for row in part}) >= 2 for part in splits.values())


def test_frozen_test_contract_rejects_later_csv_mutation(tmp_path: Path) -> None:
    test_csv = tmp_path / "test.csv"
    _write_split(test_csv, ["source_a"])
    contract = tmp_path / "test_split.sha256"
    digest = freeze_or_verify_test_split(test_csv, contract)
    assert contract.read_text(encoding="utf-8").strip() == digest
    with test_csv.open("a", encoding="utf-8") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="Frozen GeoCrack test CSV hash changed"):
        freeze_or_verify_test_split(test_csv, contract)


def test_site_mapping_stays_not_established_without_official_metadata(tmp_path: Path) -> None:
    payload = audit_site_mapping(["site_like_name_1", "site_like_name_2"], None)
    assert payload["site_mapping_status"] == "NOT_ESTABLISHED"
    assert payload["inference_from_filename"] is False
    assert payload["fallback_split_unit"] == "source_image_id"


def test_unambiguous_official_site_mapping_can_be_established(tmp_path: Path) -> None:
    mapping = tmp_path / "official.csv"
    mapping.write_text(
        "source_image_id,geological_site\nsource_a,site_1\nsource_b,site_2\n",
        encoding="utf-8",
    )
    payload = audit_site_mapping(["source_a", "source_b"], mapping)
    assert payload["site_mapping_status"] == "ESTABLISHED"
    assert payload["mapping"] == {"source_a": "site_1", "source_b": "site_2"}
