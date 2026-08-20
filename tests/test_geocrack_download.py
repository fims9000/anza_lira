from __future__ import annotations

from pathlib import Path
import threading
import time
import zipfile

from PIL import Image
import pytest

from scripts.download_geocrack import (
    assert_download_stable,
    import_local_geocrack,
    safe_extract_zip,
    select_patched_files,
    validate_local_pairs,
)
from tests.fixtures.geocrack_synthetic import CASE_NAMES, generate_synthetic_geocrack


def _metadata() -> dict:
    return {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "directoryLabel": "Patched Data",
                        "dataFile": {"id": 11, "filename": "geocrack_patches.zip", "filesize": 10},
                    },
                    {
                        "directoryLabel": "Raw Data",
                        "dataFile": {"id": 12, "filename": "raw_images.zip", "filesize": 20},
                    },
                ]
            }
        }
    }


def test_only_patched_dataverse_files_are_selected() -> None:
    selected = select_patched_files(_metadata())
    assert [item["dataFile"]["id"] for item in selected] == [11]


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_zip(archive, tmp_path / "output")


def _fixture_archive(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    generate_synthetic_geocrack(source)
    archive = tmp_path / "geocrack.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source))
    return archive


def test_manual_archive_import_validates_and_is_idempotent(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path)
    output = tmp_path / "data" / "geocrack"
    first = import_local_geocrack(
        output, local_archive=archive, expected_pairs=len(CASE_NAMES), stability_seconds=0
    )
    second = import_local_geocrack(
        output, local_archive=archive, expected_pairs=len(CASE_NAMES), stability_seconds=0
    )
    assert first["network_requests"] == 0
    assert first["validation"]["pair_count"] == len(CASE_NAMES)
    assert len(first["validation"]["dataset_sha256"]) == 64
    assert second["status"] == "already_verified"


def test_manual_root_import_does_not_copy_or_contact_network(tmp_path: Path) -> None:
    root = tmp_path / "incoming" / "extracted"
    generate_synthetic_geocrack(root)
    manifest = import_local_geocrack(
        tmp_path / "data" / "geocrack",
        local_root=root,
        expected_pairs=len(CASE_NAMES),
        stability_seconds=0,
    )
    assert manifest["data_root"] == str(root.resolve())
    assert manifest["network_requests"] == 0


@pytest.mark.parametrize("suffix", [".crdownload", ".part", ".tmp"])
def test_manual_import_rejects_browser_partial_markers(tmp_path: Path, suffix: str) -> None:
    root = tmp_path / "incoming"
    root.mkdir()
    (root / f"geocrack.zip{suffix}").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="Incomplete browser download"):
        assert_download_stable(root, stability_seconds=0)


def test_manual_import_rejects_input_whose_size_is_changing(tmp_path: Path) -> None:
    archive = tmp_path / "geocrack.zip"
    archive.write_bytes(b"started")

    def append_later() -> None:
        time.sleep(0.02)
        with archive.open("ab") as handle:
            handle.write(b"more")

    writer = threading.Thread(target=append_later)
    writer.start()
    with pytest.raises(RuntimeError, match="still changing"):
        assert_download_stable(archive, stability_seconds=0.05)
    writer.join()


def test_pair_validation_rejects_non_binary_mask(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_synthetic_geocrack(root)
    mask_path = next((root / "masks").glob("*.png"))
    mask = Image.open(mask_path).convert("L")
    mask.putpixel((1, 1), 17)
    mask.save(mask_path)
    with pytest.raises(ValueError, match="Non-binary mask"):
        validate_local_pairs(root, expected_pairs=len(CASE_NAMES))
