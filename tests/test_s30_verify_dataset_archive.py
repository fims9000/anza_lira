"""Public byte-integrity tests with toy data; no private camera bytes or marks."""
import json
from hashlib import sha256
import pytest
from scripts.research.bubble_chamber.s30_verify_dataset_archive import verify_dataset_sources

SOURCES = ("H50Z_030_01_006.7z", "1mH2Data.7z", "070_008.7z")


def fake_manifest(tmp_path):
    rows = []
    for index, name in enumerate(SOURCES):
        raw = b"toy data for isolated checksum test" + bytes([index])
        (tmp_path / name).write_bytes(raw)
        rows.append({"file": name, "size_bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(json.dumps({"schema": "anza_bubble_chamber_dataset_integrity_v1", "source_archives": rows}))
    return manifest


def test_three_source_archives_all_verified(tmp_path):
    manifest = fake_manifest(tmp_path)
    result = verify_dataset_sources(manifest, tmp_path)
    assert result["verified"]
    assert len(result["sources"]) == 3
    assert all(x["status"] == "VERIFIED" for x in result["sources"])
    assert not result["automatic_particle_joins_authorized"]


def test_corruption_detected_and_not_silently_accepted(tmp_path):
    manifest = fake_manifest(tmp_path)
    (tmp_path / SOURCES[0]).write_bytes(b"changed")
    result = verify_dataset_sources(manifest, tmp_path)
    assert not result["verified"]
    assert result["sources"][0]["status"] == "HASH_MISMATCH"


def test_missing_source_detected(tmp_path):
    manifest = fake_manifest(tmp_path)
    (tmp_path / SOURCES[1]).unlink()
    result = verify_dataset_sources(manifest, tmp_path)
    assert not result["verified"]
    assert result["sources"][1]["status"] == "MISSING"


def test_manifest_schema_and_duplicate_guard(tmp_path):
    manifest = fake_manifest(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["source_archives"][2]["file"] = SOURCES[1]
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="repeated"):
        verify_dataset_sources(manifest, tmp_path)
    doc["schema"] = "unknown"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="versioned"):
        verify_dataset_sources(manifest, tmp_path)