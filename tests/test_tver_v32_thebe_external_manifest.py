"""V32 official Thebe Dataverse filename-only inventory; no data bytes."""
from __future__ import annotations

import pytest

from scripts.tver_v32_thebe_external_manifest import (
    DATASET_PID,extract_files,summarize,_category,
)


def _mock():
    return {
        "status":"OK",
        "data":{"latestVersion":{
            "files":[
                {"dataFile":{"id":11,"filename":"seismictrain1.npy","filesize":340_000_000,
                             "checksum":{"type":"MD5","value":"abc"}}},
                {"dataFile":{"id":12,"filename":"faulttrain1.npy","filesize":140_000_000,
                             "checksum":{"type":"MD5","value":"def"}}},
                {"dataFile":{"id":21,"filename":"seismictest1.npy","filesize":1_600_000_000}},
                {"dataFile":{"id":22,"filename":"faulttest1.npy","filesize":900_000_000}},
            ],
            "metadataBlocks":{"citation":{"fields":[
                {"typeName":"title","value":"Thebe example"}]}}
        }}
    }


def test_file_manifest_integrity_and_match_hints():
    report=summarize(_mock())
    assert report["status"].startswith("V32_THEBE")
    assert report["source_pid"]==DATASET_PID
    assert report["file_count"]==4
    assert report["candidate_small_files_total"]==3
    assert report["split_names_with_both_seismic_and_fault_file_hints"]==["train","test"]
    assert report["archive_bytes_downloaded"]==0
    assert report["heldout_test_outcomes_touched"] is False
    assert report["seismic_fault_instance_IDs_verified"] is False


def test_no_arbitrary_id_duplicates_and_filenames():
    doc=_mock()
    doc["data"]["latestVersion"]["files"][1]["dataFile"]["id"]=11
    with pytest.raises(RuntimeError,match="Duplicate"):
        extract_files(doc)
    doc=_mock()
    doc["data"]["latestVersion"]["files"][0]["dataFile"]["filename"]=""
    with pytest.raises(RuntimeError,match="filename"):
        extract_files(doc)


def test_filename_heuristic_does_not_assert_a_real_pair_or_instance():
    assert _category("faultvalid2.npy")==("valid","fault_label")
    assert _category("seismictrain3.npy")==("train","seismic_image")
    assert _category("readme.txt")==("other","other")
    assert summarize(_mock())["this_does_NOT_establish_matching_shapes_or_training_splits"]
