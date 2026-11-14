"""V32 verify expert fault label NPY headers without decoding large matrix bytes."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts.tver_v32_thebe_label_headers import (
    FROZEN_MASK_NAMES,TEST_MASK_NAME,header_only_npz,execute,
)


def test_label_probe_only_claims_train_val_small_files():
    assert FROZEN_MASK_NAMES==("faulttrain1.npz","faultval1.npz")
    assert TEST_MASK_NAME=="faulttest7.npz"


def test_parse_only_npz_header_dims_dtype(tmp_path):
    path=tmp_path/"labels.npz"
    data=np.zeros((2,6,8),dtype=np.uint8)
    np.savez_compressed(path,labels=data)
    parsed=header_only_npz(path)
    assert len(parsed)==1
    assert parsed[0]["shape"]==[2,6,8]
    assert parsed[0]["dtype"]=="uint8"
    assert parsed[0]["np_array_expected_bytes"]==96
    assert parsed[0]["member"]=="labels.npy"


def test_fail_closed_when_manifest_is_untrusted_and_no_test_file_is_downloaded(tmp_path):
    p=tmp_path/"source.json"
    p.write_text(json.dumps({"source_pid":"doi:INVALID","status":"wrong"}))
    result=execute(p,tmp_path/"out")
    assert result["status"]=="V32_STOP_THEBE_SMALL_LABEL_SOURCE_SHAPE"
    assert result["downloaded_TEST_labels"] is False
    assert result["downloaded_seismic_data"] is False
    assert result["decoded_mask_pixels"] is False
