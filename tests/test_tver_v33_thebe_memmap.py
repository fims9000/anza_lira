"""Thebe streaming NPZ section-major and forbidden TEST prefixes."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.tver_v33_thebe_memmap import (
    FILES,EXPECTED_SHAPE,convert_npz_streaming,
)


def test_source_allowlist_excludes_external_test_and_persistent_types():
    assert len(FILES)==4
    assert all("test" not in name for name in FILES)
    assert FILES["seistrain1.npz"][1:] == ("float64","float32")
    assert FILES["faultval1.npz"][1:] == ("bool","bool")
    assert EXPECTED_SHAPE==(100,3174,1537)


def test_float64_source_converts_one_frame_at_a_time_into_float32_mmap(tmp_path):
    source=tmp_path/"seistrain1.npz"
    arr=np.linspace(-3,5,3*6*8).reshape(3,6,8).astype(np.float64)
    np.savez_compressed(source,arr)
    dest=tmp_path/"seismic_float32.npy"
    rec=convert_npz_streaming(
        source,dest,"float64","float32",expected_shape=(3,6,8))
    read=np.load(dest,mmap_mode="r")
    assert read.dtype==np.float32
    np.testing.assert_allclose(read,arr,rtol=1e-6)
    assert rec["all_seismic_values_finite"]
    assert rec["raw_min"]==-3.
    assert rec["raw_max"]==5.


def test_true_labels_stay_bool_and_positive_counts_recompute(tmp_path):
    source=tmp_path/"faulttrain1.npz"
    arr=np.zeros((3,6,8),dtype=bool)
    arr[0,0,:]=1
    arr[1,3,:4]=1
    np.savez_compressed(source,arr)
    dest=tmp_path/"fault_bool.npy"
    rec=convert_npz_streaming(
        source,dest,"bool","bool",expected_shape=arr.shape)
    assert rec["mask_positive_pixels"]==12
    np.testing.assert_array_equal(np.load(dest,mmap_mode="r"),arr)


def test_invalid_shape_stops_before_using_results(tmp_path):
    source=tmp_path/"data.npz"
    np.savez_compressed(source,np.ones((3,6,8),dtype=np.float64))
    with pytest.raises(RuntimeError,match="dimensions mismatch"):
        convert_npz_streaming(
            source,tmp_path/"out.npy",
            "float64","float32",expected_shape=(3,9,8))
