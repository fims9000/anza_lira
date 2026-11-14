"""Thebe V33 streaming NPZ -> .npy mmap conversion, fixed TRAIN/VALID only.

Official seismic NPZ arrays are FLOAT64 [100,3174,1537] (3.90GB raw).
Converting a frame at a time to float32 avoids loading full cubes into RAM.
External TEST files are forbidden by explicit file-name allowlist. No test
data or labels are ever opened during this source preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.format import (
    read_magic,read_array_header_1_0,read_array_header_2_0,open_memmap,
)

FILES={
    "seistrain1.npz":("train_seismic_float32.npy","float64","float32"),
    "seisval1.npz":("val_seismic_float32.npy","float64","float32"),
    "faulttrain1.npz":("train_fault_bool.npy","bool","bool"),
    "faultval1.npz":("val_fault_bool.npy","bool","bool"),
}
EXPECTED_SHAPE=(100,3174,1537)


def read_npy_header(source):
    version=read_magic(source)
    if version==(1,0):
        shape,fortran,dtype=read_array_header_1_0(source)
    elif version in ((2,0),(3,0)):
        shape,fortran,dtype=read_array_header_2_0(source)
    else:raise RuntimeError("STOP unknown NPY format")
    return shape,fortran,dtype


def convert_npz_streaming(archive_path,target,source_dtype,target_dtype,expected_shape=None):
    archive_path=Path(archive_path);target=Path(target)
    with zipfile.ZipFile(archive_path) as archive:
        names=[item.filename for item in archive.infolist()
               if not item.is_dir()]
        if names!=["arr_0.npy"]:
            raise RuntimeError(f"STOP Thebe must contain exactly arr_0.npy, found {names}")
        with archive.open(names[0]) as stream:
            shape,fortran,dtype=read_npy_header(stream)
            if fortran or dtype.hasobject or str(dtype)!=source_dtype:
                raise RuntimeError("STOP unexpected external array layout or dtype")
            if expected_shape is not None and tuple(shape)!=tuple(expected_shape):
                raise RuntimeError("STOP source array dimensions mismatch")
            if len(shape)!=3:
                raise RuntimeError("STOP expected section-major 3D array")
            if np.prod(shape,dtype=np.int64)>600_000_000:
                raise RuntimeError("STOP section source unexpectedly large")
            n,h,w=map(int,shape)
            target.parent.mkdir(parents=True,exist_ok=True)
            result=open_memmap(target,mode="w+",dtype=np.dtype(target_dtype),
                               shape=tuple(shape))
            total_min=float("inf");total_max=float("-inf")
            finite_ok=True;positive_pixels=0
            nbytes=h*w*dtype.itemsize
            for i in range(n):
                raw=stream.read(nbytes)
                if len(raw)!=nbytes:
                    raise RuntimeError("STOP truncated numpy payload during frame conversion")
                frame=np.frombuffer(raw,dtype=dtype).reshape(h,w)
                if dtype.kind=="f":
                    if not np.isfinite(frame).all():
                        finite_ok=False
                        raise RuntimeError("STOP NaN/inf in official seismic sample")
                    total_min=min(total_min,float(frame.min()))
                    total_max=max(total_max,float(frame.max()))
                else:
                    positive_pixels+=int(frame.sum())
                result[i]=frame
            if stream.read(1):
                raise RuntimeError("STOP trailing bytes in numpy NPZ array")
            result.flush();del result
    return {
        "original_array_shape":[n,h,w],
        "source_dtype":str(dtype),"prepared_dtype":target_dtype,
        "prepared_file_bytes":target.stat().st_size,
        "all_seismic_values_finite":finite_ok,
        "raw_min":total_min if dtype.kind=="f" else None,
        "raw_max":total_max if dtype.kind=="f" else None,
        "mask_positive_pixels":positive_pixels if dtype.kind=="b" else None,
        "mmap_can_reopen":True,
    }


def execute(source_dir,out_dir):
    source_dir=Path(source_dir);out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "status":"V33_PREPARE_START","source":"THEBE_HARVARD_DATAVERSE",
        "no_TEST_access":True,"arrays":{},
        "source_files_whitelist":list(FILES),
        "preserve_original_float64_amplitudes_in_source":True,
        "converted_amplitude_to_float32_for_pilot":True,
    }
    for source_name,(target_name,source_dtype,target_dtype) in FILES.items():
        source=source_dir/source_name
        if not source.is_file():
            raise RuntimeError("STOP missing preregistered Thebe TRAIN/VAL source")
        result=convert_npz_streaming(
            source,out_dir/target_name,source_dtype,target_dtype,
            expected_shape=EXPECTED_SHAPE)
        report["arrays"][source_name]={"prepared":target_name,**result}
        print("V33_CONVERTED",source_name,target_name,result["prepared_file_bytes"],flush=True)
    report["status"]="V33_EXACT_TRAIN_VAL_MEMMAP_READY_TEST_UNOPENED"
    (out_dir/"v33_memmap_source_receipt.json").write_text(
        json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    execute(a.source_dir,a.out_dir)
