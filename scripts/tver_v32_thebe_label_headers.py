"""V32 Thebe external field TRAIN/VAL fault label NPZ provenance and shape probe.

Only two explicitly chosen small expert fault MASK archives are downloaded.
NO test file, no seismic amplitudes, and NO fault pixels are loaded. This
stages an independent field experiment without casually fetching 53 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request,urlopen

from scripts.tver_v32_thebe_external_manifest import (
    DATASET_PID,extract_files,MAX_FIT_FILE_BYTES)

FROZEN_MASK_NAMES=("faulttrain1.npz","faultval1.npz")
TEST_MASK_NAME="faulttest7.npz"
FROZEN_SEISMIC_PAIRS={
    "faulttrain1.npz":"seistrain1.npz",
    "faultval1.npz":"seisval1.npz",
    "faulttest7.npz":"seistest7.npz",
}
MAX_MASK_DOWNLOAD_BYTES=20_000_000


def header_only_npz(path):
    import numpy as np
    from numpy.lib.format import (
        read_magic,read_array_header_1_0,read_array_header_2_0,
    )
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        if not members or len(members)>12:
            raise RuntimeError("STOP unexpected external label NPZ member count")
        results=[]
        for item in members:
            if item.is_dir() or not item.filename.endswith(".npy"):
                raise RuntimeError("STOP unexpected compressed label member")
            if item.filename.startswith("/") or ".." in Path(item.filename).parts:
                raise RuntimeError("STOP unsafe label NPZ member")
            # Only decompress first ~200 bytes required to parse NPY HEADER.
            with archive.open(item) as stream:
                version=read_magic(stream)
                if version==(1,0):
                    shape,fortran,dtype=read_array_header_1_0(stream)
                elif version in ((2,0),(3,0)):
                    shape,fortran,dtype=read_array_header_2_0(stream)
                else:
                    raise RuntimeError(f"STOP unsupported numpy array format {version}")
            if dtype.hasobject or not shape or len(shape)>5:
                raise RuntimeError("STOP invalid or unsafe Thebe label array")
            expected_bytes=int(np.prod(shape,dtype=np.int64))*dtype.itemsize
            if expected_bytes>12_000_000_000:
                raise RuntimeError("STOP external label array too large")
            # NPZ header-only: avoid scanning 100x3k by 1.5k pixels.
            results.append({
                "member":item.filename,"shape":list(map(int,shape)),
                "dtype":str(dtype),"fortran_order":bool(fortran),
                "np_array_expected_bytes":expected_bytes,
                "npz_uncompressed_member_bytes":item.file_size,
                "npz_member_compressed_bytes":item.compress_size,
            })
        return results


def fetch_exact(url,target,expected_bytes,expected_md5):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if expected_bytes>MAX_MASK_DOWNLOAD_BYTES:
        raise RuntimeError("STOP not a small expected expert mask file")
    for attempt in range(3):
        if target.is_file():
            if (target.stat().st_size==expected_bytes
                and hashlib.md5(target.read_bytes()).hexdigest()==expected_md5):
                return "CACHE_VERIFIED"
            target.unlink()
        tmp=target.with_name(target.name+".part")
        try:
            req=Request(url,headers={"User-Agent":"AnzaLiraV32SmallExternalMask/1.0"})
            count=0;digest=hashlib.md5()
            with urlopen(req,timeout=120) as resp,tmp.open("wb") as out:
                while True:
                    chunk=resp.read(1024*1024)
                    if not chunk:break
                    count+=len(chunk)
                    if count>MAX_MASK_DOWNLOAD_BYTES:
                        raise RuntimeError("STOP provider returned oversize file")
                    out.write(chunk);digest.update(chunk)
            if count!=expected_bytes or digest.hexdigest()!=expected_md5:
                raise RuntimeError("STOP official file size/checksum mismatch")
            tmp.replace(target)
            return "DOWNLOADED_VERIFIED"
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(1+attempt)
    raise RuntimeError("STOP external mask download retries exhausted")


def execute(manifest_path,out_dir):
    manifest_path=Path(manifest_path)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "protocol":"TV_V32_THEBE_EXTERNAL_FIELD_SMALL_LABEL_HEADERS_ONLY",
        "source_pid":DATASET_PID,
        "downloaded_TEST_labels":False,
        "downloaded_seismic_data":False,
        "decoded_mask_pixels":False,
        "selected_file_bases_before_any_numpy_contents":list(FROZEN_MASK_NAMES),
        "reserved_untouched_TEST_mask_file":TEST_MASK_NAME,
        "files":{},
    }
    try:
        doc=json.loads(manifest_path.read_text())
        if doc["source_pid"]!=DATASET_PID or not doc["status"].endswith(
            "METADATA_ONLY_COMPLETE"):
            raise RuntimeError("STOP original official Thebe source manifest missing")
        files={x["name"]:x for x in doc["files"]}
        for name in (*FROZEN_MASK_NAMES,TEST_MASK_NAME,*FROZEN_SEISMIC_PAIRS.values()):
            if name not in files:
                raise RuntimeError("STOP required predeclared Thebe label/seismic file missing")
        for name in FROZEN_MASK_NAMES:
            record=files[name]
            if (record["restricted"] or record["bytes"]>MAX_MASK_DOWNLOAD_BYTES
                or record["checksum_algorithm"]!="MD5"
                or not record["checksum_value"]):
                raise RuntimeError("STOP official source mask not eligible for trusted small download")
            target=out_dir/name
            source=fetch_exact(record["download_api"],target,
                               record["bytes"],record["checksum_value"])
            headers=header_only_npz(target)
            report["files"][name]={
                "file_id":record["id"],"source":record["download_api"],
                "provenance":"official Harvard Dataverse DATAFILE",
                "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
                "md5_verified":record["checksum_value"],
                "download_method":source,"bytes":target.stat().st_size,
                "shape_headers_only":headers,
                "paired_external_seismic_filename":FROZEN_SEISMIC_PAIRS[name],
            }
        report["status"]="V32_THEBE_TRAIN_VAL_LABEL_SHAPES_VERIFIED_TEST_UNOPENED"
        report["feasible_next_step"]="Download only seistrain1.npz and seisval1.npz if resource budget permits; match NPY shape/order without opening test."
    except Exception as error:
        report["status"]="V32_STOP_THEBE_SMALL_LABEL_SOURCE_SHAPE"
        report["error"]=f"{type(error).__name__}: {error}"
    receipt=out_dir/"v32_train_val_label_headers.json"
    receipt.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "headers":{k:v["shape_headers_only"] for k,v in report["files"].items()},
        "test_files_decoded":False,
        "seismic_files_downloaded":False,
        "error":report.get("error"),
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    execute(args.manifest,args.out_dir)
