"""V32 Thebe official TRAIN+VAL seismic chunk provenance and exact array HEADER.

No TEST bytes downloaded, no seismic pixel arrays materialized, no model
training. The official paired fault masks have been verified separately.
Each real compressed seismic source file is ~0.8–1.3GB, not the full 53GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request,urlopen

from scripts.tver_v32_thebe_label_headers import header_only_npz
from scripts.tver_v32_thebe_external_manifest import DATASET_PID

FROZEN_PAIRS=(
    ("seistrain1.npz","faulttrain1.npz"),
    ("seisval1.npz","faultval1.npz"),
)
RESERVED_TEST_PAIR=("seistest7.npz","faulttest7.npz")
MAX_SEISMIC_BYTES=1_500_000_000


def fetch_streaming_checked(record,target):
    if (record.get("restricted")
        or record.get("bytes",float("inf"))>MAX_SEISMIC_BYTES
        or record.get("checksum_algorithm")!="MD5"
        or not record.get("checksum_value")):
        raise RuntimeError("STOP Thebe original seismic file size/access/fixity unavailable")
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    size=int(record["bytes"]);digest=str(record["checksum_value"])
    def valid(path):
        if not path.is_file() or path.stat().st_size!=size:return False
        h=hashlib.md5()
        with path.open("rb") as f:
            for part in iter(lambda:f.read(2*1024*1024),b""):
                h.update(part)
        return h.hexdigest()==digest
    if valid(target):return "CACHE_VERIFIED"
    target.unlink(missing_ok=True)
    errors=[]
    for attempt in range(3):
        tmp=target.with_name(target.name+".partial")
        tmp.unlink(missing_ok=True)
        try:
            request=Request(record["download_api"],headers={
                "User-Agent":"AnzaLiraV32ThebeExternalSeismicSource/1.0"})
            n=0;h=hashlib.md5()
            with urlopen(request,timeout=150) as stream,tmp.open("wb") as out:
                while True:
                    part=stream.read(2*1024*1024)
                    if not part:break
                    n+=len(part)
                    if n>MAX_SEISMIC_BYTES:
                        raise RuntimeError("STOP streaming size exceeded safe pilot limit")
                    out.write(part);h.update(part)
            if n!=size or h.hexdigest()!=digest:
                raise RuntimeError("STOP official seismic chunk size or MD5 failed")
            tmp.replace(target);return "DOWNLOADED_MD5_VERIFIED"
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            tmp.unlink(missing_ok=True)
            if attempt<2:time.sleep(2+attempt)
    raise RuntimeError("STOP official external Thebe download failed: "+"; ".join(errors))


def execute(manifest_path,label_receipt_path,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "status":"V32_STOP_THEBE_SEISMIC_SOURCE",
        "protocol":"TV_V32_2PAIRED_TRAIN_VALID_EXTERNAL_FIELD_HEADER_ONLY",
        "reserved_test_pair":list(RESERVED_TEST_PAIR),
        "downloaded_or_decoded_TEST":False,
        "seismic_pixels_decoded":False,
        "expert_label_pixels_decoded":False,
        "matched_pair_headers":{},
    }
    try:
        m=json.loads(Path(manifest_path).read_text())
        l=json.loads(Path(label_receipt_path).read_text())
        if (m["source_pid"]!=DATASET_PID or
            m["status"]!="V32_THEBE_EXTERNAL_FIELD_DATAVERSE_METADATA_ONLY_COMPLETE"
            or l["status"]!="V32_THEBE_TRAIN_VAL_LABEL_SHAPES_VERIFIED_TEST_UNOPENED"
            or l["downloaded_TEST_labels"] or l["decoded_mask_pixels"]):
            raise RuntimeError("STOP source file ledger or label preflight invalid")
        by_name={x["name"]:x for x in m["files"]}
        for seismic,mask in FROZEN_PAIRS:
            rec=by_name[seismic]
            target=out_dir/seismic
            state=fetch_streaming_checked(rec,target)
            seismic_head=header_only_npz(target)
            mask_head=l["files"][mask]["shape_headers_only"]
            if (len(seismic_head)!=1 or len(mask_head)!=1
                or seismic_head[0]["shape"]!=mask_head[0]["shape"]):
                raise RuntimeError(
                    f"STOP original Thebe {seismic} vs {mask} NPY array shape mismatch")
            if seismic_head[0]["dtype"] not in ("float32","float64"):
                raise RuntimeError("STOP unexpected seismic amplitude dtype")
            report["matched_pair_headers"][mask]={
                "seismic_file":seismic,
                "seismic_file_id":rec["id"],
                "source_checksum_MD5":rec["checksum_value"],
                "seismic_file_size":rec["bytes"],
                "seismic_SHA256":_digest(target,"sha256"),
                "how_verified":state,
                "seismic_NPY_header":seismic_head[0],
                "expert_fault_NPY_header":mask_head[0],
                "array_dimensions_exactly_equal":True,
                "seismic_data_pixels_not_read":True,
                "reserved_external_test_unopened":True,
            }
        report["status"]="V32_THEBE_TRAIN_VAL_REAL_SEISMIC_AND_EXPERT_MASK_HEADERS_MATCH_TEST_UNOPENED"
    except Exception as error:
        report["error"]=f"{type(error).__name__}: {error}"
    target=out_dir/"v32_real_thebe_train_val_seismic_head_receipt.json"
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "verified_paired_sources":list(report["matched_pair_headers"]),
        "matching_shapes":{k:v["seismic_NPY_header"]["shape"]
                           for k,v in report["matched_pair_headers"].items()},
        "actual_test_data_opened":False,
        "error":report.get("error"),
    },indent=2))
    return report


def _digest(p,algo):
    h=hashlib.new(algo)
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(2*1024*1024),b""):
            h.update(x)
    return h.hexdigest()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--label-receipt",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.manifest,a.label_receipt,a.out_dir)
