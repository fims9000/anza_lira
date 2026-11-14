"""Thebe Gas Field external-field file inventory via official Harvard Dataverse.

Thebe and CRACKS come from DIFFERENT real geological surveys. This is a
metadata-only provenance feasibility probe; DO NOT silently download a
53-GB dataset, open TEST labels, claim fault-instance identities, or treat
pixelwise fault masks as validated relationships between broken fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,urlopen

DATASET_PID="doi:10.7910/DVN/YBYGBK"
DATAVERSE_HOST="https://dataverse.harvard.edu"
DATASET_ENDPOINT=DATAVERSE_HOST+"/api/datasets/:persistentId/?"+urlencode(
    {"persistentId":DATASET_PID})
MAX_FIT_FILE_BYTES=1_500_000_000


def extract_files(doc):
    if doc.get("status")!="OK":
        raise RuntimeError("Dataverse API response does not report OK")
    data=doc.get("data") or {}
    version=data.get("latestVersion") or data.get("datasetVersion")
    if not isinstance(version,dict):
        raise RuntimeError("No published dataset version")
    items=version.get("files")
    if not isinstance(items,list) or not items:
        raise RuntimeError("Dataverse released dataset does not expose files")
    rows=[]
    for item in items:
        node=item.get("dataFile") or {}
        file_id=node.get("id")
        size=node.get("filesize")
        name=node.get("filename")
        if not isinstance(file_id,int) or file_id<=0:
            raise RuntimeError("Invalid Dataverse file ID")
        if not isinstance(size,int) or size<0:
            raise RuntimeError("Invalid Dataverse declared file size")
        if not isinstance(name,str) or not name:
            raise RuntimeError("Invalid Dataverse filename")
        checksum=node.get("checksum") or {}
        directory=str(item.get("directoryLabel") or "").strip("/")
        rows.append({
            "id":file_id,
            "name":name,
            "path":directory+"/"+name if directory else name,
            "bytes":size,
            "content_type":node.get("contentType"),
            "restricted":bool(item.get("restricted",node.get("restricted",False))),
            "checksum_algorithm":checksum.get("type"),
            "checksum_value":checksum.get("value"),
            "download_api":f"{DATAVERSE_HOST}/api/access/datafile/{file_id}",
            "eligible_for_single_runner_size":size<=MAX_FIT_FILE_BYTES,
        })
    if len({x["id"] for x in rows})!=len(rows):
        raise RuntimeError("Duplicate Dataverse source file IDs")
    return sorted(rows,key=lambda x:(x["path"].lower(),x["id"]))


def _category(filename):
    n=filename.lower()
    split=("train" if re.search(r"train",n) else
           "valid" if re.search(r"val|valid|dev",n) else
           "test" if re.search(r"test",n) else "other")
    kind=("fault_label" if re.search(r"fault|label|mask",n) else
          "seismic_image" if re.search(r"seismic|seis|image|amplitude",n) else
          "other")
    return split,kind


def summarize(doc):
    rows=extract_files(doc)
    groups={}
    for item in rows:
        split,kind=_category(item["name"])
        groups.setdefault(split,{}).setdefault(kind,[]).append(item["id"])
    dataset=doc.get("data") or {}
    version=dataset.get("latestVersion") or dataset.get("datasetVersion") or {}
    metadata=version.get("metadataBlocks") or {}
    citation=(metadata.get("citation") or {}).get("fields") or []
    citation_summary={}
    for field in citation:
        if field.get("typeName") in ("title","license","description","productionDate"):
            citation_summary[field["typeName"]]=field.get("value")
    safe_pairs=[
        split for split in ("train","valid","test")
        if groups.get(split,{}).get("fault_label")
        and groups.get(split,{}).get("seismic_image")
    ]
    return {
        "status":"V32_THEBE_EXTERNAL_FIELD_DATAVERSE_METADATA_ONLY_COMPLETE",
        "thebe_is_different_field_from_CRACKS_F3":True,
        "source_pid":DATASET_PID,
        "official_metadata_endpoint":DATASET_ENDPOINT,
        "file_count":len(rows),
        "declared_total_bytes":sum(r["bytes"] for r in rows),
        "max_file_size_bytes":max(r["bytes"] for r in rows),
        "files":rows,
        "source_grouping_by_FILENAME_HEURISTIC_only":groups,
        "split_names_with_both_seismic_and_fault_file_hints":safe_pairs,
        "candidate_small_files_total":sum(r["eligible_for_single_runner_size"] for r in rows),
        "copyright_license_not_guessed_from_third_party":True,
        "official_metadata_citation_fields":citation_summary,
        "archive_bytes_downloaded":0,
        "image_or_label_array_content_loaded":False,
        "seismic_fault_instance_IDs_verified":False,
        "heldout_test_outcomes_touched":False,
        "this_does_NOT_establish_matching_shapes_or_training_splits":True,
        "next_action":"Inspect at most one safe-size compatible pair after confirming documented filename meanings, then freeze external source sections/labels BEFORE model training.",
    }


def fetch():
    errors=[]
    for trial in range(3):
        try:
            req=Request(DATASET_ENDPOINT,headers={
                "User-Agent":"AnzaLiraV32ExternalScientificAudit/1.0",
                "Accept":"application/json"})
            with urlopen(req,timeout=75) as handle:
                raw=handle.read(9_000_000)
            return json.loads(raw)
        except (HTTPError,URLError,TimeoutError,ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if trial!=2:time.sleep(trial+1)
    raise RuntimeError("Dataverse official source metadata unavailable: "+"; ".join(errors))


def execute(out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=summarize(fetch())
    except Exception as error:
        report={
            "status":"V32_STOP_OFFICIAL_THEBE_SOURCE_METADATA_UNAVAILABLE",
            "source_pid":DATASET_PID,
            "official_metadata_endpoint":DATASET_ENDPOINT,
            "error":str(error),
            "archive_bytes_downloaded":0,
            "heldout_test_outcomes_touched":False,
        }
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":report["status"],"source_pid":DATASET_PID,
        "file_count":report.get("file_count"),
        "total_declared_bytes":report.get("declared_total_bytes"),
        "by_filename_hint":{k:{t:len(ids) for t,ids in v.items()}
                             for k,v in report.get(
                                 "source_grouping_by_FILENAME_HEURISTIC_only",{}).items()},
        "safe_file_count":report.get("candidate_small_files_total"),
        "downloaded_real_data":False,
        "error":report.get("error"),
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    execute(args.out)
