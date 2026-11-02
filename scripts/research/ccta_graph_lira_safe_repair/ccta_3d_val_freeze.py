#!/usr/bin/env python3
"""Immutable pre-declared CT28 3D *development-validation* protocol.

Freezes one full-volume experiment before any five validation volumes are
processed by this pipeline. No held-out test patients are accepted.

This is NOT a final blind test protocol: K10/15 was already selected using
these five validation subjects in the historical scene-level study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ccta_3d_k10 import K10_15_SHA256
from ccta_3d_cli import DEFAULT_PATH
from ccta_3d_train_pilot import SELECTED_PACK
from ccta_3d_train_gap_simulator import VAL_CT28_IDS
from extract_ct28_scene_candidate_features import EXPECTED_ALIGNMENT


FREEZE_VERSION="ct28_fullvolume_development_validation_v1"
DEFAULT_SETTINGS={
    "controlled_gaps":3,
    "gap_radius_mm":1.5,
    "gap_seed":20261102,
    "repair_radius_mm":0.0,
    "max_new_voxels_per_decision":250,
    "min_component_voxels":12,
    "trace_mm":3.0,
    "max_endpoints":256,
    "max_scenes":128,
    "strict_count_support":False,
}


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda:reader.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_freeze(
    *,
    output:Path,
    model:Path=SELECTED_PACK,
    path_protocol:Path=DEFAULT_PATH,
    required_model_sha:str=K10_15_SHA256,
)->dict:
    if output.exists() or output.suffix!=".json":
        raise FileExistsError("Validation freeze output must be a new JSON")
    if output.resolve() in {model.resolve(),path_protocol.resolve()}:
        raise ValueError("Never overwrite frozen model or path protocol")
    if sha(model)!=required_model_sha:
        raise ValueError("Selected baseline K10/15 SHA mismatch")
    if not path_protocol.is_file():
        raise ValueError("Selected CT path-v2 protocol missing")
    align=pd.read_csv(EXPECTED_ALIGNMENT).set_index("scan_id")
    if set(VAL_CT28_IDS)-set(align.index.astype(int)):
        raise ValueError("Frozen CT28 validation CT SHA inventory incomplete")
    expected={
        str(scan_id):str(align.loc[scan_id,"ct_sha256"])
        for scan_id in sorted(VAL_CT28_IDS)
    }
    protocol={
        "status":"development_validation_predeclared_not_blind_test",
        "version":FREEZE_VERSION,
        "patients":sorted(VAL_CT28_IDS),
        "patient_count":5,
        "held_out_test_accessed":False,
        "settings":dict(DEFAULT_SETTINGS),
        "selected_K10_SHA256":required_model_sha,
        "selected_path_protocol_sha256":sha(path_protocol),
        "expected_validation_CT_SHA256":expected,
        "model_selection_warning":(
            "Selected K10/15 was previously chosen on this validation cohort. "
            "Full-mask development-val results are NOT an independent held-out claim."
        ),
        "evaluation_rule":(
            "Run frozen parameters without changing from any validation observations. "
            "Ground truth is for synthetic damage generation and posthoc only."
        ),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    return protocol


def verify_validation_freeze(
    path:Path,
    *,
    model:Path=SELECTED_PACK,
    path_protocol:Path=DEFAULT_PATH,
    required_model_sha:str=K10_15_SHA256,
)->dict:
    protocol=json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("status")!="development_validation_predeclared_not_blind_test"
        or protocol.get("version")!=FREEZE_VERSION
        or protocol.get("patients")!=sorted(VAL_CT28_IDS)
        or protocol.get("settings")!=DEFAULT_SETTINGS
        or protocol.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Development-validation protocol is not the expected immutable freeze")
    if (
        sha(model)!=required_model_sha
        or protocol.get("selected_K10_SHA256")!=required_model_sha
        or protocol.get("selected_path_protocol_sha256")!=sha(path_protocol)
    ):
        raise ValueError("Frozen selected model or path-v2 protocol SHA changed")
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model-pack",type=Path,default=SELECTED_PACK)
    parser.add_argument("--path-protocol",type=Path,default=DEFAULT_PATH)
    args=parser.parse_args()
    report=build_validation_freeze(
        output=args.output,model=args.model_pack,path_protocol=args.path_protocol,
    )
    print(json.dumps({
        "status":report["status"],
        "patient_count":report["patient_count"],
        "selected_K10_SHA256":report["selected_K10_SHA256"],
    },indent=2))


if __name__=="__main__":
    main()
