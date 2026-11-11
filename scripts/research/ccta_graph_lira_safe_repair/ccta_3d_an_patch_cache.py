#!/usr/bin/env python3
"""Build patient-isolated 3D CCTA candidate patch caches from CT28 scenes.

Only declared 17 train + optionally 5 validation patients can be processed.
The ImageCAS held-out test six are blocked even when using a different path.
The annotation label y is saved as a supervised target, but it never enters
the CT patch extraction function.

For fair comparisons the same compressed patch cache is used by both the
compact CNN and 3D ANZA candidate scorers. These models are not promoted into
the frozen K10/15 relation head without a later patient-level OOF protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccta_3d_an_patches import PatchConfig,extract_candidate_patch
from ccta_3d_structure_targets import sample_vessel_reference,load_aligned_train_reference
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS
import extract_ct28_scene_candidate_features as features
import run_ct28_graph_lira as graph


ROOT=Path(__file__).resolve().parents[3]
SCENES=ROOT/"artifacts/varvara/ct28_graph_lira"
DEFAULT_CACHE=ROOT/"varvara/anza_lira_local_results/ccta_3d_an_patches"
SEGMENTATIONS=ROOT/"varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations"


def digest(path:Path)->str:
    sha=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_verified_training_or_validation_ct(patient:int,archive:Path):
    if patient not in (EXPECTED_PATIENT_IDS["train"]|EXPECTED_PATIENT_IDS["val"]):
        raise ValueError("Held-out CT28 CCTA access forbidden in ANZA ablation")
    geom=pd.read_csv(features.EXPECTED_GEOMETRY).set_index("scan_id")
    align=pd.read_csv(features.EXPECTED_ALIGNMENT).set_index("scan_id")
    ct,inverse=graph.load_verified_ct(patient,archive,geom,align)
    return ct,np.linalg.inv(inverse)


def build_patient_cache(
    *,
    patient:int,
    scenes_dir:Path=SCENES,
    archive:Path=graph.DEFAULT_ARCHIVE_DIR,
    output_dir:Path=DEFAULT_CACHE,
    config:PatchConfig=PatchConfig(),
    include_val:bool=False,
    load_volume=None,
    with_vessel_targets:bool=False,
    segmentation_dir:Path=SEGMENTATIONS,
)->dict:
    train=EXPECTED_PATIENT_IDS["train"]
    val=EXPECTED_PATIENT_IDS["val"]
    if patient not in train and not (include_val and patient in val):
        raise ValueError("Only predeclared CT28 train patients allowed by default")
    split="train" if patient in train else "val"
    output=output_dir/f"patches_{patient}.npz"
    receipt=output_dir/f"patches_{patient}.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("Immutable patient patch cache already exists")

    plan_path=scenes_dir/"scene_plan_train_val.csv"
    endpoint_path=scenes_dir/"scene_endpoints_train_val.csv"
    candidate_path=scenes_dir/"scene_candidates_train_val.csv"
    plans=pd.read_csv(plan_path)
    endpoints=pd.read_csv(endpoint_path)
    candidates=pd.read_csv(candidate_path)
    correct=plans.loc[plans.scan_id.eq(patient)&plans.split.eq(split),
                      ["scene_id","scan_id","split"]]
    if len(correct)==0:
        raise ValueError("No frozen scenes for requested CT28 patient/split")
    rows=candidates.merge(correct,on="scene_id",validate="many_to_one")
    if len(rows)==0:
        raise ValueError("No candidates for patient")
    if not set(rows.candidate_type)<= {"PAIR","JUNCTION"} or not set(rows.y.astype(int))<= {0,1}:
        raise ValueError("Unknown target/candidate type")
    relevant=endpoints[endpoints.scene_id.isin(correct.scene_id)]
    lookup=relevant.set_index(["scene_id","endpoint_id"])
    if not lookup.index.is_unique:
        raise ValueError("Duplicate scene+endpoint key")
    ct,affine=(
        load_volume(patient)
        if load_volume is not None else
        load_verified_training_or_validation_ct(patient,archive)
    )
    vessel_source=None
    vessel_sha=None
    if with_vessel_targets:
        vessel_source,vessel_sha=load_aligned_train_reference(
            patient,ct.shape,affine,segmentation_dir,
            EXPECTED_PATIENT_IDS["train"],
        )
    vessel_targets=[]
    patches=[]
    labels=[]
    kinds=[]
    scene_keys=[]
    candidate_keys=[]
    inside_fracs=[]
    endpoint_margins=[]
    for row in rows.itertuples(index=False):
        endpoint_ids=str(row.endpoint_ids).split("|")
        if len(endpoint_ids)!=(2 if row.candidate_type=="PAIR" else 3):
            raise ValueError("Candidate degree does not match type")
        positions=np.stack([
            lookup.loc[(row.scene_id,endpoint_id),
                       ["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
            for endpoint_id in endpoint_ids
        ])
        patch,detail=extract_candidate_patch(ct,affine,positions,config)
        patches.append(patch.astype(np.float16))
        if with_vessel_targets:
            vessel_targets.append(sample_vessel_reference(
                vessel_source,affine,positions,config,
            ))
        labels.append(int(row.y))
        kinds.append(str(row.candidate_type))
        scene_keys.append(str(row.scene_id))
        candidate_keys.append(str(row.candidate_id))
        inside_fracs.append(detail["sampling_inside_volume_fraction"])
        endpoint_margins.append(detail["endpoint_patch_margin_mm"])
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        output,
        patches=np.stack(patches),
        y=np.asarray(labels,dtype=np.uint8),
        candidate_type=np.asarray(kinds,dtype="<U10"),
        scene_id=np.asarray(scene_keys,dtype="<U160"),
        candidate_id=np.asarray(candidate_keys,dtype="<U160"),
        inside_fraction=np.asarray(inside_fracs,dtype=np.float32),
        **({"vessel_targets":np.stack(vessel_targets).astype(np.uint8)}
           if with_vessel_targets else {}),
    )
    report={
        "phase":"supervised_candidate_patch_cache_train_or_val",
        "patient_id":patient,
        "split":split,
        "n_candidates":len(labels),
        "n_positives":sum(labels),
        "n_pair":sum(kind=="PAIR" for kind in kinds),
        "n_junction":sum(kind=="JUNCTION" for kind in kinds),
        "outside_fraction_over_20pct":sum(f<.8 for f in inside_fracs),
        "endpoint_outside_patch_count":sum(v<0 for v in endpoint_margins),
        "minimum_endpoint_patch_margin_mm":float(min(endpoint_margins)),
        "patch_protocol":vars(config),
        "scene_plan_sha256":digest(plan_path),
        "candidate_source_sha256":digest(candidate_path),
        "endpoint_source_sha256":digest(endpoint_path),
        "cache_sha256":digest(output),
        "ct_source":"verified local ImageCAS archive" if load_volume is None else "injected synthetic test volume",
        "held_out_test_accessed":False,
        "model_input_contains_ground_truth":False,
        "train_only_dense_supervision":bool(with_vessel_targets),
        "reference_segmentation_sha256":vessel_sha,
        "supervised_mask_is_separate_from_two_channel_network_input":True,
        "notes":"Candidate y saved separately as training target; not embedded into CT/heatmap patch",
    }
    receipt.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients",type=int,nargs="+",default=sorted(EXPECTED_PATIENT_IDS["train"]))
    parser.add_argument("--include-validation",action="store_true")
    parser.add_argument("--scene-dir",type=Path,default=SCENES)
    parser.add_argument("--archive-dir",type=Path,default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--patch-side",type=int,default=24)
    parser.add_argument("--voxel-mm",type=float,default=1.0)
    parser.add_argument("--frame-mode",choices=("canonical","fixed_lps"),default="canonical")
    parser.add_argument("--with-vessel-targets",action="store_true")
    parser.add_argument("--segmentation-dir",type=Path,default=SEGMENTATIONS)
    args=parser.parse_args()
    if len(args.patients)!=len(set(args.patients)):
        raise ValueError("Duplicate requested patient")
    config=PatchConfig(side=args.patch_side,voxel_mm=args.voxel_mm,frame_mode=args.frame_mode)
    for patient in args.patients:
        result=build_patient_cache(
            patient=patient,scenes_dir=args.scene_dir,archive=args.archive_dir,
            output_dir=args.out_dir,config=config,
            include_val=args.include_validation,
            with_vessel_targets=args.with_vessel_targets,
            segmentation_dir=args.segmentation_dir,
        )
        print(json.dumps({
            key:result[key] for key in (
                "patient_id","split","n_candidates","n_positives","cache_sha256"
            )
        }),flush=True)


if __name__=="__main__":
    main()
