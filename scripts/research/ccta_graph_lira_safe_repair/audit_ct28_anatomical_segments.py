#!/usr/bin/env python3
"""Audit whether original ImageCAS-X multi-label branches help PAIR/JUNCTION.

Read-only TRAIN-only study. Requires verified local ImageCAS CCTA and exact
ImageCAS-X corresponding native segmentation. Anatomical segment IDs must
NEVER be injected into validation/test feature inputs or used as a direct
truth rule: different correctly linked segments meet at bifurcations.
Results are saved under the ignored local experiment directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from ccta_3d_an_patch_cache import SCENES,SEGMENTATIONS,load_verified_training_or_validation_ct
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS
import run_ct28_graph_lira as graph


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _segments_for_endpoints(
    patient:int,locations_lps:np.ndarray,
    *,segmentations:Path,archive:Path,
)->tuple[np.ndarray,dict]:
    if patient not in EXPECTED_PATIENT_IDS["train"]:
        raise ValueError("Anatomical segment audit is restricted to CT28 train IDs")
    path=segmentations/f"{patient}.coronary.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    # Reuse the existing ImageCAS archive/affine SHA verification. The matching
    # CT is not copied, exposed to the model or written to the report.
    ct,affine=load_verified_training_or_validation_ct(patient,archive)
    image=nib.load(str(path))
    if tuple(image.shape)!=tuple(ct.shape):
        raise ValueError("ImageCAS-X segment mask and verified CT shapes differ")
    if not np.allclose(image.affine,affine,rtol=0,atol=1e-6):
        raise ValueError("Segment mask affine differs from verified CCTA")
    label_image=np.asanyarray(image.dataobj)
    if not np.isfinite(label_image).all():
        raise ValueError("Nonfinite image segmentation")
    if (np.abs(label_image-np.rint(label_image))>1e-5).any():
        raise ValueError("Expected discrete anatomical segment labels")
    unique=np.unique(label_image)
    if unique.min()<0 or unique.max()>14:
        raise ValueError("ImageCAS-X anatomical label outside declared 0..14")
    if locations_lps.ndim!=2 or locations_lps.shape[1]!=3:
        raise ValueError("Expected endpoint LPS positions Nx3")
    ras=locations_lps*np.array([-1.,-1.,1.],dtype=float)
    indices=np.rint(
        (np.linalg.inv(affine)[:3,:3]@ras.T+
         np.linalg.inv(affine)[:3,3,None]).T
    ).astype(int)
    within=np.all((indices>=0)&(indices<np.asarray(ct.shape)[None,:]),axis=1)
    output=np.full(len(indices),-1,dtype=int)
    for idx,ijk in enumerate(indices):
        if within[idx]:
            output[idx]=int(label_image[tuple(ijk)])
    return output,{
        "patient_id":patient,
        "segment_mask_sha256":_sha256(path),
        "ct_shape":list(ct.shape),
        "endpoints":len(indices),
        "out_of_volume":int((~within).sum()),
        "zero_background":int(np.sum(output==0)),
        "segment_labels_present":[int(v) for v in unique if v>0],
    }


def audit(*,scene_dir:Path,archive_dir:Path,segmentations:Path,
          patients:list[int],output:Path)->dict:
    if not patients or len(set(patients))!=len(patients):
        raise ValueError("Nonempty distinct train patient IDs required")
    if not set(patients)<=EXPECTED_PATIENT_IDS["train"]:
        raise ValueError("Validation or held-out CT28 patient forbidden in branch audit")
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError("Branch-ID audit results are immutable")
    plans=pd.read_csv(scene_dir/"scene_plan_train_val.csv")
    endpoints=pd.read_csv(scene_dir/"scene_endpoints_train_val.csv")
    candidates=pd.read_csv(scene_dir/"scene_candidates_train_val.csv")
    grouped_plan=plans.loc[
        plans.split.eq("train") & plans.scan_id.isin(patients),
        ["scene_id","scan_id"],
    ]
    if not set(patients)<=set(grouped_plan.scan_id.astype(int)):
        raise ValueError("Not all requested patients have frozen train scenes")
    filtered_endpoints=endpoints.merge(grouped_plan,on="scene_id",validate="many_to_one")
    filtered_candidates=candidates.merge(grouped_plan,on="scene_id",validate="many_to_one")
    annotations={}
    receipts=[]
    for patient in sorted(patients):
        rows=filtered_endpoints.loc[filtered_endpoints.scan_id.eq(patient)]
        result,receipt=_segments_for_endpoints(
            patient,rows[["x_lps","y_lps","z_lps"]].to_numpy(dtype=float),
            segmentations=segmentations,archive=archive_dir,
        )
        receipts.append(receipt)
        if len(result)!=len(rows):
            raise AssertionError("Endpoint segment sample count mismatch")
        for endpoint,label in zip(rows.itertuples(index=False),result):
            key=(str(endpoint.scene_id),str(endpoint.endpoint_id))
            if key in annotations:
                raise ValueError("Duplicate scene-endpoint identity")
            annotations[key]=int(label)
    records=[]
    for row in filtered_candidates.itertuples(index=False):
        ids=str(row.endpoint_ids).split("|")
        if len(ids)!=(2 if row.candidate_type=="PAIR" else 3):
            raise ValueError("Unexpected relation degree")
        values=[annotations[(str(row.scene_id),e)] for e in ids]
        valid=all(v>0 for v in values)
        records.append({
            "scene_id":row.scene_id,"scan_id":int(row.scan_id),
            "candidate_id":row.candidate_id,"candidate_type":row.candidate_type,
            "y":int(row.y),"segments":"|".join(map(str,values)),
            "all_labeled":bool(valid),
            "same_segment":bool(valid and len(set(values))==1),
            "n_distinct_segments":len(set(values)) if valid else -1,
            "n_background_or_missing":sum(v<=0 for v in values),
        })
    frame=pd.DataFrame.from_records(records)
    groups={}
    for (kind,label),part in frame.groupby(["candidate_type","y"],sort=True):
        clean=part.loc[part.all_labeled]
        groups[f"{kind}/y={int(label)}"]={
            "n":len(part),"all_labeled_n":len(clean),
            "fraction_any_unlabeled":float((~part.all_labeled).mean()),
            "same_segment_fraction_among_labeled":(
                float(clean.same_segment.mean()) if len(clean) else None
            ),
            "distinct_segment_count_histogram":{
                str(int(k)):int(v)
                for k,v in clean.n_distinct_segments.value_counts().items()
            },
        }
    report={
        "status":"train_only_anatomical_segment_identity_audit",
        "train_patients":sorted(patients),
        "held_out_test_accessed":False,
        "validation_patient_accessed":False,
        "no_model_feature_leakage":True,
        "warning":"Same segment label is NOT a valid standalone PAIR/JUNCTION ground-truth rule",
        "candidate_n":len(frame),
        "groups":groups,
        "source_sha256":{
            p.name:_sha256(p) for p in (
                scene_dir/"scene_plan_train_val.csv",
                scene_dir/"scene_endpoints_train_val.csv",
                scene_dir/"scene_candidates_train_val.csv",
            )
        },
        "patient_source_receipts":receipts,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    frame.to_csv(output.with_suffix(".csv"),index=False)
    print(json.dumps({"status":report["status"],"groups":groups,
                      "candidate_n":report["candidate_n"]},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir",type=Path,default=SCENES)
    p.add_argument("--archive-dir",type=Path,default=graph.DEFAULT_ARCHIVE_DIR)
    p.add_argument("--segmentation-dir",type=Path,default=SEGMENTATIONS)
    p.add_argument("--patients",type=int,nargs="+",default=sorted(EXPECTED_PATIENT_IDS["train"]))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    audit(scene_dir=args.scenes_dir,archive_dir=args.archive_dir,
          segmentations=args.segmentation_dir,patients=args.patients,
          output=args.out)


if __name__=="__main__":
    main()
