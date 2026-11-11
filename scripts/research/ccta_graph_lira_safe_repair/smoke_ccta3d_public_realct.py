#!/usr/bin/env python3
"""Public cardiac CTA real-voxel smoke test; NOT an accuracy benchmark.

Checks exact CT->LPS/RAS geometry, normalized HU+endpoint patch extraction,
ANZA3D and CNN numeric forward/backward on the SAME real CT patches, finite
gradients, repeated score determinism, and outside-image support. Both models
are RANDOMLY INITIALIZED. No clinical or trained-model claims are possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from models.ccta3d_candidate_encoder import Candidate3dEncoder,parameter_budget
from ccta_3d_an_patches import PatchConfig,extract_candidate_patch


def read_ct(path:Path):
    import nrrd
    data,header=nrrd.read(str(path),index_order="F")
    if data.ndim!=3 or min(data.shape)<20:
        raise ValueError("Expected full 3D CT volume")
    space=str(header.get("space","")).strip().lower()
    if space in ("left-posterior-superior","lps"):
        to_ras=np.diag([-1.,-1.,1.])
    elif space in ("right-anterior-superior","ras"):
        to_ras=np.eye(3)
    else:
        raise ValueError(f"Unknown coordinate space {space}; refusing to guess affine")
    directions=np.asarray(header["space directions"],dtype=float)
    origin=np.asarray(header["space origin"],dtype=float)
    if directions.shape!=(3,3) or origin.shape!=(3,):
        raise ValueError("Invalid NRRD direction/space origin")
    if not np.isfinite(directions).all() or abs(np.linalg.det(directions))<1e-9:
        raise ValueError("Invalid physical CT axes")
    affine=np.eye(4)
    affine[:3,:3]=to_ras@directions.T
    affine[:3,3]=to_ras@origin
    return np.asarray(data,dtype=np.float32),affine,header,space


def run(ct_path:Path,output_dir:Path,side:int=16,device:str="cpu"):
    if output_dir.exists():
        raise FileExistsError("Never overwrite smoke-test evidence")
    if not ct_path.is_file():
        raise FileNotFoundError(ct_path)
    torch.set_num_threads(2)
    data,affine,header,space=read_ct(ct_path)
    raw_sha=hashlib.sha256(ct_path.read_bytes()).hexdigest()
    percentiles=np.percentile(data,[0,1,50,90,99,100]).tolist()
    # These are spatial *test points*, not medically annotated vessels.
    ijk=(np.asarray(data.shape)-1)/2
    center_ras=affine[:3,:3]@ijk+affine[:3,3]
    center_lps=center_ras*np.array([-1.,-1.,1.])
    endpoints={
        "PAIR":np.stack((center_lps+[-3.,0.,0.],center_lps+[3.,1.,0.])),
        "JUNCTION":np.stack((center_lps+[-3.,0.,0.],center_lps+[3.,1.,0.],center_lps+[0.,4.,1.])),
    }
    dev=torch.device(device)
    torch.manual_seed(20261111)
    models={x:Candidate3dEncoder(x).to(dev) for x in ("conv3d","anza3d")}
    capacity=parameter_budget(models["anza3d"],models["conv3d"])
    results=[]
    previews=[]
    for frame in ("canonical","fixed_lps"):
        for kind,points in endpoints.items():
            patch,audit=extract_candidate_patch(
                data,affine,points,PatchConfig(side=side,frame_mode=frame)
            )
            if patch.shape!=(2,side,side,side) or not np.isfinite(patch).all():
                raise AssertionError("Invalid 3D HU/heatmap patch")
            if not audit["all_endpoints_inside_patch"]:
                raise AssertionError("Chosen CT smoke points fell outside patch")
            if audit["sampling_inside_volume_fraction"]<.95:
                raise AssertionError("CT patch not sufficiently inside the physical scan")
            if patch[0].min()<-1.00001 or patch[0].max()>1.00001:
                raise AssertionError("HU normalization outside [-1,1]")
            inputs=torch.from_numpy(patch[None]).to(dev)
            for name,model in models.items():
                model.eval()
                started=time.perf_counter()
                with torch.no_grad():
                    a=model(inputs)
                    b=model(inputs)
                seconds=time.perf_counter()-started
                if a.numel()!=1 or not torch.isfinite(a).all() or not torch.equal(a,b):
                    raise AssertionError("Model produced nonfinite/nondeterministic CT output")
                entry={
                    "frame":frame,"candidate_type":kind,"model":name,
                    "random_initialized":True,"logit":float(a.item()),
                    "score":float(a.sigmoid().item()),
                    "inference_two_pass_seconds":seconds,
                    "sampling_inside_volume_fraction":audit["sampling_inside_volume_fraction"],
                    "endpoint_patch_margin_mm":audit["endpoint_patch_margin_mm"],
                    "normalized_hu_minmax":[float(patch[0].min()),float(patch[0].max())],
                    "heatmap_nonzero":int(np.count_nonzero(patch[1])),
                }
                if kind=="PAIR" and frame=="canonical":
                    model.train()
                    model.zero_grad(set_to_none=True)
                    started=time.perf_counter()
                    out=model(inputs)
                    loss=F.binary_cross_entropy_with_logits(
                        out,torch.ones_like(out)
                    )
                    loss.backward()
                    grads=[p.grad for p in model.parameters() if p.requires_grad]
                    entry["train_step_loss"]=float(loss.item())
                    entry["backward_seconds"]=time.perf_counter()-started
                    entry["all_parameter_grads_finite"]=all(
                        g is not None and torch.isfinite(g).all() for g in grads
                    )
                    entry["total_gradient_norm"]=float(
                        sum(g.abs().sum().item() for g in grads if g is not None)
                    )
                    if not entry["all_parameter_grads_finite"] or entry["total_gradient_norm"]<=0:
                        raise AssertionError("Invalid model gradients on real CT")
                results.append(entry)
            if kind=="PAIR":
                previews.append((frame,patch[0]))
    output_dir.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    for frame,patch in previews:
        # normalized HU max-intensity projection of a cropped, public CT test patch
        z=(np.clip(patch.max(axis=2),-1.,1.)+1.)*127.5
        Image.fromarray(z.astype(np.uint8)).save(output_dir/f"{frame}_ct_patch_mip.png")
    result={
        "status":"real_cardio_ct_smoke_pass",
        "scope":"single unannotated public CTA volume, numeric and I/O verification only",
        "data_source":"3D Slicer CTA-cardio public testing sample",
        "raw_sha256":raw_sha,
        "nrrd_space":space,
        "ct_shape":list(data.shape),
        "affine_voxel_to_ras":affine.tolist(),
        "ct_value_percentiles":[float(x) for x in percentiles],
        "patch_side":side,
        "device":device,
        "capacity_match":capacity,
        "models_random_initialized":True,
        "no_clinical_coronary_labels":True,
        "no_trained_ct28_checkpoints":True,
        "medical_accuracy_claimed":False,
        "heldout_ct28_test_accessed":False,
        "results":results,
    }
    (output_dir/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--volume",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--side",type=int,default=16)
    parser.add_argument("--device",default="cpu")
    a=parser.parse_args()
    run(a.volume,a.out,side=a.side,device=a.device)
