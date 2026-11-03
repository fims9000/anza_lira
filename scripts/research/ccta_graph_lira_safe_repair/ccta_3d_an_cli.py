#!/usr/bin/env python3
"""Experimental ANZA/CNN full-volume CCTA Graph-LIRA inference.

Requires a verified selected K10/15 base pack, two patient-OOF neural CT
candidate evidence packs (PAIR/JUNCTION), and their clean-retrained 84-feature
relation head. The 15-perturbation gate runs but the neural relation head is NOT
augmentation-matched to selected K10/15. This is not a clinical or final test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccta_3d_an_runtime import NeuralCCTA3dInference
from ccta_3d_cli import run_cli,DEFAULT_PATH
from ccta_3d_runtime import HELD_OUT_CT28_IDS


ROOT=Path(__file__).resolve().parents[3]
K10_PACK=ROOT/(
    "varvara/anza_lira_local_results/ct28_copy_count_ablations/"
    "joint_check/fusion_10_relation_15/relation_models.joblib"
)


def run_neural(
    *,
    ct:Path,
    mask:Path,
    patient_id:str,
    base_k10:Path,
    relation_dir:Path,
    pair_oof:Path,
    junction_oof:Path,
    variant:str,
    output_mask:Path,
    report:Path,
    output_additions:Path|None=None,
    device:str="cpu",
    path_protocol:Path=DEFAULT_PATH,
    radius_mm:float=0.,
    max_new_voxels:int=250,
    strict_count_support:bool=False,
    pair_identity_source:str="frozen_fusion",
)->dict:
    if str(patient_id).strip() in HELD_OUT_CT28_IDS:
        raise ValueError("Graph-LIRA held-out test remains closed")
    engine=NeuralCCTA3dInference.from_experiment(
        k10_pack=base_k10,relation_dir=relation_dir,
        pair_oof_dir=pair_oof,junction_oof_dir=junction_oof,
        variant=variant,device=device,
        pair_identity_source=pair_identity_source,
    )
    evidence={
        "relation_dir":str(relation_dir),
        "pair_oof":str(pair_oof),
        "junction_oof":str(junction_oof),
        "variant":variant,
        "pair_identity_source":pair_identity_source,
        "not_frozen_K10_relation_head":True,
        "note":"neural CT scores and clean relation head; 15x consistency exploratory",
    }
    outcome=run_cli(
        ct_path=ct,mask_path=mask,patient_id=patient_id,
        mode="k10",model_pack=base_k10,
        output_mask=output_mask,report_path=report,
        output_additions=output_additions,
        radius_mm=radius_mm,max_new_voxels=max_new_voxels,
        min_component_voxels=12,trace_mm=3.,
        max_endpoints=256,max_scenes=128,threshold=.5,
        path_protocol=path_protocol,
        strict_count_support=strict_count_support,
        engine_override=engine,
        experimental_evidence=evidence,
    )
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct",type=Path,required=True)
    parser.add_argument("--mask",type=Path,required=True)
    parser.add_argument("--patient-id",required=True)
    parser.add_argument("--base-k10",type=Path,default=K10_PACK)
    parser.add_argument("--relation-dir",type=Path,required=True)
    parser.add_argument("--pair-oof",type=Path,required=True)
    parser.add_argument("--junction-oof",type=Path,required=True)
    parser.add_argument("--variant",choices=("conv3d","anza3d"),required=True)
    parser.add_argument("--pair-identity-source",choices=("frozen_fusion","neural_pair"),default="frozen_fusion")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--path-protocol",type=Path,default=DEFAULT_PATH)
    parser.add_argument("--output-mask",type=Path,required=True)
    parser.add_argument("--output-additions",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--radius-mm",type=float,default=0.)
    parser.add_argument("--max-new-voxels",type=int,default=250)
    parser.add_argument("--strict-count-support",action="store_true")
    args=parser.parse_args()
    result=run_neural(
        ct=args.ct,mask=args.mask,patient_id=args.patient_id,
        base_k10=args.base_k10,relation_dir=args.relation_dir,
        pair_oof=args.pair_oof,junction_oof=args.junction_oof,
        variant=args.variant,output_mask=args.output_mask,
        output_additions=args.output_additions,report=args.report,
        device=args.device,path_protocol=args.path_protocol,
        radius_mm=args.radius_mm,max_new_voxels=args.max_new_voxels,
        strict_count_support=args.strict_count_support,
        pair_identity_source=args.pair_identity_source,
    )
    print(json.dumps({
        "status":result["status"],
        "encoder":result["experimental_evidence"]["variant"],
        "scenes":result["scene_count"],
        "accepted_repairs":result["accepted_decisions"],
        "added_voxels":result["added_voxels"],
    },indent=2))


if __name__=="__main__":
    main()
