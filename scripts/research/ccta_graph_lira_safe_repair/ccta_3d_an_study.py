#!/usr/bin/env python3
"""Reproducible ANZA-vs-CNN study on CT28 TRAIN patients with 3D mask outputs.

One command can prepare caches, OOF scores, train clean relation HGB heads,
create a controlled-damage K10 pilot, run both neural 3D CCTA models and
measure paired Dice/clDice on the SAME damaged input. Every stage is create-
only and resumable only when previous artifacts pass basic checksum checks.

This is a development study, NOT an independent clinical/test evaluation.
It does not open any of CT28 held-out test patients.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ccta_3d_an_patch_cache import build_patient_cache, DEFAULT_CACHE
from ccta_3d_an_patches import PatchConfig
from ccta_3d_an_oof import build_oof, sha256
from ccta_3d_an_relation import run_relation_training
from ccta_3d_an_cli import run_neural
from ccta_3d_train_pilot import execute_train_pilot, SELECTED_PACK, DEFAULT_ROOT
from ccta_3d_evaluate import evaluate_files
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


TRAIN=sorted(EXPECTED_PATIENT_IDS["train"])
VAL=sorted(EXPECTED_PATIENT_IDS["val"])
DEFAULT_STUDY=DEFAULT_ROOT/"anza_vs_cnn_development"


def _load_json(path:Path)->dict:
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def _verified_cache(cache:Path,patient:int,frame_mode:str="canonical",with_vessel_targets:bool=False,endpoint_encoding:str="summed")->bool:
    path=cache/f"patches_{patient}.npz"
    report=cache/f"patches_{patient}.json"
    if not path.exists() and not report.exists():
        return False
    if not path.is_file() or not report.is_file():
        raise ValueError(f"Partial cached CT patch evidence for patient {patient}")
    meta=_load_json(report)
    split="train" if patient in TRAIN else "val"
    if meta.get("patient_id")!=patient or meta.get("split")!=split or meta.get("cache_sha256")!=sha256(path):
        raise ValueError(f"Corrupted or patient/split-mismatched CT patch cache {patient}")
    expected=vars(PatchConfig(frame_mode=frame_mode,endpoint_encoding=endpoint_encoding))
    recorded=meta.get("patch_protocol") or {}
    for field,value in expected.items():
        saved=recorded.get(field,"summed" if field=="endpoint_encoding" else None)
        if saved!=value:
            raise ValueError(
                f"CT patch protocol {field} mismatch for patient {patient}: "
                f"requested {value!r}, cached {saved!r}"
            )
    if patient in TRAIN and bool(meta.get("train_only_dense_supervision",False))!=with_vessel_targets:
        raise ValueError(f"Train-only vessel target protocol mismatch for patient {patient}")
    if patient in VAL and meta.get("train_only_dense_supervision",False):
        raise ValueError("Validation vessel targets must not enter the candidate cache")
    return True


def _verified_oof(path:Path,kind:str,variant:str)->bool:
    if not path.exists():
        return False
    report=path/"oof_protocol.json"
    scores=path/f"{kind}_{variant}_oof_val.csv"
    if not report.exists() or not scores.exists():
        raise ValueError(f"Partial OOF model evidence: {path}")
    m=_load_json(report)
    if m.get("status")!="train_patient_OOF_experimental_CT_evidence_not_promoted_K10" or (
        m.get("variant"),m.get("candidate_type")
    )!=(variant,kind) or m.get("score_sha256")!=sha256(scores):
        raise ValueError(f"OOF model score protocol changed: {path}")
    if m.get("n_train_patients")!=17 or m.get("n_validation_patients")!=5 or m.get("held_out_test_accessed") is not False:
        raise ValueError("Invalid train/val OOF cohort")
    if m.get("source_input_control_mode","full_ct")!="full_ct":
        raise ValueError("Cannot reuse a negative-control encoder for main 3D CCTA repair study")
    checkpoint=path/f"{kind}_{variant}_fulltrain.pt"
    if not checkpoint.exists() or sha256(checkpoint)!=m.get("fulltrain_model_sha256"):
        raise ValueError("OOF neural model weights changed")
    return True


def _relation_protocol(mode:str)->str:
    statuses={
        "clean":"experimental_clean_relation_head_no_consistency_gate",
        "augmented_15":"experimental_patient_OOF_15copy_augmented_relation_head",
    }
    if mode not in statuses:
        raise ValueError("Expected clean or augmented_15 CT28 relation training mode")
    return statuses[mode]


def _verified_relation(path:Path,variant:str,mode:str="clean")->bool:
    status=_relation_protocol(mode)
    if not path.exists():
        return False
    manifest=path/"relation_protocol.json"
    if not manifest.is_file():
        raise ValueError(f"Partial relation-model evidence: {path}")
    m=_load_json(manifest)
    if m.get("status")!=status or m.get("variant")!=variant:
        raise ValueError("Wrong experimental relation variant or augmentation mode")
    name="clean_relation_head.joblib" if mode=="clean" else "augmented_relation_head.joblib"
    predictions=(
        "clean_relation_predictions_val.csv" if mode=="clean"
        else "augmented_relation_predictions_val.csv"
    )
    model=path/name
    pred=path/predictions
    if not all(x.is_file() for x in (model,pred)):
        raise ValueError(f"Partial relation-model evidence: {path}")
    if m.get("model_sha256")!=sha256(model) or m.get("predictions_sha256")!=sha256(pred):
        raise ValueError("Relation model/prediction hash mismatch")
    if mode=="augmented_15" and (
        m.get("relation_training_copies_per_scene")!=15
        or m.get("train_scenes")!=1036
        or m.get("validation_scenes")!=368
        or m.get("augmented_train_scenes")!=1036*15
        or m.get("held_out_test_accessed") is not False
    ):
        raise ValueError("The 15-copy relation training protocol is incomplete")
    return True


def ensure_patch_cache(cache:Path,frame_mode:str="canonical",with_vessel_targets:bool=False,endpoint_encoding:str="summed")->None:
    for patient in TRAIN+VAL:
        if _verified_cache(cache,patient,frame_mode=frame_mode,with_vessel_targets=with_vessel_targets,endpoint_encoding=endpoint_encoding):
            continue
        build_patient_cache(patient=patient,output_dir=cache,
                            config=PatchConfig(frame_mode=frame_mode,endpoint_encoding=endpoint_encoding),
                            include_val=patient in VAL,
                            with_vessel_targets=with_vessel_targets and patient in TRAIN)
        print(f"CT28 3D patches complete: {patient}",flush=True)


def ensure_oof(cache:Path,root:Path,*,epochs:int,device:str,folds:int,batch_size:int,frame_mode:str="canonical",seed:int=20261102,variants:tuple[str,...]=("conv3d","anza3d"),endpoint_encoding:str="summed",ranking_weight:float=0.)->None:
    structure=all(v.endswith("_structure") for v in variants)
    if any(("_sep_" in variant)!=(endpoint_encoding=="separate") for variant in variants):
        raise ValueError("Model variant and patch endpoint-encoding mismatch")
    if not all(_verified_cache(cache,p,frame_mode=frame_mode,with_vessel_targets=structure,endpoint_encoding=endpoint_encoding) for p in TRAIN+VAL):
        raise ValueError("Frozen patient patch cache incomplete for requested frame mode")
    for kind in ("PAIR","JUNCTION"):
        for variant in variants:
            out=root/f"oof_{kind.lower()}_{variant}"
            if _verified_oof(out,kind,variant):
                meta=_load_json(out/"oof_protocol.json")
                expected={"epochs":epochs,"batch_size":batch_size,"seed":seed,"device":device,"ranking_weight":ranking_weight}
                actual=meta.get("fixed_train_hyperparameters",{})
                if meta.get("fold_count")!=folds or any(actual.get(k,0. if k=="ranking_weight" else None)!=v for k,v in expected.items()):
                    raise ValueError("Existing OOF run uses different seed, folds, device or training budget")
                continue
            build_oof(
                cache_dir=cache,out_dir=out,candidate_type=kind,
                variant=variant,train_patients=TRAIN,val_patients=VAL,
                folds=folds,epochs=epochs,batch_size=batch_size,
                seed=seed,device=device,ranking_weight=ranking_weight,
            )
            print(f"Patient-OOF candidate evidence: {kind}/{variant}",flush=True)


def verify_upstream_ranking_protocol(root:Path,variants:tuple[str,...],expected_weight:float)->None:
    """Refuse relations/volume generated with stale candidate ranking objective."""
    if not math.isfinite(expected_weight) or expected_weight<0:
        raise ValueError("Invalid train-only ranking weight")
    for variant in variants:
        for kind in ("PAIR","JUNCTION"):
            output=root/f"oof_{kind.lower()}_{variant}"
            if not _verified_oof(output,kind,variant):
                raise ValueError(f"Missing verified upstream OOF for {kind}/{variant}")
            meta=_load_json(output/"oof_protocol.json")
            weight=meta.get("fixed_train_hyperparameters",{}).get("ranking_weight",0.)
            if weight!=expected_weight:
                raise ValueError(
                    f"OOF {kind}/{variant} ranking-weight mismatch: "
                    f"{weight} != {expected_weight}"
                )


def ensure_relations(root:Path,*,mode:str="clean",device:str="cpu",variants:tuple[str,...]=("conv3d","anza3d"))->None:
    _relation_protocol(mode)
    for variant in variants:
        path=root/f"relation_{variant}"
        if _verified_relation(path,variant,mode):
            continue
        options={
            "pair_oof":root/f"oof_pair_{variant}",
            "junction_oof":root/f"oof_junction_{variant}",
            "variant":variant,
            "output_dir":path,
        }
        if mode=="clean":
            run_relation_training(**options)
        else:
            from ccta_3d_an_relation_augmented import train_augmented_neural_relation
            train_augmented_neural_relation(**options,device=device)
        print(f"{mode} 84-feature relation trained: {variant}",flush=True)


def ensure_pilot(root:Path,*,patient:int)->Path:
    if patient not in TRAIN:
        raise ValueError("Full-volume ANZA comparison requires a train patient")
    output=root/f"pilot_{patient}"
    if output.exists():
        manifest=output/"pilot_manifest.json"
        if not manifest.exists():
            raise ValueError("Partial previous K10 pilot exists")
        meta=_load_json(manifest)
        if meta.get("status")!="completed_experimental_train_pilot" or meta.get("patient_id")!=patient or meta.get("mode")!="k10":
            raise ValueError("Wrong previous K10 patient pilot")
    else:
        execute_train_pilot(
            scan_id=patient,output_dir=output,mode="k10",
            gaps=3,gap_radius_mm=1.5,
            repair_radius_mm=0.,max_new_voxels=250,
        )
    return output


def ensure_neural_inference(root:Path,*,pilot:Path,patient:int,device:str,
                            pair_identity_source:str="frozen_fusion",
                            relation_mode:str="clean",
                            variants:tuple[str,...]=("conv3d","anza3d"))->None:
    expected_protocol=_relation_protocol(relation_mode)
    for variant in variants:
        repaired=root/f"{variant}_repaired.nii.gz"
        additions=root/f"{variant}_additions.nii.gz"
        decisions=root/f"{variant}_decisions.json"
        metrics=root/f"{variant}_posthoc.json"
        artifacts=(repaired,additions,decisions,metrics)
        if any(path.exists() for path in artifacts):
            if not all(path.is_file() for path in artifacts):
                raise ValueError(f"Partial neural volume evidence for {variant}")
            provenance=_load_json(decisions)
            if provenance.get("experimental_evidence",{}).get("variant")!=variant or provenance.get("patient_label")!=str(patient) or provenance.get("experimental_evidence",{}).get("pair_identity_source","frozen_fusion")!=pair_identity_source or provenance.get("experimental_evidence",{}).get("relation_train_protocol",_relation_protocol("clean"))!=expected_protocol:
                raise ValueError("Wrong neural volume inference provenance")
            evaluation=_load_json(metrics)
            if evaluation.get("patient_id")!=str(patient):
                raise ValueError("Wrong posthoc evaluation patient")
            continue
        if not _verified_relation(root/f"relation_{variant}",variant,relation_mode):
            raise ValueError("Neural relation head for requested augmentation mode is missing")
        run_neural(
            ct=pilot/"ct.nii.gz",
            mask=pilot/"broken_mask.nii.gz",
            patient_id=str(patient),
            base_k10=SELECTED_PACK,
            relation_dir=root/f"relation_{variant}",
            pair_oof=root/f"oof_pair_{variant}",
            junction_oof=root/f"oof_junction_{variant}",
            variant=variant,output_mask=repaired,
            output_additions=additions,
            report=decisions,device=device,
            radius_mm=0.,max_new_voxels=250,
            pair_identity_source=pair_identity_source,
        )
        evaluate_files(
            patient_id=str(patient),
            before=pilot/"broken_mask.nii.gz",
            after=repaired,
            reference=(
                Path(__file__).resolve().parents[3]/
                f"varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations/{patient}.coronary.nii.gz"
            ),
            output=metrics,
        )
        print(f"3D neural CT inference and Dice/clDice computed: {variant}",flush=True)


def summarize(root:Path,*,pilot:Path,patient:int,
              pair_identity_source:str="frozen_fusion",
              relation_mode:str="clean",
              variants:tuple[str,...]=("conv3d","anza3d"))->dict:
    expected_protocol=_relation_protocol(relation_mode)
    k10_metrics=_load_json(pilot/"posthoc_metrics.json")
    inputs=_load_json(pilot/"decisions.json")
    records={}
    for variant in ("K10",*variants):
        m=k10_metrics if variant=="K10" else _load_json(root/f"{variant}_posthoc.json")
        decisions=inputs if variant=="K10" else _load_json(root/f"{variant}_decisions.json")
        if decisions.get("patient_label")!=str(patient) or m.get("patient_id")!=str(patient):
            raise ValueError(f"Comparative segmentation evidence patient mismatch: {variant}")
        if not all(isinstance(m.get(key),(int,float)) for key in (
            "dice_before","dice_after","dice_delta",
            "cldice_before","cldice_after","cldice_delta",
        )):
            raise ValueError(f"Missing paired 3D mask metrics for {variant}")
        if variant!="K10":
            if decisions.get("experimental_evidence",{}).get("variant")!=variant or decisions.get("experimental_evidence",{}).get("pair_identity_source","frozen_fusion")!=pair_identity_source:
                raise ValueError("Neural encoder/PAIR identity provenance mismatch")
            if decisions.get("experimental_evidence",{}).get("relation_train_protocol",_relation_protocol("clean"))!=expected_protocol:
                raise ValueError("Neural relation training mode provenance mismatch")
            if m["before_sha256"]!=k10_metrics["before_sha256"]:
                raise ValueError("Not the same broken 3D segmentation input")
            if m["reference_sha256"]!=k10_metrics["reference_sha256"]:
                raise ValueError("Not the same reference volume in posthoc comparison")
        records[variant]={
            "accepted_repairs":decisions["accepted_decisions"],
            "added_voxels":decisions["added_voxels"],
            "dice_before":m["dice_before"],
            "dice_after":m["dice_after"],
            "delta_dice":m["dice_delta"],
            "cldice_before":m["cldice_before"],
            "cldice_after":m["cldice_after"],
            "delta_cldice":m["cldice_delta"],
            "reference_unsupported_added_voxels":m["reference_unsupported_added_voxels"],
            "false_GT_component_bridge_pairs":m.get("gt_inconsistent_new_component_pairs"),
            "gt_consistent_component_pairs":m.get("gt_consistent_new_component_pairs"),
            "gt_unknown_component_pairs":m.get("gt_unassigned_new_component_pairs"),
        }
    result={
        "status":"train_only_exploratory_single_patient_volume_comparison",
        "patient":patient,
        "pair_identity_source":pair_identity_source,
        "relation_train_protocol":expected_protocol,
        "source":"controlled damaged ImageCAS-X reference segmentation",
        "same_ct_mask_and_reference_for_all":True,
        "held_out_test_accessed":False,
        "models":records,
        "limitations":[
            (
                "ANZA/CNN relation HGB is clean-only and not matched to K10/15's 15 train perturbations"
                if relation_mode=="clean" else
                "ANZA/CNN relation HGB uses 15 patient-OOF train perturbations but neural candidate encoder was fitted on clean inputs"
            ),
            "Single training patient is not generalization evidence",
            "Real natural predicted mask errors remain untested",
            "ANZA advantage requires independent subjects and matched train augmentations",
        ],
    }
    report=root/f"train_{patient}_three_model_comparison.json"
    md=root/f"train_{patient}_three_model_comparison.md"
    if report.exists() or md.exists():
        raise FileExistsError("Study summary already exists")
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=[
        f"# CT28 CCTA 3D train-only controlled gap comparison — patient {patient}",
        "",
        "| Variant | Repairs | Added voxels | False added voxels | GT-inconsistent merge pairs | Dice before | Dice after | clDice before | clDice after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant,row in records.items():
        invalid_merges=(
            str(row["false_GT_component_bridge_pairs"])
            if row["false_GT_component_bridge_pairs"] is not None
            else "NOT MEASURED"
        )
        lines.append(f"| {variant} | {row['accepted_repairs']} | {row['added_voxels']} | "
                     f"{row['reference_unsupported_added_voxels']} | {invalid_merges} | "
                     f"{row['dice_before']:.5f} | {row['dice_after']:.5f} | "
                     f"{row['cldice_before']:.5f} | {row['cldice_after']:.5f} |")
    lines.extend(["","## Limitations",""]+[f"- {x}" for x in result["limitations"]])
    md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("patches","oof","relations","pilot","volume","all"),default="all")
    parser.add_argument("--root",type=Path,default=DEFAULT_STUDY)
    parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--frame-mode",choices=("canonical","fixed_lps"),default="canonical")
    parser.add_argument("--structure-models",action="store_true",help="Matched dense clDice+path CNN/ANZA variants")
    parser.add_argument("--separate-endpoints",action="store_true",help="Lossless individually encoded PAIR/JUNCTION tips; paired symmetric CNN/ANZA")
    parser.add_argument("--deepsets",action="store_true",help="Nonlinear symmetric per-tip moments; requires separate endpoints")
    parser.add_argument("--include-noaxial-control",action="store_true",help="Third matched ANZA without forced axial metric")
    parser.add_argument("--patient-id",type=int,default=953)
    parser.add_argument("--pair-identity-source",choices=("frozen_fusion","neural_pair","neural_pair_junction"),default="frozen_fusion")
    parser.add_argument("--relation-training",choices=("clean","augmented_15"),default="clean")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--ranking-weight",type=float,default=0.,help="Train-only same-scene candidate ranking ablation (0 preserves baseline)")
    args=parser.parse_args()
    if (args.frame_mode=="fixed_lps" or args.structure_models or args.separate_endpoints) and (args.cache_dir.resolve()==DEFAULT_CACHE.resolve() or args.root.resolve()==DEFAULT_STUDY.resolve()):
        raise ValueError("Fixed LPS ablation requires separate --cache-dir and --root; keep historical runs immutable")
    if args.patient_id not in TRAIN:
        raise ValueError("Held-out or validation patient forbidden in end-to-end study")
    if (args.separate_endpoints or args.include_noaxial_control) and not args.structure_models:
        raise ValueError("Separate tips and no-axial ablation require --structure-models")
    if args.deepsets and (not args.separate_endpoints or not args.structure_models):
        raise ValueError("Nonlinear DeepSets tip stem needs --structure-models --separate-endpoints")
    encoding="separate" if args.separate_endpoints else "summed"
    if args.structure_models:
        if args.deepsets:
            variants=("conv3d_deepsets_sep_structure","anza3d_deepsets_sep_structure")
            if args.include_noaxial_control:
                variants=(*variants,"anza3d_noaxial_deepsets_sep_structure")
        else:
            variants=(("conv3d_sep_structure","anza3d_sep_structure") if args.separate_endpoints
                      else ("conv3d_structure","anza3d_structure"))
            if args.include_noaxial_control:
                variants=(*variants,("anza3d_noaxial_sep_structure" if args.separate_endpoints
                                     else "anza3d_noaxial_structure"))
    else:
        variants=("conv3d","anza3d")
    if not math.isfinite(args.ranking_weight) or args.ranking_weight<0:
        raise ValueError("Ranking-weight must be a finite nonnegative number")
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage in ("patches","all"):
        ensure_patch_cache(args.cache_dir,frame_mode=args.frame_mode,with_vessel_targets=args.structure_models,endpoint_encoding=encoding)
    if args.stage in ("oof","all"):
        ensure_oof(args.cache_dir,args.root,epochs=args.epochs,device=args.device,
                   folds=args.folds,batch_size=args.batch_size,frame_mode=args.frame_mode,seed=args.seed,variants=variants,endpoint_encoding=encoding,ranking_weight=args.ranking_weight)
    if args.stage in ("relations","volume","all"):
        verify_upstream_ranking_protocol(args.root,variants,args.ranking_weight)
    if args.stage in ("relations","all"):
        ensure_relations(args.root,mode=args.relation_training,device=args.device,variants=variants)
    pilot=None
    if args.stage in ("pilot","volume","all"):
        pilot=ensure_pilot(args.root,patient=args.patient_id)
    if args.stage in ("volume","all"):
        ensure_neural_inference(args.root,pilot=pilot,patient=args.patient_id,device=args.device,
                                pair_identity_source=args.pair_identity_source,
                                relation_mode=args.relation_training,variants=variants)
        result=summarize(args.root,pilot=pilot,patient=args.patient_id,
                         pair_identity_source=args.pair_identity_source,
                         relation_mode=args.relation_training,variants=variants)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        print(json.dumps({"status":"stage_complete","stage":args.stage},indent=2))


if __name__=="__main__":
    main()
