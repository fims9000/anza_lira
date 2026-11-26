#!/usr/bin/env python3
"""Posthoc controlled-gap path oracle and optional HGB/consistency diagnostics.

Reads five frozen development patients only. Reference truth selects the
diagnostic oracle endpoints, never a learned prediction or a production repair.
The oracle runs each gap independently against the unchanged broken mask.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

import run_ct28_graph_lira as graph
from ccta_3d_endpoints import extract_endpoints, build_candidate_scenes
from ccta_3d_runtime import frozen_path_protocol, validate_ct_inputs, _paths_for_decision
from ccta_3d_an_runtime import NeuralCCTA3dInference
from ccta_3d_k10 import EVAL_SEED, EVAL_PERTURBATIONS, EVAL_CONSISTENCY
from ccta_3d_train_gap_simulator import VAL_CT28_IDS
from ccta_3d_train_pilot import REFERENCE_DIR, SELECTED_PACK, digest
from ccta_3d_val_an_compare import check_val_pilot, pair_experiment_paths
from ccta_3d_val_freeze import verify_validation_freeze
from ccta_3d_val_pilot import DEFAULT_OUTPUT
from ccta_3d_cli import DEFAULT_PATH
from ccta_3d_shift import training_count_support
from ct28_mask_reconstruction import apply_approved_paths
from ct28_perturbations import perturb_scene_endpoints
from ct28_safety_policy import should_accept
from ct28_runtime_gap_supervision import gap_pair_labels, window_labels

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_gap(gap, scenes, endpoints, ct, affine, broken, reference, path, slack, settings, radius):
    if gap["status"] != "unique":
        return dict(gap, restored=False, outcome="oracle_"+gap["status"])
    pair = tuple(gap["candidate_pairs"][0])
    matches = [(s,c) for s in scenes for c in s.candidates
               if c.kind=="PAIR" and tuple(sorted(c.endpoint_ids))==pair]
    if not matches:
        raise ValueError("Oracle candidate missing from runtime windows")
    scene, candidate = sorted(matches,key=lambda x:x[0].scene_id)[0]
    proposal = SimpleNamespace(pair_candidate_id=candidate.candidate_id,
                                junction_candidate_id=None, selected_relation="PAIR")
    result = dict(gap, selected_pair=list(pair), scene_id=scene.scene_id,
                  restored=False, outcome="not_run")
    try:
        paths, _ = _paths_for_decision(proposal,scene,endpoints,ct,affine,path,slack,None)
        reconstructed = apply_approved_paths(broken,affine,paths,
             radius_mm=settings["repair_radius_mm"],
             max_new_voxels=settings["max_new_voxels_per_decision"])
        if not reconstructed.accepted:
            return dict(result,outcome="mask_gate:"+reconstructed.reason)
        added = reconstructed.mask & ~broken
        coords = np.argwhere(added)
        center = np.asarray(gap["center_voxel"])
        within = np.linalg.norm((coords-center)@affine[:3,:3].T,axis=1)<=radius+1e-7
        overlap = int(np.count_nonzero(within & reference[tuple(coords.T)]))
        return dict(result,outcome="ok" if overlap else "path_misses_removed_region",
                    restored=bool(overlap), added_voxels=int(added.sum()),
                    unsupported_added_voxels=int(np.count_nonzero(added & ~reference)),
                    gap_overlap_voxels=overlap,
                    gap_lumen_recall=overlap/gap["removed_voxels"])
    except (ValueError, RuntimeError, KeyError, IndexError) as exc:
        return dict(result,outcome="path_error",detail=str(exc))


def audit_windows(engine, scenes, endpoints, ct, affine, labels, gaps, support):
    """Re-score full-mask windows with original weights, no fit/threshold changes."""
    inverse=np.linalg.inv(affine)
    true_pairs={tuple(g["candidate_pairs"][0]) for g in gaps if g["status"]=="unique"}
    rows=[]
    for scene in scenes:
        _, frame, candidates=engine._frames(scene,endpoints)
        row={"scene_id":scene.scene_id,"oracle_relation":labels.get(scene.scene_id,"AMBIGUOUS")}
        counts=tuple(sum(c.kind==k for c in scene.candidates) for k in ("PAIR","JUNCTION"))
        row.update(pair_candidates=counts[0],junction_candidates=counts[1],
                   train_count_support=int(support.get(counts,0)))
        try:
            scores, decisions=engine._score(scene,frame,candidates,ct,inverse)
            base=decisions["ct_conditioned"]
            row.update(base_relation=base["raw_relation"],threshold_relation=base["relation"],
                       confidence=base["confidence"],relation_probabilities=base["probabilities"],
                       base_pair=base["pair_candidate_id"],base_junction=base["junction_candidate_id"],
                       endpoint_patch_fov_abstain=bool(base.get("endpoint_patch_fov_abstain",False)))
            rank=sorted((c for c in scene.candidates if c.kind=="PAIR"),
                        key=lambda c:-float(scores.loc[scores.candidate_id.eq(c.candidate_id),"ct_score"].iloc[0]))
            row["true_pair_ranks"]=[i for i,c in enumerate(rank,1) if tuple(sorted(c.endpoint_ids)) in true_pairs]
            row["pair_scores"]=[{"candidate_id":c.candidate_id,
                                "score":float(scores.loc[scores.candidate_id.eq(c.candidate_id),"ct_score"].iloc[0])}
                                for c in rank]
            conditions={}
            for name,pos,tan in (("none",0.,0.),("tangent",0.,20.),
                                  ("position",.8,0.),("both",.8,20.)):
                keys=[]; tau_fail=0; fov_fail=0; rank_changes=0
                for repetition in range(EVAL_PERTURBATIONS):
                    if name=="none":
                        decision=base
                    else:
                        perturbed=perturb_scene_endpoints(frame,seed=EVAL_SEED,
                           repetition=repetition,position_noise_mm=pos,tangent_noise_deg=tan)
                        _, d=engine._score(scene,perturbed,candidates,ct,inverse)
                        decision=d["ct_conditioned"]
                    keys.append(decision["relation_key"])
                    tau_fail+=int(decision["confidence"]<.85)
                    fov_fail+=int(bool(decision.get("endpoint_patch_fov_abstain",False)))
                    rank_changes+=int(decision["relation_key"]!=base["relation_key"])
                consistency=graph.consistency_value(base["relation_key"],keys)
                accepted=should_accept(base["relation"],base["selected_relation"],
                          bool(base["compatibility_satisfied"]),consistency,EVAL_CONSISTENCY)
                conditions[name]={"consistency":consistency,"accepted":accepted,
                   "below_tau_repeats":tau_fail,"fov_abstain_repeats":fov_fail,
                   "decision_key_changes":rank_changes,"keys":keys}
            row["perturbation_conditions"]=conditions
        except (ValueError,RuntimeError,KeyError,IndexError) as exc:
            row["error"]=str(exc)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",type=Path,required=True)
    p.add_argument("--pilot-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--study-root",type=Path,help="Optional existing matched model root for window diagnostics")
    p.add_argument("--deepsets",action="store_true")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Immutable diagnostic output")
    frozen=verify_validation_freeze(a.freeze)
    settings=frozen["settings"]
    path,slack=frozen_path_protocol(DEFAULT_PATH)
    engines={}
    if a.study_root is not None:
        variants=("conv3d_deepsets_sep_structure","anza3d_deepsets_sep_structure") if a.deepsets else ("conv3d_sep_structure","anza3d_sep_structure")
        for variant in variants:
            relation,pair,junction=pair_experiment_paths(a.study_root,variant)
            engines[variant]=NeuralCCTA3dInference.from_experiment(
                k10_pack=SELECTED_PACK,relation_dir=relation,pair_oof_dir=pair,
                junction_oof_dir=junction,variant=variant,device=a.device,
                pair_identity_source="neural_pair_junction")
    support=training_count_support()
    report=[]
    for patient in sorted(VAL_CT28_IDS):
        pilot=a.pilot_root/f"patient_{patient}"
        check_val_pilot(pilot_dir=pilot,patient=patient,freeze_path=a.freeze,frozen=frozen)
        ref_path=REFERENCE_DIR/f"{patient}.coronary.nii.gz"
        if digest(ref_path)!=read(pilot/"posthoc_metrics.json")["reference_sha256"]:
            raise ValueError("Reference mask changed after frozen evaluation")
        images=[nib.load(str(x)) for x in (pilot/"ct.nii.gz",pilot/"broken_mask.nii.gz",ref_path)]
        if any(img.shape!=images[0].shape or not np.allclose(img.affine,images[0].affine,atol=1e-6,rtol=0) for img in images[1:]):
            raise ValueError("Unaligned diagnostic volumes")
        ct=np.asarray(images[0].dataobj,dtype=np.float32)
        broken=np.asarray(images[1].dataobj)>.5
        reference=np.asarray(images[2].dataobj)>.5
        affine=images[0].affine
        validate_ct_inputs(ct,broken,affine)
        endpoints,_=extract_endpoints(broken,affine,min_component_voxels=settings["min_component_voxels"],
                  trace_mm=settings["trace_mm"],max_endpoints=settings["max_endpoints"])
        scenes=build_candidate_scenes(endpoints,patient_label=str(patient),max_scenes=settings["max_scenes"])
        gap_meta=read(pilot/"gaps.json")
        gaps=gap_pair_labels(reference,broken,affine,gap_meta,endpoints,scenes)
        labels,_=window_labels(scenes,gaps)
        oracles=[oracle_gap(g,scenes,endpoints,ct,affine,broken,reference,path,slack,settings,
                          float(gap_meta["gap_radius_mm"])) for g in gaps]
        row={"patient_id":patient,"window_count":len(scenes),"oracle_gaps":oracles,
             "label_counts":dict(Counter(labels.values())),
             "window_diagnostics":{name:audit_windows(engine,scenes,endpoints,ct,affine,labels,gaps,support)
                                   for name,engine in engines.items()}}
        report.append(row)
        print(f"Oracle {patient}: {sum(g['restored'] for g in oracles)}/{len(oracles)}",flush=True)
    all_gaps=[g for r in report for g in r["oracle_gaps"]]
    result={"protocol":"posthoc_oracle_independent_gaps_v1",
            "held_out_test_accessed":False,"oracle_uses_ground_truth":True,
            "not_model_performance":True,
            "total_gaps":len(all_gaps),"restored_with_oracle":sum(g["restored"] for g in all_gaps),
            "outcomes":dict(Counter(g["outcome"] for g in all_gaps)),"patients":report}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k!="patients"},indent=2))

if __name__=="__main__":
    main()
