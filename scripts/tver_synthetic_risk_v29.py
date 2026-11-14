#!/usr/bin/env python3
"""V29 actual V23/V25 selective-link policies on newly generated synthetic lines.

Scope: This is a controlled diagnostic of the repo's real link/NO_LINK code on
idealized probability maps, NOT training or inference by the original ANZA
network. Synthetic line instance IDs are accessible ONLY in evaluation.
An observationally identical positive/negative counterfactual is measured
separately from the identifiable distinct-negative risk-calibration stratum.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.cracks_real_auto_core_v23 import run_pipeline
from scripts.cracks_real_region_v25 import run_regions

SHAPE=(112,168)
THRESHOLD=.60
CAL_GROUPS=75
TEST_GROUPS=150
GROUP_BASE={"CAL":620000000,"TEST":720000000}
FAMILIES=("easy_positive","near_rival_positive",
          "indistinguishable_negative","distinct_negative")
RANKED=("V23_nearest","V23_geometry","V23_context","V25_region_hybrid")
OTHERS=("morph_close_radius4","NO_LINK")
METHODS=RANKED+OTHERS
QUANTILES=tuple(np.linspace(0,1,21))
MAX_CAL_FALSE_FRACTION=.02
NEIGHBOR=np.ones((3,3),dtype=np.uint8)


@dataclass(frozen=True)
class SyntheticScene:
    condition:str
    split:str
    group_id:int
    gap_band:str
    probability:np.ndarray
    left_anchor:tuple[int,int]
    right_anchor:tuple[int,int]
    left_port_truth:tuple[int,int]
    right_port_truth:tuple[int,int]
    distractor_anchor:tuple[int,int]|None
    physically_connected:bool
    def __post_init__(self):
        if self.condition not in FAMILIES:
            raise ValueError("Unknown frozen synthetic condition")
        if self.probability.shape!=SHAPE or not np.isfinite(self.probability).all():
            raise ValueError("Synthetic observation malformed")


def _curve_parameters(rng,group:int):
    center_y=int(rng.integers(42,69))
    slope=float(rng.uniform(-.12,.12))
    curve=float(rng.uniform(-.00085,.00085))
    gap=(12,22,32)[group%3]
    return center_y,slope,curve,gap


def _sample_curve(xs,base,slope,curve,offset=0,extra_slope=0):
    dx=np.asarray(xs,dtype=float)-83.
    return np.rint(base+offset+(slope+extra_slope)*dx+curve*dx*dx).astype(int)


def _paint_line(xs,ys):
    canvas=np.zeros(SHAPE,dtype=bool)
    xx=np.asarray(xs,dtype=int)
    yy=np.asarray(ys,dtype=int)
    valid=(yy>=4)&(yy<SHAPE[0]-4)&(xx>=4)&(xx<SHAPE[1]-4)
    canvas[yy[valid],xx[valid]]=True
    return ndi.binary_dilation(canvas,structure=NEIGHBOR,iterations=1)


def generate_group(split:str,group:int)->tuple[SyntheticScene,...]:
    if split not in GROUP_BASE or group<0:
        raise ValueError("Only frozen CAL and TEST synthetic streams supported")
    rng=np.random.default_rng(GROUP_BASE[split]+int(group)*7919)
    center,slope,curv,gap=_curve_parameters(rng,group)
    left_stop=83-gap//2
    right_start=83+(gap-gap//2)
    xleft=np.arange(17,left_stop+1)
    xright=np.arange(right_start,154)
    yleft=_sample_curve(xleft,center,slope,curv)
    yright=_sample_curve(xright,center,slope,curv)
    left=_paint_line(xleft,yleft)
    right=_paint_line(xright,yright)
    left_anchor=(int(_sample_curve([left_stop-6],center,slope,curv)[0]),left_stop-6)
    right_anchor=(int(_sample_curve([right_start+6],center,slope,curv)[0]),right_start+6)
    left_port=(int(yleft[-1]),int(xleft[-1]))
    right_port=(int(yright[0]),int(xright[0]))
    if np.any(left&right):
        raise AssertionError("Generator produced an already connected truth pair")
    # Correlated nuisance is sampled exactly once for the four related scenes.
    randomfield=ndi.gaussian_filter(rng.normal(size=SHAPE),sigma=1.4)
    yy,xx=np.indices(SHAPE)
    # Smooth stratiform analogue is deliberately not a faithful F3 simulator.
    background=np.clip(.045+.035*np.sin((yy+.13*xx)/5.3)
                       +.028*randomfield, .003,.16).astype(np.float32)
    ghost=np.zeros(SHAPE,dtype=bool)
    xgap=np.arange(left_stop+1,right_start)
    yg=_sample_curve(xgap,center,slope,curv)
    ghost=ndi.gaussian_filter(_paint_line(xgap,yg).astype(float),sigma=1.2)
    def probability(visible,soft_support=.11):
        p=np.clip(background+soft_support*ghost,0,.25).astype(np.float32)
        p[visible]=.93
        return p

    simple=probability(left|right)
    # Rival shares almost the same tangent but is a distinct physical line.
    offset=int(rng.integers(8,13))*int(rng.choice([-1,1]))
    xr=np.arange(right_start+1,155)
    yr=_sample_curve(xr,center,slope,curv,offset=offset,
                     extra_slope=float(rng.uniform(-.018,.018)))
    rival=_paint_line(xr,yr)
    if np.any(rival&(left|right)):
        raise AssertionError("Close-rival construction merged physical identities")
    rival_anchor=(int(yr[min(9,len(yr)-1)]),
                  int(xr[min(9,len(xr)-1)]))
    near=probability(left|right|rival)
    # Visible line orientation on right is deliberately incompatible here.
    extra=int(rng.choice([-1,1]))*int(rng.integers(17,25))
    rotated=_sample_curve(xright,center,slope,curv,offset=extra,
                          extra_slope=float(rng.uniform(.09,.16))*
                          int(np.sign(extra)))
    unrelated=_paint_line(xright,rotated)
    if np.any(unrelated&left):
        raise AssertionError("Distinct negative source mask unexpectedly touches source")
    neg=probability(left|unrelated,soft_support=.11)
    new_right=(int(rotated[min(6,len(rotated)-1)]),
               int(xright[min(6,len(xright)-1)]))
    common=dict(split=split,group_id=group,
                gap_band={12:"short_12px",22:"mid_22px",32:"long_32px"}[gap],
                left_anchor=left_anchor,left_port_truth=left_port)
    result=(
        SyntheticScene(condition="easy_positive",probability=simple,
            right_anchor=right_anchor,right_port_truth=right_port,
            distractor_anchor=None,physically_connected=True,**common),
        SyntheticScene(condition="near_rival_positive",probability=near,
            right_anchor=right_anchor,right_port_truth=right_port,
            distractor_anchor=rival_anchor,physically_connected=True,**common),
        # Exactly same bytes as easy positive; its latent physical origin
        # has changed but NOTHING in the observation changed.
        SyntheticScene(condition="indistinguishable_negative",
            probability=simple.copy(),right_anchor=right_anchor,
            right_port_truth=right_port,distractor_anchor=None,
            physically_connected=False,**common),
        SyntheticScene(condition="distinct_negative",probability=neg,
            right_anchor=new_right,right_port_truth=(
                int(rotated[0]),int(xright[0])),
            distractor_anchor=None,physically_connected=False,**common),
    )
    if not np.array_equal(result[0].probability,result[2].probability):
        raise AssertionError("Identical-observation impossibility test was broken")
    if len(result)!=4 or any(s.probability.max()<.8 for s in result):
        raise AssertionError("Invalid synthetic scene bank")
    return result


def _label_at(labels,point):
    y,x=point
    if not (0<=y<labels.shape[0] and 0<=x<labels.shape[1]):
        return 0
    return int(labels[y,x])


def _mask_connected(mask,a,b):
    labels,_=ndi.label(mask,structure=NEIGHBOR)
    first=_label_at(labels,a)
    second=_label_at(labels,b)
    return first>0 and first==second


def _candidate_contains_truth(pred,scene,radius=5.5):
    if not scene.physically_connected:
        return False
    ports=pred.get("ports",())
    candidates=pred.get("candidates",())
    if not len(ports) or not len(candidates):
        return False
    for i,j,*_ in candidates:
        a=np.asarray(ports[i].point,dtype=float)
        b=np.asarray(ports[j].point,dtype=float)
        x=np.asarray(scene.left_port_truth,float)
        y=np.asarray(scene.right_port_truth,float)
        if (np.linalg.norm(a-x)<=radius and np.linalg.norm(b-y)<=radius or
            np.linalg.norm(a-y)<=radius and np.linalg.norm(b-x)<=radius):
            return True
    return False


def raw_decision(scene,method):
    p=scene.probability
    if method=="V23_nearest":
        return run_pipeline(p,threshold=THRESHOLD,strategy="nearest")
    if method=="V23_geometry":
        return run_pipeline(p,threshold=THRESHOLD,strategy="geometry")
    if method=="V23_context":
        return run_pipeline(p,threshold=THRESHOLD,strategy="context")
    if method=="V25_region_hybrid":
        return run_regions(p,THRESHOLD,mode="region_hybrid")
    mask=p>=THRESHOLD
    if method=="NO_LINK":
        return {"raw_mask":mask,"completed_mask":mask.copy(),
                "selected":(),"ports":(),"candidates":()}
    if method=="morph_close_radius4":
        completed=ndi.binary_closing(mask,structure=NEIGHBOR,iterations=4)
        return {"raw_mask":mask,"completed_mask":completed,
                "selected":(),"ports":(),"candidates":()}
    raise ValueError(method)


def _filtered_mask(result,threshold):
    if threshold is None:
        return np.asarray(result["completed_mask"],dtype=bool)
    mask=np.asarray(result["raw_mask"],dtype=bool).copy()
    for edge in result["selected"]:
        if float(edge["score"])>=threshold:
            pixels=np.asarray(edge["path"],int)
            if len(pixels):
                mask[pixels[:,0],pixels[:,1]]=True
    return mask


def _passes_selected_count(result,threshold):
    if threshold is None:
        return len(result["selected"])
    return sum(float(e["score"])>=threshold for e in result["selected"])


def evaluate(scene,method,result,gate=None):
    mask=_filtered_mask(result,gate)
    connected=_mask_connected(mask,scene.left_anchor,scene.right_anchor)
    wrong=bool(scene.distractor_anchor is not None and
                (_mask_connected(mask,scene.left_anchor,scene.distractor_anchor)
                 or _mask_connected(mask,scene.right_anchor,scene.distractor_anchor)))
    inserted=_passes_selected_count(result,gate)
    # Negative contains only physically unrelated fragments, hence every
    # predicted link between their connected components is false.
    false=bool((not scene.physically_connected) and
               (connected or inserted>0))
    correct=bool(scene.physically_connected and connected and not wrong)
    return {
        "correct_connected":int(correct),
        "wrong_competitor":int(wrong),
        "false_bridge":int(false),
        "two_primary_anchors_connected":int(connected),
        "abstained":int(inserted==0 and not connected),
        "newly_selected_links":int(inserted),
        "truth_pair_in_auto_candidate_set":int(
            _candidate_contains_truth(result,scene))
            if scene.physically_connected else None,
    }


def _run_stream(split,n_groups):
    rows=[]
    for group in range(n_groups):
        scenes=generate_group(split,group)
        predictions={}
        for scene in scenes:
            for method in METHODS:
                prediction=raw_decision(scene,method)
                predictions[scene.condition,method]=prediction
                measured=evaluate(scene,method,prediction)
                rows.append({"split":split,"group_id":group,
                             "condition":scene.condition,
                             "gap_band":scene.gap_band,
                             "method":method,
                             "policy":"frozen_original",
                             **measured})
        # The observable pair is bitwise identical and all methods must
        # act identically, regardless of the physical-truth counterfactual.
        for method in METHODS:
            a=predictions["easy_positive",method]
            b=predictions["indistinguishable_negative",method]
            if (not np.array_equal(a["completed_mask"],b["completed_mask"])
                    or len(a["selected"])!=len(b["selected"])):
                raise AssertionError("Latent truth leaked into link/NO_LINK prediction")
        yield rows, scenes, predictions
        rows=[]


def _case_key(scene):
    return (scene.split,scene.group_id,scene.condition)


def _precompute(split,n_groups):
    examples={}
    results={}
    original=[]
    for group_rows,scenes,preds in _run_stream(split,n_groups):
        original.extend(group_rows)
        for scene in scenes:
            examples[_case_key(scene)]=scene
            for method in RANKED:
                results[(*_case_key(scene),method)]=preds[scene.condition,method]
    return examples,results,original


def choose_cal_reject_gates(examples,predictions):
    """CAL only: maximize successful positive links under <=2% distinct-negative FPs."""
    thresholds={}
    receipts={}
    cal_cases=list(examples.values())
    neg=[s for s in cal_cases if s.condition=="distinct_negative"]
    positive=[s for s in cal_cases if s.physically_connected]
    assert len(neg)==CAL_GROUPS and len(positive)==2*CAL_GROUPS
    budget=int(np.floor(MAX_CAL_FALSE_FRACTION*len(neg)))
    for method in RANKED:
        scores=[float(edge["score"]) for s in cal_cases
                for edge in predictions[(*_case_key(s),method)]["selected"]]
        # Frozen grid is quantiles of CAL method scores plus "accept no links".
        grid=sorted(set([-1.e6,1.e6] +
                        ([float(z) for z in np.quantile(
                            scores,QUANTILES)] if scores else [])))
        choices=[]
        for gate in grid:
            false_count=sum(evaluate(s,method,
                predictions[(*_case_key(s),method)],gate)["false_bridge"]
                for s in neg)
            positives=sum(evaluate(s,method,
                predictions[(*_case_key(s),method)],gate)["correct_connected"]
                for s in positive)
            wrong=sum(evaluate(s,method,
                predictions[(*_case_key(s),method)],gate)["wrong_competitor"]
                for s in positive)
            if false_count<=budget:
                choices.append((positives,-wrong,-false_count,gate))
        if not choices:
            raise AssertionError("Reject-all policy should satisfy CAL risk")
        chosen=max(choices)
        gate=float(chosen[-1])
        thresholds[method]=gate
        receipts[method]={
            "CAL_positive_correct":int(chosen[0]),
            "CAL_positive_wrong_competitor":int(-chosen[1]),
            "CAL_distinct_negative_false":int(-chosen[2]),
            "CAL_distinct_negative_n":len(neg),
            "CAL_empirical_false_budget":budget,
            "CAL_accepted_unique_gates_evaluated":len(grid),
            "selected_additional_reject_gate":gate,
            "only_removes_existing_historical_edges":True,
            "TEST_labels_used_for_threshold_choice":False,
        }
    return thresholds,receipts


def _postprocess(examples,precomputed,original,gates,split):
    out=list(original)
    for scene in examples.values():
        for method in RANKED:
            pred=precomputed[(*_case_key(scene),method)]
            row=evaluate(scene,method,pred,gates[method])
            out.append({"split":split,"group_id":scene.group_id,
                        "condition":scene.condition,"gap_band":scene.gap_band,
                        "method":method,"policy":"CAL_reject_only",
                        **row})
    return out


def _summarize(rows,split):
    out=[]
    methods=list(METHODS)
    for method in methods:
        for policy in (("frozen_original","CAL_reject_only") if
                       method in RANKED else ("frozen_original",)):
            filtered=[r for r in rows if r["split"]==split and
                      r["method"]==method and r["policy"]==policy]
            if len(filtered)!=4*(CAL_GROUPS if split=="CAL" else TEST_GROUPS):
                raise AssertionError("Incomplete method/split benchmark rows")
            for condition in FAMILIES:
                subset=[r for r in filtered if r["condition"]==condition]
                out.append({
                    "method":method,"policy":policy,"split":split,
                    "condition":condition,
                    "n_original_independent_nuisance_groups":len(subset),
                    "positive_correct":sum(r["correct_connected"] for r in subset),
                    "negative_false_bridge":sum(r["false_bridge"] for r in subset),
                    "wrong_competitor":sum(r["wrong_competitor"] for r in subset),
                    "abstentions":sum(r["abstained"] for r in subset),
                    "newly_selected_links":sum(r["newly_selected_links"] for r in subset),
                    "truth_in_auto_candidate_set":sum(
                        int(r["truth_pair_in_auto_candidate_set"])
                        for r in subset if r["truth_pair_in_auto_candidate_set"] is not None),
                    "N":len(subset),
                })
    return out


def _paired_group_bootstrap(rows,*,policy="CAL_reject_only",repeats=4000):
    def groups(method,condition,metric):
        d={(r["group_id"],r["condition"]):r for r in rows
           if r["split"]=="TEST" and r["method"]==method and r["policy"]==policy}
        return np.asarray([d[(g,condition)][metric] for g in range(TEST_GROUPS)],
                          dtype=float)
    if not any(r["policy"]==policy for r in rows):
        return {"status":"NO_CAL_REJECT_RESULTS"}
    comparisons={}
    rng=np.random.default_rng(2026112914)
    ix=rng.integers(0,TEST_GROUPS,size=(repeats,TEST_GROUPS))
    for condition in ("easy_positive","near_rival_positive",
                      "distinct_negative","indistinguishable_negative"):
        metric="false_bridge" if "negative" in condition else "correct_connected"
        ours=groups("V25_region_hybrid",condition,metric)
        analog=groups("V23_geometry",condition,metric)
        delta=ours-analog
        bootstrap=delta[ix].mean(axis=1)
        comparisons[condition]={
            "metric":metric,
            "V25_minus_V23_geometry":float(delta.mean()),
            "paired_group_bootstrap95_descriptive":
                np.quantile(bootstrap,[.025,.975]).tolist(),
            "TEST_independent_nuisance_groups":TEST_GROUPS,
            "not_geological_external_or_prospective":True,
        }
    return comparisons


def _save_csv(out,rows,name):
    with (out/name).open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(out):
    out=Path(out)
    if out.exists():
        raise FileExistsError("Never overwrite V29 locked TEST evidence")
    cal_scenes,cal_preds,cal_original=_precompute("CAL",CAL_GROUPS)
    gates,cal_receipts=choose_cal_reject_gates(cal_scenes,cal_preds)
    cal_rows=_postprocess(cal_scenes,cal_preds,cal_original,gates,"CAL")
    # This block is the *first* opening of the new V29 heldout test stream.
    test_scenes,test_preds,test_original=_precompute("TEST",TEST_GROUPS)
    test_rows=_postprocess(test_scenes,test_preds,test_original,gates,"TEST")
    rows=cal_rows+test_rows
    summary=_summarize(rows,"CAL")+_summarize(rows,"TEST")
    paired=_paired_group_bootstrap(test_rows)
    status={
        "status":"V29_NEW_SYNTHETIC_PROBABILITY_POLICY_STRESS_TEST_EXECUTED",
        "protocol":"docs/research/tver2026/V29_FROZEN_SYNTHETIC_NEAR_RIVAL_SELECTIVE_LINKS_20261114.md",
        "not_original_pretrained_ANZA_or_original_trained_LIRA":True,
        "real_repository_V23_V25_selective_rules_executed":True,
        "source_is_PROCEDURAL_PROBABILITY_not_learned_detection":True,
        "new_independent_synthetic_generator_does_not_simulate_F3_physics":True,
        "test_opened_only_after_CAL_thresholds_frozen":True,
        "pixel_or_endpoint_or_latent_true_instance_id_supplied_to_inference":False,
        "fixed_segmentation_probability_threshold":THRESHOLD,
        "CAL_group_count":CAL_GROUPS,
        "TEST_group_count":TEST_GROUPS,
        "n_observationally_identical_counterfactual_TEST_pairs":TEST_GROUPS,
        "identical_observation_positive_negative_2D_identifiability_impossible":True,
        "CAL_risk_gate_negative_type":"distinct_negative ONLY",
        "CAL_empirical_max_false_bridge_fraction":MAX_CAL_FALSE_FRACTION,
        "CAL_additional_reject_only_gates":gates,
        "CAL_reject_gate_selection_receipts":cal_receipts,
        "all_synthetic_metrics":summary,
        "paired_section_family_bootstrap":paired,
        "no_genuine_geological_fault_instance_truth_or_external_survey":True,
        "all_methods_same_synthetic_probability_inputs":True,
        "risk_control_guaranteed_outside_CAL":False,
        "historical_main_Tver_V28_results_unchanged":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    _save_csv(out,rows,"PER_SCENE.csv")
    _save_csv(out,summary,"METRICS.csv")
    (out/"STATUS.json").write_text(json.dumps(
        status,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
        encoding="utf8")
    print(json.dumps({
        "event":"V29_NEW_SYNTHETIC_SELECTIVE_LINK_EVALUATED",
        "n_cal_scenes":4*CAL_GROUPS,
        "n_test_scenes":4*TEST_GROUPS,
        "CAL_gates":gates,
        "test":[r for r in summary if r["split"]=="TEST"],
        "paired":paired,
    },ensure_ascii=False,allow_nan=False),flush=True)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir)


if __name__=="__main__":
    main()
