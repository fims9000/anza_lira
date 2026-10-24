"""Independently replay saved S9 weights against synthetic held-out labels."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s7_original_retrain import (
    RunConfig,verify_sources,make_datasets,build_models,TrackDataset,split_fingerprints,
    predict_probabilities,metrics_counts,metric_from_counts,robust_validation_threshold)

def verify(seed:int,measurements:Path,results_root:Path):
    original=verify_sources(ROOT)
    root=results_root/f"s9_replication_seed{seed}"
    report=json.loads((root/"metrics.json").read_text())
    cfg=RunConfig(**{**report["config"],"widths":tuple(report["config"]["widths"])})
    if cfg.seed!=seed or cfg.source!="visualmetro":raise ValueError("Wrong split")
    data,_=make_datasets(cfg,measurements)
    fingerprints=split_fingerprints(data)
    if fingerprints["test"]["source_groups"]<20:raise ValueError("Non-independent test frames")
    if not {r[2] for r in data["train"]}.isdisjoint(r[2] for r in data["test"]):
        raise ValueError("Train/test frame leakage")
    out={}
    for name,model in build_models(cfg).items():
        ckpt=torch.load(root/name/"best_final.pt",map_location="cpu",weights_only=False)
        if ckpt["source_blobs"]!=original or ckpt["data_fingerprints"]!=fingerprints:
            raise ValueError("Source or dataset drift")
        model.load_state_dict(ckpt["state_dict"])
        v=TrackDataset(data["val"]);t=TrackDataset(data["test"])
        pv,yv=predict_probabilities(model,v,cfg)
        pvs,yvs=predict_probabilities(model,v,cfg,stress=True)
        if not np.array_equal(yv,yvs):raise ValueError("Validation truth changed")
        threshold=robust_validation_threshold(pv,pvs,yv)
        if threshold!=ckpt["validation_selected_threshold"]:raise ValueError("Threshold changed")
        for stress in (False,True):
            p,y=predict_probabilities(model,t,cfg,stress=stress)
            measures=metric_from_counts(metrics_counts(p>=threshold,y>=.5))
            expected=report["results"][name]["test_stress" if stress else "test_clean"]
            if any(measures[k]!=expected[k] for k in ("tp","fp","fn")):
                raise AssertionError("Held-out prediction mismatch")
            out[name+("_stress" if stress else "_clean")]=measures["dice"]
    return {"seed":seed,"test_frames":fingerprints["test"]["source_groups"],"dice":out,"pass":True}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--measurements",type=Path,required=True)
    p.add_argument("--results",type=Path,default=ROOT/"results")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    results=[verify(s,a.measurements,a.results) for s in (20261024,20261025,20261026)]
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))
