"""V50 small deterministic synthetic sanity: learning curved path vs ribbon alone.

This is a single hand-written renderer family, NOT held-out geological proof.
It explicitly uses predicted endpoints (no boundary-anchor refinement) for
both models so it measures learned heatmap versus geometry-only ribbon.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

from scripts.tver_v50_conditional_path import (
    fit_path_head,path_heatmap,ribbon_dp_route)


def scene(seed):
    rng=np.random.default_rng(seed)
    amplitude=rng.uniform(-5,5)
    slope=rng.uniform(-.16,.16)
    shift=rng.uniform(-2,2)
    def line_x(y):
        return 48+shift+slope*(y-48)+amplitude*np.sin((y-40)*np.pi/18)
    ys=np.arange(96)
    xx=np.arange(96)
    center=np.asarray([line_x(y) for y in ys])
    ridge=np.exp(-.5*((xx[None,:]-center[:,None])/1.25)**2)
    gray=np.clip(.18+.7*ridge+rng.normal(0,.05,(96,96)),0,1).astype(np.float32)
    p=np.clip(.02+.91*ridge+rng.normal(0,.03,(96,96)),0,1).astype(np.float32)
    p[39:57,37:60]=0
    a=(32.,line_x(32)+rng.normal(0,3))
    b=(63.,line_x(63)+rng.normal(0,3))
    good={"source":a,"target":b,"geometry":-.4,"y":1}
    bad={"source":(32.,line_x(32)+8),"target":(63.,line_x(63)-8),
         "geometry":-.8,"y":0}
    return {"kind":"LINK","section":seed,"id":str(seed),
            "proposals":[good,bad],"gray":gray,"visible_prob":p,
            "rectangle":(39,57,37,60),"origin":(0,0),
            "hidden_centerline":[(int(y),int(round(line_x(y)))) for y in range(41,55)]}


def hidden_success(s,path):
    y0,y1,x0,x1=s["rectangle"]
    pred=np.zeros((96,96),bool)
    for y,x in path:
        if y0<=y<y1 and x0<=x<x1:pred[y,x]=True
    truth=np.zeros_like(pred)
    for y,x in s["hidden_centerline"]:
        if y0<=y<y1 and x0<=x<x1:truth[y,x]=True
    if not pred.any():return False
    prec=float(ndi.binary_dilation(truth,iterations=2)[pred].mean())
    rec=float(ndi.binary_dilation(pred,iterations=2)[truth].mean())
    return prec>=.8 and rec>=.8


def run(out):
    train=[scene(i) for i in range(100)]
    test=[scene(i) for i in range(200,240)]
    model,passport=fit_path_head(train,Path(out)/"train_receipt")
    result={"learned_ribbon":0,"untrained_ribbon":0}
    for s in test:
        good=s["proposals"][0]
        a,b=np.asarray(good["source"]),np.asarray(good["target"])
        learned=path_heatmap(model,s,good)
        yy,xx=np.mgrid[:96,:96]
        pts=np.stack([yy,xx],axis=-1)
        d=b-a
        t=np.clip(np.sum((pts-a)*d,axis=-1)/max(float(d@d),1.),0,1)
        foot=a+t[...,None]*d
        untrained=np.exp(-np.sum((pts-foot)**2,axis=-1)/(2*3**2))
        for mode,h in (("learned_ribbon",learned),("untrained_ribbon",untrained)):
            path=ribbon_dp_route(h,a,b)
            result[mode]+=int(hidden_success(s,path))
    report={
        "status":"V50_ONE_RENDERER_FAMILY_SYNTHETIC_SANITY_NOT_GEOLOGICAL",
        "train_independent_synthetic_seeds":"0..99",
        "test_independent_synthetic_seeds":"200..239",
        "n_test":len(test),"success_counts":result,
        "same_generator_family":True,
        "oracle_generated_controlled_endpoints":True,
        "not_real_ANZA_SBPP_or_geology":True,
        "boundary_anchor_refinement_disabled_for_both_methods":True,
        "path_model_TRAIN_receipt":passport,
    }
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    (p/"V50_SYNTHETIC_SANITY.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="path_model_TRAIN_receipt"},indent=2))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    run(args.out_dir)
