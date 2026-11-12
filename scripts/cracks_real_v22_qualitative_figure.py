#!/usr/bin/env python3
"""Make a publication-quality honest SUCCESS/FAILURE figure from real CRACKS V22.

Deterministically select first correct and first failure sorted by section,
pair, direction. Draw same raw seismic pixels with V21 source-visible expert
stroke, V22 true expert-annotated endpoint, synthetic reflected fake endpoint.
The expert-hidden source trace is NOT drawn. This is a visual counterfactual
illustration, not a natural geological failure or full ANZA/LIRA output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import COLORS, paths_from_mask, sections
from scripts.cracks_real_neighbor_path_v17 import gray, rgb
from scripts.cracks_real_matched_nolink_v21 import source_gap
from scripts.cracks_real_mirrored_v22 import (
    V21_SOURCE_SHA256, V19_SOURCE_SHA256, canonical_v21_digest,
    V21_SEMANTIC_SHA256,
)

def select_examples(rows):
    candidates=sorted(rows,key=lambda r:(
        int(r["section"]),int(r["expert_pair_a"]),
        int(r["expert_pair_b"]),int(r["direction"])))
    ok=next((r for r in candidates if int(r["central_correct"])==1),None)
    bad=next((r for r in candidates if int(r["central_correct"])==0),None)
    if ok is None or bad is None:
        raise RuntimeError("Need both successful and failed V22 real-image cases")
    return ok,bad

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--v19-cases",required=True)
    p.add_argument("--v21-cases",required=True)
    p.add_argument("--v22-cases",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--report",required=True)
    a=p.parse_args()

    def raw_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if raw_sha(a.v19_cases)!=V19_SOURCE_SHA256:
        raise RuntimeError("V19 case SHA source drift")
    if canonical_v21_digest(Path(a.v21_cases))!=V21_SEMANTIC_SHA256:
        raise RuntimeError("V21 labels or candidate evidence changed")
    with Path(a.v19_cases).open(newline="",encoding="utf8") as f:
        records={(int(r["section"]),int(r["path_id_a"]),int(r["path_id_b"])):r
                 for r in csv.DictReader(f)}
    with Path(a.v22_cases).open(newline="",encoding="utf8") as f:
        allrows=list(csv.DictReader(f))
    chosen=select_examples(allrows)
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.75),layout="constrained")
    details=[]
    with zipfile.ZipFile(a.images) as zi,zipfile.ZipFile(a.annotations) as za:
        ims=sections(zi)
        experts=sections(za,expert=True)
        for ax,row,tag in zip(axes,chosen,("а) Правильный выбор","б) Ошибка выбора")):
            sid=int(row["section"]);pair_a=int(row["expert_pair_a"])
            pair_b=int(row["expert_pair_b"])
            direction=int(row["direction"])
            r19=records[sid,pair_a,pair_b]
            labels=rgb(za,experts[sid])
            blue=np.all(labels==np.asarray(COLORS["blue"],np.uint8),axis=2)
            green=np.all(labels==np.asarray(COLORS["green"],np.uint8),axis=2)
            clean=blue&~ndi.binary_dilation(green,iterations=5)
            paths=sorted((pth for pth in paths_from_mask(clean) if len(pth)>=36),
                         key=lambda p:(-len(p),p[0][0],p[0][1]))
            source_ix=int(r19["interior_index_a" if direction==0 else "interior_index_b"])
            trace=paths[pair_a if direction==0 else pair_b]
            source=source_gap(trace,source_ix,int(row["gap"]),int(row["shift"]))
            if source is None:
                raise RuntimeError("Source anchor not found for published case")
            e0,tan,true,ttrue,span=source
            endpoints=[np.asarray(json.loads(row[f"candidate_{i}"]),float)
                       for i in (0,1)]
            correct=int(row["correct_candidate_index"])
            assert np.allclose(endpoints[correct],true), "V22 source path mismatch"
            fake=endpoints[1-correct]
            img=gray(zi,ims[sid])
            allpts=np.vstack([e0,true,fake])
            pad=50
            xlo=max(0,int(np.floor(allpts[:,1].min()))-pad)
            xhi=min(img.shape[1],int(np.ceil(allpts[:,1].max()))+pad+1)
            ylo=max(0,int(np.floor(allpts[:,0].min()))-pad)
            yhi=min(img.shape[0],int(np.ceil(allpts[:,0].max()))+pad+1)
            ax.imshow(img,cmap="gray",vmin=0,vmax=1,origin="upper")
            part=np.asarray(trace[max(0,span[0]-22):span[0]])
            ax.plot(part[:,1],part[:,0],color="#15dbff",lw=2.3,
                    label="Наблюдаемый фрагмент")
            ax.plot([e0[1],true[1]],[e0[0],true[0]],
                    color="#7aff6d",lw=2.3,label="Эталонная пара")
            ax.plot([e0[1],fake[1]],[e0[0],fake[0]],
                    color="#ff7d91",lw=2.1,ls="--",label="Зеркальная альтернатива")
            ax.scatter([e0[1]],[e0[0]],s=56,edgecolor="black",
                       facecolor="#15dbff",linewidths=.7,zorder=5)
            ax.scatter([true[1]],[true[0]],s=55,marker="o",facecolor="#7aff6d",
                       edgecolor="black",lw=.7,zorder=5)
            ax.scatter([fake[1]],[fake[0]],s=63,marker="x",c="#ff7d91",
                       linewidths=2.6,zorder=5)
            ax.set_xlim(xlo,xhi);ax.set_ylim(yhi,ylo)
            ax.set_aspect("equal",adjustable="box")
            ax.set_title(f"{tag} · срез {sid}",fontsize=12,weight="bold",pad=9)
            ax.set_xlabel("Пиксели по горизонтали",fontsize=9)
            ax.set_ylabel("Пиксели по вертикали",fontsize=9)
            ax.tick_params(labelsize=8)
            win=bool(int(row["central_correct"]))
            ax.text(.02,.02,"Центральный градиент: верно" if win else
                    "Центральный градиент: ошибка",
                    transform=ax.transAxes,fontsize=8.9,color="white",
                    bbox={"facecolor":"black","alpha":.80,"pad":5},
                    va="bottom",ha="left")
            details.append({"section":sid,"expert_pair":[pair_a,pair_b],
                "direction":direction,"case_gap_samples":int(row["gap"]),
                "central_prediction_correct":win,
                "candidate_endpoints_yx":[x.tolist() for x in endpoints],
                "positive_index":correct,"source_visible_endpoint_yx":e0.tolist()})
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,
               fontsize=9.5,frameon=False)
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=400,facecolor="white",format="jpeg",
                pil_kwargs={"quality":96,"subsampling":0,"dpi":(400,400)})
    plt.close(fig)
    report={"experiment":"V22 real-CRACKS original pixels qualitative example",
            "selection_rule":"first chronologically sorted correct and first wrong central-gradient outcomes",
            "not_end_to_end":True,
            "not_natural_geological_negatives":True,
            "cases":details,"source_v22_cases_sha256":raw_sha(a.v22_cases),
            "illustration_jpg_sha256":raw_sha(output)}
    Path(a.report).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                               encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
