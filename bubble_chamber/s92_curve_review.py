"""S92: private, reversible shape/ribbon and long-gap review on frozen S86 BMPs.

Analyzes S86 verified *observed* skeleton polylines. S87 joins exact pixel
adjacency across core boundaries; S92 distinguishes small strokes/hairpins from
extended curling lines and proposes curved (unobserved) gap continuations.
No change to source pixels, S86 masks, S72/S85 models, or physical identities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8, sha256_file
from .s87_seam_continuity import _read_source, assemble
from .s92_curve_geometry import (
    TraceReviewConfig,construct_chain_hypotheses,describe_trace,propose_gap_links,
)

SCHEMA="ANZA_LIRA_S92_REVERSIBLE_CURVE_AND_GAP_REVIEW_V1"


def _write_private(path:Path,data:bytes):
    with path.open("xb") as file:
        file.write(data)
    path.chmod(0o600)


def _json(data):
    return (json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf8")


def _ndjson(data):
    return ("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n"
                    for row in data)).encode("utf8")


def collapse_observed_components(records:list[dict])->tuple[list[dict],dict]:
    """Include every S86 fragment once; extend shapes through S87 seams."""
    stitched,decisions,stats=assemble(records)
    original={
        f["fragment_id"]:np.asarray(f["centerline_xy_source_px"],float)
        for rec in records for f in rec["fragments"]
    }
    stitched_members=set()
    output=[]
    for t in stitched:
        ids=t["source_fragment_ids_in_geometric_order"]
        if not ids or any(fid in stitched_members or fid not in original for fid in ids):
            raise ValueError("S87 reused a source fragment")
        stitched_members.update(ids)
        output.append({
            "component_id":t["track_hypothesis_id"],
            "source_fragment_ids":ids,
            "points_xy":np.asarray(t["points_xy_source_px"],float),
            "construction":"S87_OBSERVED_8NEIGHBOR_SEAM_ONLY",
        })
    for fid,p in sorted(original.items()):
        if fid not in stitched_members:
            output.append({
                "component_id":fid,
                "source_fragment_ids":[fid],
                "points_xy":p,
                "construction":"S86_ORIGINAL_OBSERVED_CENTERLINE_ONLY",
            })
    counts=Counter(fid for row in output for fid in row["source_fragment_ids"])
    if set(counts)!=set(original) or any(c!=1 for c in counts.values()):
        raise ValueError("S92 lost or duplicated S86 observed source fragments")
    return output,stats


def photo_gap_proxy(bmp:GrayBMP8,interpolated_xy)->dict:
    """Photometric *advisory* only. Missing pixels are not guaranteed to be dark.

    Compare center to transverse flanks at bridge interior positions using
    source grayscale. This is not likelihood, dE/dx, or a physics veto.
    """
    p=np.asarray(interpolated_xy,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<7 or not np.isfinite(p).all():
        raise ValueError("Expected finite subpixel bridge geometry")
    step=np.gradient(p,axis=0)
    length=np.linalg.norm(step,axis=1)
    if np.any(length<1e-8):
        return {"status":"UNRESOLVED_DEGENERATE_BRIDGE"}
    normal=np.stack([-step[:,1],step[:,0]],axis=1)/length[:,None]
    mid=p[2:-2]
    flanks=(mid+normal[2:-2]*3,mid-normal[2:-2]*3)
    all_points=np.concatenate([mid,*flanks],axis=0)
    x0=int(np.floor(all_points[:,0].min()))-3
    y0=int(np.floor(all_points[:,1].min()))-3
    x1=int(np.ceil(all_points[:,0].max()))+4
    y1=int(np.ceil(all_points[:,1].max()))+4
    if x0<0 or y0<0 or x1>bmp.width or y1>bmp.height:
        return {"status":"OUT_OF_BOUNDS_UNKNOWN"}
    view=np.ascontiguousarray(bmp._data[y0:y1,x0:x1],dtype=np.float32)
    def sample(points):
        xy=(points-np.array([x0,y0])).astype("float32")
        values=cv2.remap(view,xy[:,0].reshape(-1,1),
                         xy[:,1].reshape(-1,1),interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
        return values.reshape(-1)
    center=sample(mid)
    side=(sample(flanks[0])+sample(flanks[1]))/2.
    contrast=side-center
    return {"status":"IMAGE_PHOTOMETRY_ADVISORY",
            "median_transverse_darkness_gray_units":round(float(np.median(contrast)),4),
            "fraction_positive_darkness":round(float(np.mean(contrast>0)),4),
            "not_calibrated_probability":True}


def _preview(bmp:GrayBMP8,out:Path,cases:list[dict],limit:int)->list[dict]:
    previews=[]
    (out/"preview").mkdir(mode=0o700,exist_ok=True)
    for i,case in enumerate(cases[:limit]):
        p=np.asarray(case["points_xy"],float)
        if len(p)<4:continue
        center=np.median(p,axis=0)
        center=np.clip(center,[0,0],[bmp.width-1,bmp.height-1])
        size=min(384,bmp.width,bmp.height)
        size=int(size//16*16)
        if size<64:continue
        gray,location=bmp.crop(tuple(center.tolist()),size=size)
        x0,y0,_,_=location.original_xyxy
        overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        pixel=np.rint(p-np.array([x0,y0])).astype("int32")
        cv2.polylines(overlay,[pixel.reshape(-1,1,2)],False,
                      (42,220,40) if case["type"]=="COILING" else (255,150,40),
                      thickness=2,lineType=cv2.LINE_AA)
        if case.get("bridge") is not None:
            gap=np.asarray(case["bridge"],float)
            gapxy=np.rint(gap-np.array([x0,y0])).astype("int32")
            for k in range(0,len(gapxy)-2,2):
                cv2.line(overlay,tuple(gapxy[k]),tuple(gapxy[k+1]),
                         (180,70,230),thickness=2,lineType=cv2.LINE_AA)
        f=f"case_{i:03d}"
        for suffix,frame in (("source",gray),("overlay",overlay)):
            ok,png=cv2.imencode(".png",frame)
            if not ok:raise OSError("Failed to encode private S92 PNG")
            _write_private(out/"preview"/f"{f}_{suffix}.png",png.tobytes())
        previews.append({
            "case_id":f,
            "kind":case["type"],
            "source":f"preview/{f}_source.png",
            "overlay":f"preview/{f}_overlay.png",
            "label":case["label"],
            "image_bounds_xyxy":list(location.original_xyxy),
        })
    return previews


def _cases(components,classification,selected,limit):
    cases=[]
    order={
        "COILING_TRACE_REVIEW":0,
        "LOW_PRIORITY_POSSIBLE_ARTIFACT":1,
        "EXTENDED_IMAGE_TRACE_CANDIDATE":2,
    }
    for comp,info in zip(components,classification):
        cases.append({
            "type":("COILING" if info["shape_review_category"]=="COILING_TRACE_REVIEW"
                    else "SHORT_HAIR_OR_TICK" if info["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
                    else "EXTENDED"),
            "order":order[info["shape_review_category"]],
            "label":comp["component_id"],
            "points_xy":comp["points_xy"],
        })
    id_to_points={c["component_id"]:c["points_xy"] for c in components}
    for p in selected:
        cases.append({
            "type":"GAP_HYPOTHESIS",
            "order":-1,
            "label":p["proposal_id"],
            "points_xy":np.concatenate([
                id_to_points[p["component_a"]],id_to_points[p["component_b"]]
            ],axis=0),
            "bridge":p["inferred_bridge_points_xy"],
        })
    # Force category coverage before filling remaining preview budget.
    bykind={}
    for case in sorted(cases,key=lambda c:(c["order"],c["label"])):
        bykind.setdefault(case["type"],[]).append(case)
    chosen=[]
    for kind in ("COILING","SHORT_HAIR_OR_TICK","GAP_HYPOTHESIS","EXTENDED"):
        if bykind.get(kind):chosen.append(bykind[kind].pop(0))
    while len(chosen)<limit:
        remaining=[c for l in bykind.values() for c in l]
        if not remaining:break
        remaining.sort(key=lambda c:(c["order"],c["label"]))
        item=remaining[0];chosen.append(item);bykind[item["type"]].remove(item)
    return chosen[:limit]


def run_s92(s86_directory:Path,private_output:Path,*,
            bmp_path:Path|None=None,cfg:TraceReviewConfig=TraceReviewConfig(),
            preview_limit:int=20)->dict:
    cfg.validate()
    s86_directory=Path(s86_directory).resolve()
    private_output=Path(private_output).resolve()
    root=Path(__file__).resolve().parents[1]
    if (private_output==root or root in private_output.parents or
        private_output==s86_directory or
        s86_directory in private_output.parents or
        private_output in s86_directory.parents):
        raise ValueError("S92 PRIVATE output must be outside Git and frozen S86")
    if not 0<=preview_limit<=64:
        raise ValueError("Invalid S92 private preview limit")
    if private_output.exists():
        raise FileExistsError("S92 output is immutable; choose a new empty directory")
    manifest,records,source_manifest_sha=_read_source(s86_directory)
    bmp=None
    if bmp_path is not None:
        bmp_path=Path(bmp_path).resolve()
        if not bmp_path.is_file() or sha256_file(bmp_path)!=manifest["source_bmp_sha256"]:
            raise ValueError("Source BMP SHA does not match frozen S86 manifest")
        bmp=GrayBMP8(bmp_path)
        if [bmp.height,bmp.width]!=manifest["image_shape_yx"]:
            raise ValueError("Source BMP dimensions differ from S86")
    components,seam_stats=collapse_observed_components(records)
    annotated=[]
    for comp in components:
        info=describe_trace(comp["points_xy"],cfg)
        annotated.append({
            "component_id":comp["component_id"],
            "source_fragment_ids":comp["source_fragment_ids"],
            "construction":comp["construction"],
            **info,
            "machine_display_priority":(
                "REVIEW_FIRST_SPIRAL" if info["shape_review_category"]=="COILING_TRACE_REVIEW"
                else "LOW_PRIORITY_RETAINED" if info["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
                else "UNVERIFIED_TRACE_CANDIDATE"
            ),
            "source_fragment_remains_available":True,
        })
    proposals,selected,gap_stats=propose_gap_links(components,cfg)
    if bmp is not None:
        for p in proposals:
            p["source_image_gap_proxy"]=photo_gap_proxy(bmp,p["inferred_bridge_points_xy"])
    chains=construct_chain_hypotheses(components,selected,cfg)
    counts=Counter(item["shape_review_category"] for item in annotated)
    report={
        "schema":SCHEMA,"status":"COMPLETE_PRIVATE_REVIEW_ONLY",
        "source_s86_manifest_sha256":source_manifest_sha,
        "source_bmp_sha256":manifest["source_bmp_sha256"],
        "source_bmp_name":manifest["source_bmp_name"],
        "source_s86_method":manifest["config"]["method"],
        "source_fragments":sum(len(rec["fragments"]) for rec in records),
        "observed_components_after_S87":len(components),
        "source_S87_stitched_components":seam_stats["stitched_image_components"],
        "image_shape_yx":manifest["image_shape_yx"],
        "shape_review_categories":dict(counts),
        "gap":gap_stats,
        "preview_count":0,
        "image_photometry_available":bmp is not None,
        "source_line_pixels_mutated":False,
        "source_observed_centerlines_mutated":False,
        "suggested_gaps_are_observed_pixels":False,
        "physical_GT_count":0,
        "physical_auto_joins_authorized":False,
        "particle_identity_proven":False,
        "3D_reconstruction_proven":False,
        "config":asdict(cfg),
        "limitations":[
            "Short marks and tight hairpins may represent genuine small particle tracks; retain for review.",
            "Coiling signature is 2D curvature evidence, not a neutron interaction or a 3D helix classification.",
            "Gaps are image-only proposals; missing pixels are never reclassified as observed.",
            "A gap selected for review may still connect separate physical particles.",
            "Performance requires specialist-confirmed different-run ground truth.",
        ],
    }
    private_output.mkdir(mode=0o700,parents=True)
    private_output.chmod(0o700)
    for name,rows in (
        ("observed_shape_review.ndjson",annotated),
        ("gap_proposals_unknown.ndjson",proposals),
        ("stitched_chain_hypotheses_unknown.ndjson",chains),
    ):
        _write_private(private_output/name,_ndjson(rows))
    previews=_preview(bmp,private_output,
                      _cases(components,annotated,selected,preview_limit),
                      preview_limit) if bmp is not None and preview_limit else []
    report["preview_count"]=len(previews)
    _write_private(private_output/"preview_manifest.json",_json(previews))
    cards="".join(
        f"<section><h3>{html.escape(x['kind'])}: {html.escape(x['label'])}</h3>"
        f"<img src='{html.escape(x['source'])}' alt='original'><img src='{html.escape(x['overlay'])}' alt='machine geometry'>"
        f"</section>" for x in previews
    )
    body=(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>S92 image trace review — UNKNOWN</title>"
        "<style>body{font:16px sans-serif;margin:2em;background:#111;color:#eee}"
        "img{max-width:45%;margin:1%;image-rendering:auto}"
        "section{border-top:1px solid #666;padding:1em}</style>"
        "<h1>S92: гипотезы линий, разрывов и спиралей (НЕ физическая разметка)</h1>"
        "<p>Зелёный: наблюдаемая скрученная линия. Голубой: другие наблюдаемые линии. "
        "Пурпурный пунктир: ПРЕДПОЛОЖЕННЫЙ промежуток без наблюдавшихся пикселей.</p>"
        f"<p>Фрагменты: {report['source_fragments']}; после S87: {len(components)}; "
        f"неподтверждённых предложений соединения: {len(selected)}. "
        "Ни одно соединение не принято как физически достоверное.</p>"
        +cards+"</html>"
    )
    _write_private(private_output/"index.html",body.encode("utf8"))
    if bmp_path is not None and sha256_file(bmp_path)!=manifest["source_bmp_sha256"]:
        raise ValueError("Source BMP changed during S92 review")
    _write_private(private_output/"report.json",_json(report))
    return report
