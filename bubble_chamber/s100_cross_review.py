"""S100 private whole-frame geometric-X REVIEW over verifiable H50Z S86–S96.

Reopen and validate exact source BMP pixels, all S86 masks and complete
S86→S92→S95→S96 lineage BEFORE computing anything. Source end-arm
coordinates are independently tied to frozen S86/S87 centerlines. Optional
S96 mask-branch arms are tied to exact S86 binary tile pixels AND SHA.
Machine-selected examples are NOT blind original-image sample nor GT.

Outputs new immutable PRIVATE geometry candidates, a small actual original
photo-vs-observed-source-overlay gallery, and a compact ZIP. No original
200MB BMP, weights, or full mask collection is copied into the ZIP.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import html
import json
import math

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .real_bmp_run_receipts import verify_stage_receipts
from .s87_seam_continuity import _read_source
from .s92_curve_review import collapse_observed_components,_write_private,_json,_ndjson
from .s100_cross_geometry import (
    SCHEMA,SOURCE_ORIGINAL,SOURCE_MASK,ACCEPT,XShapeConfig,
    build_x_shape_candidate_inventory
)


def _read_ndjson(path:Path,*,limit:int):
    if not path.is_file():
        raise FileNotFoundError(f"S100 required source report missing: {path.name}")
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            if len(rows)>=limit:
                raise ValueError("S100 source junction/arm inventory budget exceeded")
            item=json.loads(line)
            if not isinstance(item,dict):
                raise ValueError("S100 malformed source NDJSON record")
            rows.append(item)
    return rows


def _verify_junction_ports(junctions:list[dict],sources:dict[str,dict],
                           *,source_repr:str,mask_tiles:dict|None=None):
    """Require actual source path endpoint/tangent data for all S96 ports."""
    for j in junctions:
        if source_repr==SOURCE_MASK:
            tid=j.get("source_mask_tile_id")
            if not isinstance(tid,str) or tid not in mask_tiles:
                raise ValueError("S100 mask X-junction lacks exact S86 tile provenance")
        for port in j.get("ports",[]):
            cid=port.get("component_id")
            idx=port.get("port")
            if cid not in sources or idx not in (0,1):
                raise ValueError("S100 S96 junction points to nonexistent source arm")
            src=sources[cid]
            if source_repr==SOURCE_MASK and src["source_mask_tile_id"]!=tid:
                raise ValueError("S100 mask junction crosses source S86 mask tile identities")
            path=np.asarray(src["points_xy"],dtype=float)
            if path.ndim!=2 or path.shape[1]!=2 or len(path)<4:
                raise ValueError("S100 observed original source arm lacks >=4 points")
            endpoint=path[0] if idx==0 else path[-1]
            if np.linalg.norm(np.asarray(port["point_xy"],float)-endpoint)>0.025:
                raise ValueError("S100 S96 port endpoint was not observed in source S86 pixels")
            interior=path[min(6,len(path)-1)] if idx==0 else path[max(0,len(path)-7)]
            outward=endpoint-interior
            outward/=np.linalg.norm(outward)
            u=np.asarray(port["outward_unit"],float)
            if not np.all(np.isfinite(u)) or np.linalg.norm(u-outward)>0.025:
                raise ValueError("S100 S96 port tangent differs from stored source line")


def _mask_origin_inventory(s86_dir:Path,records:list[dict],s96_dir:Path,report:dict,
                           *,max_records:int):
    meta=report.get("optional_alternative_mask_junction_audit")
    if meta is None:
        return [],{},{}
    if (meta.get("status")!="ALTERNATIVE_PIXEL_BRANCH_HYPOTHESES_PRIVATE_ONLY"
        or meta.get("source_S86_mask_tiles_changed")!=0
        or meta.get("source_S86_fragment_count_modified")!=0
        or meta.get("all_original_S86_fragment_IDs_preserved") is not True
        or meta.get("source_S86_tiles_examined")!=len(records)):
        raise ValueError("S100 mask-derived S96 evidence has invalid source-state summary")
    arms=_read_ndjson(s96_dir/"mask_derived_virtual_arms.ndjson",limit=max_records*10)
    junctions=_read_ndjson(s96_dir/"mask_derived_junctions.ndjson",limit=max_records)
    pairs=_read_ndjson(s96_dir/"mask_derived_visual_pairs_unknown.ndjson",
                       limit=max_records*4)
    if (len(arms)!=meta["source_skeleton_arm_hypotheses"]
        or len(junctions)!=meta["source_mask_junction_candidates"]
        or len(pairs)!=meta["four_arm_crossing_pair_hypotheses"]):
        raise ValueError("S100 source-derived alternative arm/junction NDJSON count mismatch")
    records_by_tile={x["tile_id"]:x for x in records}
    cache={}
    lookup={}
    for row in arms:
        cid=row.get("component_id")
        tid=row.get("source_mask_tile_id")
        if cid in lookup or tid not in records_by_tile:
            raise ValueError("S100 duplicated/non-source S96 mask-derived arm")
        record=records_by_tile[tid]
        if (row.get("derived_from_mask_sha256")!=record["mask_sha256"]
            or row.get("source_core_xyxy")!=record["core_xyxy"]
            or row.get("source_core_skeleton_observed_pixels_only") is not True
            or row.get("original_S86_fragments_replaced") is not False):
            raise ValueError("S100 mask-derived arm was not derived from exact S86 shard")
        if tid not in cache:
            p=s86_dir/record["mask_file"]
            if not p.is_file() or sha256_file(p)!=record["mask_sha256"]:
                raise ValueError("S100 S86 mask tile changed during source X audit")
            mask=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            x0,y0,x1,y1=record["core_xyxy"]
            if (mask is None or mask.shape!=(y1-y0,x1-x0)
                or not np.isin(mask,(0,255)).all()):
                raise ValueError("S100 original S86 mask is not binary foreground")
            cache[tid]=mask
        pixels=np.asarray(row.get("points_xy"),float)
        if (pixels.ndim!=2 or pixels.shape[1]!=2 or len(pixels)<4
            or not np.isfinite(pixels).all()):
            raise ValueError("S100 nonfinite mask-derived source arm")
        x0,y0,x1,y1=record["core_xyxy"]
        xy=np.rint(pixels).astype(int)
        if (np.max(np.abs(xy-pixels))>.025
            or not ((xy[:,0]>=x0)&(xy[:,0]<x1)
                     &(xy[:,1]>=y0)&(xy[:,1]<y1)).all()
            or not np.all(cache[tid][xy[:,1]-y0,xy[:,0]-x0]==255)):
            raise ValueError("S100 mask-derived source arm includes non-observed pixel")
        lookup[cid]=row
    _verify_junction_ports(junctions,lookup,source_repr=SOURCE_MASK,mask_tiles=cache)
    return junctions,lookup,{
        "source_alternative_mask_virtual_arms_exact_foreground_verified":len(arms),
        "alternative_mask_tile_sha256s_reverified":len(cache),
        "source_alternative_mask_junctions_verified":len(junctions),
        "S96_alternative_mask_original_components_replaced":False,
    }


def _source_crop_and_overlay(bmp:GrayBMP8,geometry:dict,raw:dict,
                             source_paths:dict,*,crop_size:int):
    center=tuple(geometry["center_xy_source"])
    gray,loc=bmp.crop(center,size=crop_size)
    origin=np.array(loc.original_xyxy[:2],dtype=float)
    overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    ports=raw["ports"]
    accepted=geometry["cross_shape_review_status"]==ACCEPT
    pairing=geometry["paired_source_arm_ids_proposed_visual_only"]
    owner_color={}
    if accepted:
        for i,group in enumerate(pairing):
            for cid in group:
                owner_color[cid]=((255,210,50),(45,190,85))[i]
    for port in ports:
        cid=port["component_id"]
        path=source_paths[cid]
        xy=np.asarray(path["points_xy"],float)
        coords=np.rint(xy-origin).astype(np.int32)
        cv2.polylines(overlay,[coords.reshape(-1,1,2)],False,
                      owner_color.get(cid,(25,120,245)),1,lineType=cv2.LINE_8)
        endpoint=np.rint(np.asarray(port["point_xy"],float)-origin).astype(int)
        cv2.circle(overlay,tuple(endpoint),3,(200,30,200),1,lineType=cv2.LINE_8)
    c=np.rint(np.asarray(center)-origin).astype(int)
    cv2.circle(overlay,tuple(c),8,(210,25,220),1,lineType=cv2.LINE_8)
    # Do NOT draw a fake bridging source centerline through an unseen vertex.
    compare=np.concatenate([cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),overlay],axis=1)
    return gray,compare,list(loc.original_xyxy)


def _save_png(path,pixels):
    ok,b=cv2.imencode(".png",pixels,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not ok:raise OSError("S100 original-source image PNG cannot be encoded")
    _write_private(path,b.tobytes())
    return sha256_file(path)


def _pick_review_cases(candidates,max_cases):
    if type(max_cases) is not int or not 0<=max_cases<=48:
        raise ValueError("S100 source review cases must be 0..48")
    # Machine-selected cases are diagnostic only, not randomly sampled native
    # frame pixels. Preserve different failure modes and representations.
    strata={}
    for x in candidates:
        key=(x["source_representation"],x["cross_shape_review_status"])
        strata.setdefault(key,[]).append(x)
    keys=sorted(strata)
    for key in keys:
        strata[key].sort(key=lambda x:x["candidate_id"])
    chosen=[]
    while len(chosen)<max_cases and any(strata.values()):
        for key in keys:
            if strata[key] and len(chosen)<max_cases:
                chosen.append(strata[key].pop(0))
    return chosen


def run_s100_unlabeled_cross_review(
    *,s86_dir:Path,s92_dir:Path,s95_dir:Path,s96_dir:Path,
    bmp_path:Path,private_out_dir:Path,
    case_name:str="H50Z_REAL",max_review_cases:int=12,
    crop_size:int=320,cfg:XShapeConfig=XShapeConfig(),
):
    cfg.validate()
    if type(crop_size) is not int or not 160<=crop_size<=512 or crop_size%16:
        raise ValueError("S100 actual-pixel preview crop must be 160..512px, 16-aligned")
    s86_dir=Path(s86_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    s95_dir=Path(s95_dir).resolve()
    s96_dir=Path(s96_dir).resolve()
    bmp_path=Path(bmp_path).resolve()
    private_out_dir=Path(private_out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    parents=[s86_dir,s92_dir,s95_dir,s96_dir,bmp_path]
    if (private_out_dir.exists() or private_out_dir==repo
        or repo in private_out_dir.parents
        or any(private_out_dir==p or p in private_out_dir.parents
               or private_out_dir in p.parents for p in parents)):
        raise ValueError("S100 requires a NEW PRIVATE source-separated destination")
    # S86 masks and every S92/S95/S96 exact SHA-256 from one original picture.
    verified=verify_stage_receipts(
        case_name=case_name,s86_dir=s86_dir,s92_dir=s92_dir,
        s95_dir=s95_dir,s96_dir=s96_dir,original_bmp=bmp_path
    )
    manifest,records,s86sha=_read_source(s86_dir)
    observed,_=collapse_observed_components(records)
    lookup={x["component_id"]:x for x in observed}
    if len(lookup)!=len(observed):
        raise ValueError("S100 source S86/S87 observed component duplicate identity")
    s96= json.loads((s96_dir/"report.json").read_text(encoding="utf-8"))
    rows=_read_ndjson(s96_dir/"junction_hypotheses.ndjson",
                      limit=cfg.max_junction_records)
    if len(rows)!=s96["geometry"].get("candidate_junctions",0):
        raise ValueError("S100 original S96 junction inventory differs from machine report")
    _verify_junction_ports(rows,lookup,source_repr=SOURCE_ORIGINAL)
    masks,mask_lookup,mask_integrity=_mask_origin_inventory(
        s86_dir,records,s96_dir,s96,max_records=cfg.max_junction_records
    )
    source_sha=verified["original_bmp_sha256"]
    candidates,summary=build_x_shape_candidate_inventory(
        rows,masks,source_bmp_sha256=source_sha,cfg=cfg
    )
    if sum(x["cross_shape_review_status"]==ACCEPT for x in candidates)!=summary[
        "four_arm_geometric_X_review_candidates"
    ]:
        raise RuntimeError("S100 cross geometry summary/count mismatch")
    for c in candidates:
        if c["cross_shape_review_status"]==ACCEPT and c[
            "source_representation"
        ]==SOURCE_ORIGINAL:
            assert all(cid in lookup for pair in
                       c["paired_source_arm_ids_proposed_visual_only"] for cid in pair)
    original_sha=sha256_file(bmp_path)
    if original_sha!=source_sha:
        raise ValueError("S100 original photo changed before native preview")
    bmp=GrayBMP8(bmp_path)
    if [bmp.height,bmp.width]!=manifest["image_shape_yx"]:
        raise ValueError("S100 original photo dimension receipt mismatch")
    picked=_pick_review_cases(candidates,max_review_cases)
    ids_by_mode={(SOURCE_ORIGINAL,x["junction_id"]):x for x in rows}
    ids_by_mode.update({(SOURCE_MASK,x["junction_id"]):x for x in masks})
    private_out_dir.mkdir(parents=True,mode=0o700)
    private_out_dir.chmod(0o700)
    (private_out_dir/"images").mkdir(mode=0o700)
    previews=[]
    for i,c in enumerate(picked):
        mode=c["source_representation"]
        raw=ids_by_mode[(mode,c["source_s96_junction_id"])]
        sources=lookup if mode==SOURCE_ORIGINAL else mask_lookup
        original,side,xyxy=_source_crop_and_overlay(
            bmp,c,raw,sources,crop_size=crop_size
        )
        oid=f"images/case_{i+1:03d}_original.png"
        mid=f"images/case_{i+1:03d}_observed_source_overlay.png"
        osha=_save_png(private_out_dir/oid,original)
        msha=_save_png(private_out_dir/mid,side)
        previews.append({
            "candidate_id":c["candidate_id"],
            "junction_id":c["source_s96_junction_id"],
            "source_representation":mode,
            "source_xyxy":xyxy,
            "geometric_cross_status":c["cross_shape_review_status"],
            "original_photo_png":oid,"original_photo_png_sha256":osha,
            "observed_source_overlay_png":mid,
            "observed_source_overlay_png_sha256":msha,
            "left_panel_bmp_source_pixels_exact":True,
            "right_panel_arms_from_observed_S86_centerlines_or_mask_only":True,
            "no_center_bridge_or_same_particle_identity_drawn":True,
            "machine_selected_example_not_blind_source_ROI":True,
        })
    summary.update({
        "status":"SOURCE_SHA_VERIFIED_ORIGINAL_BMP_MACHINE_X_SHAPE_REVIEW_NOT_GT",
        "source_bmp_name":verified["original_bmp_name"],
        "original_bmp_bytes_SHA_verified":True,
        "S86_manifest_sha256":sha256_file(s86_dir/"manifest.json"),
        "S92_report_sha256":sha256_file(s92_dir/"report.json"),
        "S95_report_sha256":sha256_file(s95_dir/"report.json"),
        "S96_report_sha256":sha256_file(s96_dir/"report.json"),
        "actual_original_film_previews_exported":len(previews),
        "requested_max_review_cases":max_review_cases,
        "requested_crop_size_px":crop_size,
        "alternative_mask_source_integrity":mask_integrity,
        "S86_S92_S95_S96_real_saved_artifact_lineage_verified":True,
        "not_independent_fiducial_detection_accuracy":True,
        "not_real_photo_F1_or_particle_GT":True,
        "machine_class_stratified_photo_cases_NOT_population_random":True,
        "privately_saved_full_source_S86_S96_counts":verified,
    })
    _write_private(private_out_dir/"candidates.ndjson",_ndjson(candidates))
    _write_private(private_out_dir/"report.json",_json(summary))
    _write_private(private_out_dir/"preview_receipts.json",_json({
        "schema":SCHEMA,"source_original_sha256":source_sha,"cases":previews
    }))
    html_sections=[
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>H50Z S100 — геометрические X без f1..f8</title>"
        "<style>body{font:14px system-ui;margin:20px;color:#20252b}"
        "p{max-width:970px}img{max-width:100%;height:auto;border:1px solid #ccc}"
        "section{border-top:1px solid #aaa;padding:14px 0}</style>"
        "<h1>Кандидаты геометрического X без заранее размеченных крестов</h1>"
        "<p>S100 показывает машинные предложения: где четыре наблюдаемых "
        "луча могли образовать X. Репер/два наложенных трека/столкновение "
        "НЕ различаются по этой 2D геометрии. Исходный BMP СЛЕВА "
        "побайтно не меняется при анализе, справа те же пиксели с "
        "наблюдаемыми исходными ветвями (без придуманного мостика).</p>"
        "<p>Входы: S86→S92→S95→S96, отдельный исходный BMP SHA-256. "
        "Цифры означают количество машинных кандидатов, "
        "не число физических треков. Примеры выбраны алгоритмом "
        "по типу ошибки и НЕ являются слепой популяционной выборкой.</p>"
    ]
    for x in previews:
        html_sections.extend([
            "<section><h3>"+html.escape(x["candidate_id"])+" — "+
            html.escape(x["geometric_cross_status"])+"</h3>",
            "<p>Представление: "+html.escape(x["source_representation"])+
            "; исходный прямоугольник: "+
            html.escape(str(x["source_xyxy"]))+"</p>",
            "<img src='"+html.escape(x["observed_source_overlay_png"])+"' "+
            "alt='original source pixels left, observed machine branches right' />",
            "</section>"
        ])
    html_sections.append("</html>")
    _write_private(private_out_dir/"index.html","".join(html_sections).encode("utf-8"))
    # A compact diagnosis-only ZIP, never the 200MB original BMP.
    archive=private_out_dir/"S100_REAL_SOURCE_X_SHAPE_DIAGNOSIS_PRIVATE.zip"
    with ZipFile(archive,"x",compression=ZIP_DEFLATED,compresslevel=6) as z:
        for source in sorted(private_out_dir.rglob("*")):
            if not source.is_file() or source==archive:continue
            z.write(source,arcname=str(source.relative_to(private_out_dir)))
    archive.chmod(0o600)
    if archive.stat().st_size>32*1024*1024:
        raise ValueError("S100 image review archive exceeds 32 MB, use fewer cases")
    if sha256_file(bmp_path)!=original_sha:
        raise ValueError("S100 original BMP changed during source-image export")
    if sha256_file(s86_dir/"manifest.json")!=summary["S86_manifest_sha256"]:
        raise ValueError("S100 original S86 manifest changed while exporting image evidence")
    return {**summary,
            "compact_source_real_image_evidence_zip":archive.name,
            "compact_source_real_image_evidence_zip_sha256":sha256_file(archive),
            "compact_zip_compressed_bytes":archive.stat().st_size}



def verify_existing_s100_gallery(
    *,bmp_path:Path,s86_dir:Path,s92_dir:Path,
    s95_dir:Path,s96_dir:Path,
    out_dir:Path,max_review_cases:int,crop_size:int,
):
    """Reopen a COMPLETE native-photo S100 output without modifying inputs.

    Original source pixels in every PNG and in the left half of every
    overlay must exactly match the same SHA-verified uncompressed BMP.
    Sealed zip must contain byte-identical members; no source-BMP leakage.
    """
    folder=Path(out_dir).resolve()
    report_file=folder/"report.json"
    preview_file=folder/"preview_receipts.json"
    zip_path=folder/"S100_REAL_SOURCE_X_SHAPE_DIAGNOSIS_PRIVATE.zip"
    mandatory=(report_file,preview_file,folder/"candidates.ndjson",
               folder/"index.html",zip_path)
    if any(not p.is_file() for p in mandatory):
        raise ValueError("Existing S100 gallery is incomplete: use a NEW private output")
    report=json.loads(report_file.read_text(encoding="utf-8"))
    preview=json.loads(preview_file.read_text(encoding="utf-8"))
    sha=sha256_file(bmp_path)
    if (report.get("schema")!=SCHEMA
        or report.get("status")!="SOURCE_SHA_VERIFIED_ORIGINAL_BMP_MACHINE_X_SHAPE_REVIEW_NOT_GT"
        or report.get("source_bmp_sha256")!=sha
        or report.get("source_bmp_name")!=Path(bmp_path).name
        or report.get("S86_manifest_sha256")!=sha256_file(Path(s86_dir)/"manifest.json")
        or report.get("S92_report_sha256")!=sha256_file(Path(s92_dir)/"report.json")
        or report.get("S95_report_sha256")!=sha256_file(Path(s95_dir)/"report.json")
        or report.get("S96_report_sha256")!=sha256_file(Path(s96_dir)/"report.json")
        or report.get("requested_max_review_cases")!=max_review_cases
        or report.get("requested_crop_size_px")!=crop_size
        or preview.get("source_original_sha256")!=sha
        or len(preview.get("cases",[]))!=report.get("actual_original_film_previews_exported")):
        raise ValueError("Existing S100 source/model/crop evidence differs from exact BMP or source stages")
    bmp=GrayBMP8(Path(bmp_path))
    if [bmp.height,bmp.width]!=report.get("source_shape_yx",[
        bmp.height,bmp.width
    ]):
        raise ValueError("S100 saved original source image dimensions differ")
    for case in preview["cases"]:
        x0,y0,x1,y1=case["source_xyxy"]
        if (x1-x0!=crop_size or y1-y0!=crop_size
            or x0<0 or y0<0 or x1>bmp.width or y1>bmp.height):
            raise ValueError("S100 old crop went outside original photo")
        o=folder/case["original_photo_png"]
        p=folder/case["observed_source_overlay_png"]
        if (folder not in o.resolve().parents or folder not in p.resolve().parents
            or not o.is_file() or not p.is_file()
            or sha256_file(o)!=case["original_photo_png_sha256"]
            or sha256_file(p)!=case["observed_source_overlay_png_sha256"]):
            raise ValueError("S100 old native photo overlay image has changed")
        original=cv2.imread(str(o),cv2.IMREAD_GRAYSCALE)
        overlay=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if (original is None or overlay is None
            or original.shape!=(crop_size,crop_size)
            or overlay.shape!=(crop_size,2*crop_size,3)):
            raise ValueError("S100 previously exported native photos have invalid dimensions")
        matched,_=bmp.crop(((x0+x1)/2.,(y0+y1)/2.),size=crop_size)
        if (not np.array_equal(matched,original)
            or not np.array_equal(
                cv2.cvtColor(overlay[:,:crop_size],cv2.COLOR_BGR2GRAY),
                matched
            )):
            raise ValueError("S100 saved original-image pixels differ from true source BMP")
    allowed=[
        p for p in folder.rglob("*")
        if p.is_file() and p!=zip_path
    ]
    with ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise ValueError("S100 old compact ZIP CRC check failed")
        names=set(z.namelist())
        expected={str(p.relative_to(folder)) for p in allowed}
        if names!=expected or any(x.lower().endswith(
            (".bmp",".pt",".pth",".7z",".npy")) for x in names):
            raise ValueError("S100 saved ZIP membership is incomplete or leaks a private BMP")
        for p in allowed:
            if z.read(str(p.relative_to(folder)))!=p.read_bytes():
                raise ValueError("S100 compressed source image is not byte-identical")
    return {
        **report,
        "compact_source_real_image_evidence_zip":zip_path.name,
        "compact_source_real_image_evidence_zip_sha256":sha256_file(zip_path),
        "compact_zip_compressed_bytes":zip_path.stat().st_size,
        "saved_S100_original_source_images_reopened_and_pixel_exact":True,
    }
