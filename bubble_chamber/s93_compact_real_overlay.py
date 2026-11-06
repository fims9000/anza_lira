"""S93: one-command *real* H50Z BMP -> S86 -> S92 -> compact visual evidence.

No upload to GitHub. Original BMP is mmap-read, never decoded into a full
in-memory RGB frame. Export only requested reduced-resolution JPGs and
small original-resolution comparison crops; S86/S92 source and provenance are
verified. Never call machine candidate lines verified physical particle tracks.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8, sha256_file
from .full_bmp_segmentation import FullBMPConfig, run_full_bmp
from .s92_curve_geometry import TraceReviewConfig
from .s92_curve_review import run_s92


RESULT_SCHEMA="S93_PRIVATE_REAL_BMP_COMPACT_IMAGE_V1"


def _private(path:Path):
    path.chmod(0o600)


def _read_json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf8"))


def _read_jsonl(path:Path):
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def low_memory_gray_preview(bmp:GrayBMP8,*,max_side:int=2048)->tuple[np.ndarray,float]:
    """Read just sampled source pixels from disk-backed BMP; never materialize it."""
    if not isinstance(max_side,int) or not 512<=max_side<=4096:
        raise ValueError("max_side must be integer in 512..4096")
    ratio=min(1.,max_side/max(bmp.width,bmp.height))
    width=max(1,int(round(bmp.width*ratio)))
    height=max(1,int(round(bmp.height*ratio)))
    # Nearest-neighbor sampling can delete a 1px film trace entirely when
    # the image is reduced 10-15x. Instead read ONLY narrow source row bands
    # from the disk-backed BMP, average pixels into one output row and keep
    # total RAM bounded by a single stripe (< 2 MiB for H50Z dimensions).
    out=np.empty((height,width),dtype=np.uint8)
    for i in range(height):
        start=min(bmp.height-1,int(math.floor(i*bmp.height/height)))
        stop=min(bmp.height,int(math.ceil((i+1)*bmp.height/height)))
        stop=max(start+1,stop)
        source_band=np.ascontiguousarray(
            bmp._data[start:stop,:bmp.width],dtype=np.uint8
        )
        out[i,:]=cv2.resize(
            source_band,(width,1),interpolation=cv2.INTER_AREA
        )[0]
    return out,ratio


def _colorized_gray(gray:np.ndarray)->np.ndarray:
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


def _sample_draw_points(xy,scale,width,height):
    data=np.asarray(xy,dtype=np.float64)
    if data.ndim!=2 or data.shape[1]!=2 or len(data)<2 or not np.isfinite(data).all():
        raise ValueError("S92 geometry must contain >=2 finite source-coordinate points")
    xy_scaled=np.rint(data*scale).astype(np.int32)
    xy_scaled[:,0]=np.clip(xy_scaled[:,0],0,width-1)
    xy_scaled[:,1]=np.clip(xy_scaled[:,1],0,height-1)
    return xy_scaled


def export_compact_overview(
    bmp:GrayBMP8,
    s92_dir:Path,
    destination:Path,
    *,
    max_side:int=2048,
    max_detail_cases:int=6,
)->dict:
    """Export side-by-side small photo/overlay and event details. No fake result."""
    s92_dir=Path(s92_dir)
    destination=Path(destination)
    repo=Path(__file__).resolve().parents[1]
    if repo==destination or repo in destination.parents:
        raise ValueError("Preview outputs must remain PRIVATE and outside Git")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Compact image output is immutable")
    report=_read_json(s92_dir/"report.json")
    if (report.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY"
        or report.get("physical_auto_joins_authorized") is not False
        or report.get("source_bmp_sha256")!=sha256_file(bmp.path)
        or report.get("image_shape_yx")!=[bmp.height,bmp.width]):
        raise ValueError("S92 input is incomplete or refers to another BMP")
    if not isinstance(max_detail_cases,int) or not 0<=max_detail_cases<=12:
        raise ValueError("max_detail_cases must be in 0..12")

    gray,scale=low_memory_gray_preview(bmp,max_side=max_side)
    rgb=_colorized_gray(gray)
    overlay=rgb.copy()
    h,w=gray.shape
    primary=list(_read_jsonl(s92_dir/"priority_view_observed_only.ndjson"))
    allinfo={
        row["component_id"]:row
        for row in _read_jsonl(s92_dir/"observed_shape_review.ndjson")
    }
    if len(allinfo)!=int(report["observed_components_after_S87"]):
        raise ValueError("S92 source component-count mismatch")
    if len(primary)!=int(report["primary_view_observed_components"]):
        raise ValueError("S92 source primary layer-count mismatch")

    observed_count=0
    coil_count=0
    for item in primary:
        info=allinfo.get(item["component_id"])
        if info is None or info.get("show_in_primary_view") is not True:
            raise ValueError("S92 filtered source IDs do not match shape review")
        spiral=info["shape_review_category"]=="COILING_TRACE_REVIEW"
        coords=_sample_draw_points(item["observed_only_centerline_xy"],
                                   scale,w,h)
        if spiral:
            coil_count+=1
        observed_count+=1
        # OpenCV BGR: cyan ordinary observed, green spiral candidate.
        color=(40,220,80) if spiral else (255,230,55)
        cv2.polylines(overlay,[coords.reshape(-1,1,2)],False,
                      color,2 if spiral else 1,lineType=cv2.LINE_AA)

    proposed=0
    for item in _read_jsonl(s92_dir/"gap_proposals_unknown.ndjson"):
        if item.get("review_decision")!="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
            continue
        if item.get("status")!="UNKNOWN" or item.get("physical_auto_join") is not False:
            raise ValueError("Unsafe source gap decision")
        coords=_sample_draw_points(item["inferred_bridge_points_xy"],
                                   scale,w,h)
        # Dashed magenta: hypothetical gap ONLY. No filled mask pixels.
        for start in range(0,len(coords)-1,4):
            finish=min(len(coords)-1,start+2)
            if finish>start:
                cv2.line(overlay,tuple(coords[start]),tuple(coords[finish]),
                         (235,65,240),2,lineType=cv2.LINE_AA)
        proposed+=1

    if not (overlay.shape==rgb.shape and overlay.shape[0]>0):
        raise RuntimeError("Bad compact preview geometry")
    destination.mkdir(parents=True,mode=0o700,exist_ok=True)
    original_file=destination/"H50Z_original_small.jpg"
    overlay_file=destination/"H50Z_S92_annotated_small.jpg"
    options=[cv2.IMWRITE_JPEG_QUALITY,91]
    if (not cv2.imwrite(str(original_file),rgb,options)
        or not cv2.imwrite(str(overlay_file),overlay,options)):
        raise OSError("Failed compact source/overlay JPEG export")
    _private(original_file)
    _private(overlay_file)

    # Small contact sheet of original 384px image crops against the *real*
    # S92 generated overlays. Reject missing crops rather than drawing
    # a pseudo-annotation from a generic stock/demo image.
    cases=_read_json(s92_dir/"preview_manifest.json")
    selected=cases[:max_detail_cases]
    crops=[]
    descriptions=[]
    for case in selected:
        source=s92_dir/case["source"]
        predicted=s92_dir/case["overlay"]
        if not source.is_file() or not predicted.is_file():
            raise FileNotFoundError("S92 case preview missing")
        a=cv2.imread(str(source),cv2.IMREAD_COLOR)
        b=cv2.imread(str(predicted),cv2.IMREAD_COLOR)
        if a is None or b is None or a.shape!=b.shape or a.ndim!=3:
            raise ValueError("S92 photo and overlay case dimensions mismatch")
        # Keep original pixels in detail case comparisons; no resize.
        side=np.concatenate([a,b],axis=1)
        cv2.putText(side,case["kind"],(10,22),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,255),2)
        crops.append(side)
        descriptions.append({"case_id":case["case_id"],
                             "kind":case["kind"],
                             "crop_xyxy_in_original_BMP":case["image_bounds_xyxy"]})
    sheet_file=None
    if crops:
        width=max(item.shape[1] for item in crops)
        rows=[]
        for item in crops:
            pad=width-item.shape[1]
            rows.append(cv2.copyMakeBorder(item,0,0,0,pad,
                                           cv2.BORDER_CONSTANT,value=(10,10,10)))
        sheet=np.concatenate(rows,axis=0)
        sheet_file=destination/"H50Z_S92_details.jpg"
        if not cv2.imwrite(str(sheet_file),sheet,options):
            raise OSError("Failed S92 detail contact sheet")
        _private(sheet_file)

    manifest={
        "schema":RESULT_SCHEMA,
        "method":report["source_s86_method"],
        "source_bmp_name":report["source_bmp_name"],
        "source_bmp_sha256":report["source_bmp_sha256"],
        "source_shape_yx":[bmp.height,bmp.width],
        "output_shape_yx":[h,w],
        "scale_x":w/bmp.width,
        "scale_y":h/bmp.height,
        "observed_components_colored":observed_count,
        "coiling_components_colored_green":coil_count,
        "provisional_gaps_dashed_magenta":proposed,
        "original_small_file":original_file.name,
        "annotated_small_file":overlay_file.name,
        "detail_comparison_file":sheet_file.name if sheet_file else None,
        "review_cases":descriptions,
        "original_small_sha256":sha256_file(original_file),
        "annotated_small_sha256":sha256_file(overlay_file),
        "details_sha256":sha256_file(sheet_file) if sheet_file else None,
        "machine_predictions_only":True,
        "physical_auto_joins_authorized":False,
        "unobserved_bridge_pixels_never_presented_as_source":True,
        "source_photo_not_edited":True,
        "note":"Reduced image is for human review; full-res PNG crops carry fine detail.",
    }
    manifest_file=destination/"compact_preview_manifest.json"
    with manifest_file.open("x",encoding="utf8") as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    _private(manifest_file)
    return manifest


def run_one_command(
    bmp_path:Path,
    private_root:Path,
    *,
    method:str="s88-steerable",
    checkpoint:Path|None=None,
    max_side:int=2048,
    max_gap_px:float=26.,
    preview_limit:int=12,
    resume:bool=False,
    labels_path:Path|None=None,
    min_segment_length_px:float=12.,
    min_component_area:int=15,
    max_fragments_per_tile:int=512,
)->dict:
    """Reuse verified existing S86/S92 output; never silently overwrite."""
    bmp_path=Path(bmp_path).resolve()
    private_root=Path(private_root).resolve()
    repo=Path(__file__).resolve().parents[1]
    if private_root==repo or repo in private_root.parents:
        raise ValueError("Private result must be outside Git")
    if not bmp_path.is_file():
        raise FileNotFoundError(bmp_path)
    if labels_path is not None:
        labels_path=Path(labels_path).resolve()
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)
    if method not in ("s88-steerable","frangi","contrast","azunet"):
        raise ValueError("Unsupported source segmentation method")
    if method=="azunet" and checkpoint is None:
        raise ValueError("AZUNet requires a trained trusted source-locked checkpoint")
    if method!="azunet" and checkpoint is not None:
        raise ValueError("Checkpoint only permitted for AZUNet")
    if preview_limit<0 or preview_limit>64:
        raise ValueError("Preview limit must be 0..64")
    if (not math.isfinite(min_segment_length_px)
        or min_segment_length_px<4 or min_segment_length_px>200
        or not 1<=min_component_area<=100
        or not 1<=max_fragments_per_tile<=4096):
        raise ValueError("Invalid S86 candidate extraction sensitivity")
    if private_root.exists() and not resume:
        raise FileExistsError("Private output exists; pass --resume after confirming same BMP")
    private_root.mkdir(mode=0o700,parents=True,exist_ok=True)
    private_root.chmod(0o700)
    s86_dir=private_root/"S86"
    s92_dir=private_root/"S92"
    compact=private_root/"SMALL_IMAGE"
    bmp=GrayBMP8(bmp_path)
    source_sha=sha256_file(bmp_path)
    run_identity=private_root/"run_identity.json"
    identity={
        "schema":RESULT_SCHEMA,
        "input_bmp_sha256":source_sha,
        "input_bmp_name":bmp_path.name,
        "source_shape_yx":[bmp.height,bmp.width],
        "method":method,
        "checkpoint_sha256":sha256_file(checkpoint) if checkpoint else None,
        "sparse_labels_sha256":sha256_file(labels_path) if labels_path else None,
        "max_side":max_side,
        "max_gap_px":max_gap_px,
        "preview_limit":preview_limit,
        "min_segment_length_px":min_segment_length_px,
        "min_component_area":min_component_area,
        "max_fragments_per_tile":max_fragments_per_tile,
        "S86_config":("s88_calibration_2026_11_synthetic_not_real_GT"),
    }
    if run_identity.exists():
        if _read_json(run_identity)!=identity:
            raise ValueError("Source photo, method, checkpoint or params changed on resume")
    elif any(private_root.iterdir()):
        raise ValueError("Nonempty unverified private run directory")
    else:
        with run_identity.open("x",encoding="utf8") as file:
            json.dump(identity,file,ensure_ascii=False,indent=2)
            file.write("\n")
        _private(run_identity)

    config=FullBMPConfig(
        method=method,core_size=384,halo=48,preview_limit=0,
        s88_low_snr=.816,s88_high_snr=1.7,
        min_segment_length_px=min_segment_length_px,
        min_component_area=min_component_area,
        max_fragments_per_tile=max_fragments_per_tile,
    )
    s86_result=run_full_bmp(
        bmp_path,s86_dir,config,checkpoint=checkpoint,resume=resume
    )
    if s86_result.get("status")!="COMPLETE_MACHINE_ONLY":
        raise RuntimeError("S86 output is incomplete")
    if s86_result["source_bmp_sha256"]!=source_sha:
        raise ValueError("S86 source BMP SHA mismatch")
    cfg=TraceReviewConfig(max_gap_px=max_gap_px)
    if s92_dir.exists():
        if not resume or not (s92_dir/"report.json").is_file():
            raise ValueError("S92 unfinished; preserve evidence and choose a new output directory")
        s92_result=_read_json(s92_dir/"report.json")
        if (s92_result.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY"
            or s92_result.get("source_bmp_sha256")!=source_sha
            or s92_result.get("sparse_labels_sha256")!=identity["sparse_labels_sha256"]):
            raise ValueError("S92 resume provenance mismatch")
    else:
        s92_result=run_s92(
            s86_dir,s92_dir,bmp_path=bmp_path,cfg=cfg,
            preview_limit=preview_limit,export_priority_tiles=False,
            labels_path=labels_path
        )
    if compact.exists():
        if not resume or not (compact/"compact_preview_manifest.json").is_file():
            raise FileExistsError("Compact output incomplete; use a new private root")
        manifest=_read_json(compact/"compact_preview_manifest.json")
        if (manifest.get("source_bmp_sha256")!=source_sha
            or not (compact/manifest.get("annotated_small_file","missing")).is_file()
            or sha256_file(compact/manifest["annotated_small_file"])!=
               manifest["annotated_small_sha256"]):
            raise ValueError("Compact image resume SHA mismatch")
    else:
        manifest=export_compact_overview(
            bmp,s92_dir,compact,max_side=max_side,
            max_detail_cases=min(preview_limit,6),
        )
    return {
        "status":"COMPLETE_REAL_SOURCE_MACHINE_OVERLAY_NOT_PHYSICAL_GROUND_TRUTH",
        "source":bmp_path.name,
        "source_SHA256":source_sha,
        "s86_candidate_fragments":s86_result["candidate_fragments"],
        "s92_shape_categories":s92_result["shape_review_categories"],
        "small_annotated_photo":str(compact/manifest["annotated_small_file"]),
        "small_original_photo":str(compact/manifest["original_small_file"]),
        "comparison_montage":(str(compact/manifest["detail_comparison_file"])
                              if manifest["detail_comparison_file"] else None),
        "output_root":str(private_root),
        "physical_auto_joins_authorized":False,
    }
