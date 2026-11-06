"""S95 complete-source observed fragment typing and optional image overlay.

Input: frozen S86, verified S92 and optional exact-photo human operator labels.
S95 NEVER deletes even one S86 fragment and ALWAYS emits UNKNOWN physical
identity. 'Track', 'artifact', 'bundle', 'coil', and 'fiducial cross' are
REVIEW LAYERS, not binary ground truth or clinical/physical event claims.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import html
import json

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .s87_seam_continuity import _read_source
from .s92_curve_review import collapse_observed_components,_write_private,_json,_ndjson
from .s93_compact_real_overlay import low_memory_gray_preview,_sample_draw_points
from .s94_sparse_keypoint_guard import parse_photo_labels
from .s95_operator_curves import validate_operator_curve_package
from .s95_all_structure_types import (
    PALETTE_BGR,type_all_observed_components,human_mark_objects,
)

SCHEMA="S95_PRIVATE_ALL_STRUCTURES_REVIEW_V1"


def _read_ndjson(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def _render_all(gray,components,typed,marks,curves,gaps,scale_x,scale_y):
    photo=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    picture=photo.copy()
    height,width=gray.shape
    # Do not omit artifacts, unknowns or close-to-start short lines.
    for component,row in zip(components,typed):
        p=np.asarray(component["points_xy"],float)
        coordinates=np.rint(p*np.array([scale_x,scale_y])).astype("int32")
        coordinates[:,0]=np.clip(coordinates[:,0],0,width-1)
        coordinates[:,1]=np.clip(coordinates[:,1],0,height-1)
        k=row["primary_review_category"]
        cv2.polylines(picture,[coordinates.reshape(-1,1,2)],False,
                      PALETTE_BGR[k],2 if k in (
                          "COILING_TRACE_CANDIDATE",
                          "BUNDLE_OR_INTERSECTION_REVIEW",
                      ) else 1,lineType=cv2.LINE_AA)
    for gap in gaps:
        if gap.get("review_decision")!="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
            continue
        if gap.get("physical_auto_join") is not False or gap.get("status")!="UNKNOWN":
            raise ValueError("Unsafe physical annotation in S92 gap proposal")
        points=np.asarray(gap["inferred_bridge_points_xy"],dtype=float)
        coords=np.rint(points*np.array([scale_x,scale_y])).astype("int32")
        for j in range(0,len(coords)-1,4):
            end=min(j+2,len(coords)-1)
            if end>j:
                cv2.line(picture,tuple(coords[j]),tuple(coords[end]),
                         (235,65,240),2,lineType=cv2.LINE_AA)
    for curve in curves:
        guide=np.asarray(curve["sampled_guide_xy_source"],float)
        # Yellow broken strokes = operator-supplied/fitted 2D guide, NEVER
        # claim this was detected by the image model.
        xy=np.rint(guide*np.array([scale_x,scale_y])).astype("int32")
        for i in range(0,len(xy)-1,8):
            b=min(i+4,len(xy)-1)
            if b>i:
                cv2.line(picture,tuple(xy[i]),tuple(xy[b]),(0,240,245),
                         1,lineType=cv2.LINE_AA)
    for mark in marks:
        pt=np.asarray(mark["position_xy"],float)
        x=int(round(pt[0]*scale_x));y=int(round(pt[1]*scale_y))
        x=int(np.clip(x,0,width-1));y=int(np.clip(y,0,height-1))
        if mark["category"]=="HUMAN_FIDUCIAL_MARK_LOCATION":
            # Orange calibration-cross centers marked by the human annotator.
            # Do not claim a new image-based cross detector or mask.
            cv2.line(picture,(x-6,y-6),(x+6,y+6),(0,145,255),2)
            cv2.line(picture,(x-6,y+6),(x+6,y-6),(0,145,255),2)
        else:
            cv2.circle(picture,(x,y),8,(25,230,250),1,lineType=cv2.LINE_AA)
    return photo,picture


def _captioned_montage(original,annotated,*,categories,source_name):
    h,w=original.shape[:2]
    if annotated.shape!=original.shape:
        raise ValueError("Original and annotated preview dimensions differ")
    top=np.full((136,w*2,3),(31,32,35),dtype=np.uint8)
    cv2.putText(top,"LEFT: original film pixels", (12,23),
                cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(top,"RIGHT: S95 machine types / guide hypotheses",(w+12,23),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
    legend=[
        ("Cross region",PALETTE_BGR["CALIBRATION_CROSS_CONTEXT"]),
        ("Track-like",PALETTE_BGR["TRACK_LIKE_CANDIDATE"]),
        ("Artifact-like",PALETTE_BGR["ARTIFACT_LIKE_CANDIDATE"]),
        ("Bundle/crossing",PALETTE_BGR["BUNDLE_OR_INTERSECTION_REVIEW"]),
        ("Coiling",PALETTE_BGR["COILING_TRACE_CANDIDATE"]),
        ("Unknown",PALETTE_BGR["UNKNOWN"]),
    ]
    for i,(name,color) in enumerate(legend):
        col=i%3;line=i//3
        x=16+col*max(160,(w*2)//3)
        y=48+line*28
        cv2.line(top,(x,y),(x+32,y),color,4)
        cv2.putText(top,name,(x+40,y+5),cv2.FONT_HERSHEY_SIMPLEX,
                    .48,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(top,"Yellow dashed = operator guide; magenta gaps are separate UNKNOWN hypotheses.",
                (14,117),cv2.FONT_HERSHEY_SIMPLEX,.42,(215,215,215),1,cv2.LINE_AA)
    return np.vstack([top,np.hstack([original,annotated])])


def run_s95(
    s86_directory:Path,s92_directory:Path,private_output:Path,*,
    bmp_path:Path|None=None,labels_path:Path|None=None,
    operator_curves_path:Path|None=None,max_side:int=2048,
)->dict:
    s86_directory=Path(s86_directory).resolve()
    s92_directory=Path(s92_directory).resolve()
    private_output=Path(private_output).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (private_output==repo or repo in private_output.parents
        or private_output==s86_directory or private_output==s92_directory
        or s86_directory in private_output.parents
        or s92_directory in private_output.parents
        or private_output in s86_directory.parents
        or private_output in s92_directory.parents):
        raise ValueError("S95 new PRIVATE destination must be outside Git/S86/S92")
    if private_output.exists():
        raise FileExistsError("S95 output is immutable")
    manifest,records,manifest_sha=_read_source(s86_directory)
    s92=json.loads((s92_directory/"report.json").read_text(encoding="utf8"))
    if (s92.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY"
        or s92.get("source_s86_manifest_sha256")!=manifest_sha
        or s92.get("source_bmp_sha256")!=manifest["source_bmp_sha256"]):
        raise ValueError("S95 requires same frozen S86 -> S92 provenance")
    components,_=collapse_observed_components(records)
    annotations=_read_ndjson(s92_directory/"observed_shape_review.ndjson")
    if len(annotations)!=len(components):
        raise ValueError("S92 record count mismatch")
    by_id={x["component_id"]:x for x in annotations}
    if len(by_id)!=len(annotations):
        raise ValueError("Duplicated S92 shape records")
    annotations=[by_id[x["component_id"]] for x in components]
    source_name=manifest["source_bmp_name"]
    shape=manifest["image_shape_yx"]

    context=None
    labels_sha=None
    if labels_path is not None:
        labels_path=Path(labels_path).resolve()
        labels_sha=sha256_file(labels_path)
        if s92.get("sparse_labels_sha256") not in (None,labels_sha):
            raise ValueError("Different operator labels than verified S92")
        context=parse_photo_labels(
            json.loads(labels_path.read_text(encoding="utf-8-sig")),
            source_name,shape
        )
    elif s92.get("sparse_labels_sha256") is not None:
        raise ValueError("Supply matching original labels.json to preserve S92 guards")

    operator_sha=None
    guides=[]
    if operator_curves_path is not None:
        operator_curves_path=Path(operator_curves_path).resolve()
        operator_sha=sha256_file(operator_curves_path)
        payload=json.loads(operator_curves_path.read_text(encoding="utf-8-sig"))
        guides=validate_operator_curve_package(
            payload,source_bmp_name=source_name,
            source_bmp_sha256=manifest["source_bmp_sha256"],
            source_shape_yx=shape
        )
    objects,summary=type_all_observed_components(
        components,annotations,keypoint_context=context,
        operator_curves=guides,
    )
    gaps=_read_ndjson(s92_directory/"gap_proposals_unknown.ndjson")
    marks=human_mark_objects(context)
    report={
        "schema":SCHEMA,
        "status":"ALL_S86_EXTRACTED_FRAGMENTS_TYPED_MACHINE_REVIEW_ONLY",
        "source_bmp_name":source_name,
        "source_bmp_sha256":manifest["source_bmp_sha256"],
        "source_s86_manifest_sha256":manifest_sha,
        "source_s92_report_sha256":sha256_file(s92_directory/"report.json"),
        "source_labels_sha256":labels_sha,
        "operator_curves_sha256":operator_sha,
        "operator_guides":len(guides),
        "source_s92_gap_proposals":len(gaps),
        "human_annotation_marker_locations":len(marks),
        "all_extracted_line_categories":summary,
        "source_s86_extracted_fragments":manifest["candidate_fragments"],
        "source_s86_tiles_skipped":manifest["tiles_with_fragment_extraction_skipped"],
        "source_fragments_deleted":0,
        "full_recall_not_verified":True,
        "all_objects_include_short_garbage_like_and_unknown":True,
        "no_physical_track_links_authorized":True,
        "collisions_neutron_origin_unknown":True,
        "not_ground_truth_segmentation":True,
    }
    if bmp_path is not None:
        bmp_path=Path(bmp_path).resolve()
        if sha256_file(bmp_path)!=manifest["source_bmp_sha256"]:
            raise ValueError("S95 source BMP SHA mismatch")
        bmp=GrayBMP8(bmp_path)
        if [bmp.height,bmp.width]!=shape:
            raise ValueError("S95 original photo shape mismatch")
        gray,_=low_memory_gray_preview(bmp,max_side=max_side)
        h,w=gray.shape
        source,predicted=_render_all(
            gray,components,objects,marks,guides,gaps,
            w/bmp.width,h/bmp.height
        )
        montage=_captioned_montage(
            source,predicted,categories=summary["categories"],
            source_name=source_name
        )
    else:
        source=predicted=montage=None

    private_output.mkdir(parents=True,mode=0o700)
    private_output.chmod(0o700)
    for name,rows in (
        ("all_source_fragment_types.ndjson",objects),
        ("human_operator_mark_locations.ndjson",marks),
        ("operator_2d_curve_guides.ndjson",guides),
        ("all_gap_proposals_unknown.ndjson",gaps),
    ):
        _write_private(private_output/name,_ndjson(rows))
    if source is not None:
        for name,pixels in (
            ("S95_original_small.jpg",source),
            ("S95_all_extracted_types_small.jpg",predicted),
            ("S95_all_classes_comparison.jpg",montage),
        ):
            path=private_output/name
            ok,buffer=cv2.imencode(".jpg",pixels,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok:
                raise OSError("Cannot export real reduced BMP pixels")
            _write_private(path,buffer.tobytes())
        report["all_classes_comparison_file"]="S95_all_classes_comparison.jpg"
        report["all_classes_comparison_sha256"]=sha256_file(
            private_output/"S95_all_classes_comparison.jpg"
        )
    else:
        report["all_classes_comparison_file"]=None
    body=(
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>S95: all observed structures</title>"
        "<style>body{font:15px sans-serif;background:#101319;color:#e9edf2;margin:2rem}"
        "img{max-width:100%;height:auto}</style>"
        "<h1>S95: все наблюдаемые S86 фрагменты — машинная классификация</h1>"
        "<p>Ничего из S86 не удалено. Видимые линии и реперы могут пересекаться. "
        "Пучок и завиток не означают подтверждённое столкновение или тип частицы.</p>"
        "<p>Категории: "+html.escape(str(summary["categories"]))+"</p>"
        +("<img src='S95_all_classes_comparison.jpg' alt='real machine overlay'/>"
          if source is not None else
          "<p>Нет исходного BMP: экспортированы координаты и классификация без фото.</p>")
        +"</html>"
    )
    _write_private(private_output/"index.html",body.encode("utf8"))
    # Defend against edited input while computing image and metadata.
    if sha256_file(s92_directory/"report.json")!=report["source_s92_report_sha256"]:
        raise ValueError("S92 report changed mid-run")
    if labels_path is not None and sha256_file(labels_path)!=labels_sha:
        raise ValueError("Sparse source labels changed mid-run")
    if operator_curves_path is not None and sha256_file(operator_curves_path)!=operator_sha:
        raise ValueError("Operator curve source changed mid-run")
    if bmp_path is not None and sha256_file(bmp_path)!=report["source_bmp_sha256"]:
        raise ValueError("Original source photo changed mid-run")
    _write_private(private_output/"report.json",_json(report))
    return report
