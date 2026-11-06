"""S99 POST-HOC local source-image gallery for morphology disagreements.

This file MUST NOT be imported/executed by the original-only S97 annotation
webpage. The S97 independent operator JSON must already be exported, source-
verified and frozen before any S95 machine colors can be shown.

Original source BMP tile pixels appear losslessly on the left, while right
uses exact pixel coordinates with S95 MACHINE class hypotheses and separately
visible independent expert curves. This gallery does NOT constitute physics
ground truth or any full-frame precision estimate.
"""
from __future__ import annotations

from pathlib import Path
import html
import json

import cv2
import numpy as np

from .bitmap_tiles import sha256_file
from .s92_curve_review import _write_private,_json
from .s99_morphology_audit import VALID_MACHINE_TYPES

COLORS_BGR={
    "TRACK_LIKE_CANDIDATE":(255,165,30),
    "COILING_TRACE_CANDIDATE":(40,190,85),
    "CALIBRATION_CROSS_CONTEXT":(60,190,255),
    "ARTIFACT_LIKE_CANDIDATE":(95,110,230),
    "BUNDLE_OR_INTERSECTION_REVIEW":(45,50,225),
    "UNKNOWN":(165,165,165),
}
HUMAN_MARK_BGR=(205,25,230)
HUMAN_UNCERTAIN_BGR=(160,160,160)
HUMAN_LABELS={
    "TRACK_LIKE":"H:TRACK",
    "COILING_TRACK_LIKE":"H:COIL",
    "FIDUCIAL_CROSS":"H:CROSS",
    "ARTIFACT":"H:ARTIFACT",
    "CROSSING_OR_BUNDLE":"H:BUNDLE",
    "UNKNOWN":"H:UNKNOWN"
}


def _source_points_near_rect(points,rect):
    xy=np.asarray(points,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2:
        raise ValueError("S99 original component lacks valid 2D source geometry")
    if not np.isfinite(xy).all():
        raise ValueError("S99 nonfinite source geometry")
    x0,y0,x1,y1=rect
    return not (
        np.max(xy[:,0])<x0 or np.min(xy[:,0])>=x1
        or np.max(xy[:,1])<y0 or np.min(xy[:,1])>=y1
    )


def _overlay(gray,rect,components,typed,human):
    canvas=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    x0,y0,x1,y1=rect
    origin=np.array([x0,y0],dtype=float)
    for comp in components:
        cid=comp["component_id"]
        if cid not in typed:
            raise ValueError("S99 source component not in matching S95 review type set")
        pts=np.asarray(comp["points_xy"],dtype=float)
        if not _source_points_near_rect(pts,rect):
            continue
        local=np.rint(pts-origin).astype(np.int32)
        cv2.polylines(
            canvas,[local.reshape(-1,1,2)],
            isClosed=False,color=COLORS_BGR[typed[cid]],
            thickness=1,lineType=cv2.LINE_8
        )
    for row in human:
        pts=np.asarray(row["points_xy_source_px"],dtype=float)
        if not _source_points_near_rect(pts,rect):
            continue
        local=np.rint(pts-origin).astype(np.int32)
        # Human drawn strokes are a separate visual layer, never machine GT.
        known=row["certainty"]=="CLEAR" and row["kind"]!="UNKNOWN"
        color=HUMAN_MARK_BGR if known else HUMAN_UNCERTAIN_BGR
        cv2.polylines(
            canvas,[local.reshape(-1,1,2)],isClosed=False,
            color=color,thickness=2,lineType=cv2.LINE_8
        )
        caption=HUMAN_LABELS[row["kind"]] if known else "H:UNCERTAIN"
        loc=(int(np.clip(local[0,0]+3,1,canvas.shape[1]-10)),
             int(np.clip(local[0,1]-4,12,canvas.shape[0]-2)))
        cv2.putText(canvas,caption,loc,cv2.FONT_HERSHEY_SIMPLEX,
                    .34,(25,25,25),2,lineType=cv2.LINE_AA)
        cv2.putText(canvas,caption,loc,cv2.FONT_HERSHEY_SIMPLEX,
                    .34,color,1,lineType=cv2.LINE_AA)
    return canvas


def export_s99_posthoc_review(
    *,pack_dir:Path,pack:dict,human:list[dict],
    candidates:list[tuple[str,list[dict],dict[str,str]]],
    out_dir:Path,max_preview_tiles:int=9
):
    pack_dir=Path(pack_dir).resolve()
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (out_dir.exists() or out_dir==repo or repo in out_dir.parents
        or out_dir==pack_dir or pack_dir in out_dir.parents):
        raise ValueError("S99 posthoc gallery must be a NEW PRIVATE output")
    if type(max_preview_tiles) is not int or not 0<=max_preview_tiles<=64:
        raise ValueError("S99 posthoc gallery image limit invalid")
    names=[name for name,_,_ in candidates]
    if not names or len(set(names))!=len(names) or len(names)>8:
        raise ValueError("S99 posthoc gallery candidate inventory invalid")
    for name,components,typed in candidates:
        if set(typed)!={c["component_id"] for c in components}:
            raise ValueError("S99 posthoc gallery source inventory does not match S95 classes")
        if not set(typed.values()).issubset(VALID_MACHINE_TYPES):
            raise ValueError("S99 invalid S95 morphology palette type")

    tiles=pack["source_image_tiles"][:max_preview_tiles]
    out_dir.mkdir(parents=True,mode=0o700)
    out_dir.chmod(0o700)
    (out_dir/"gallery").mkdir(mode=0o700)
    cases=[]
    for ti,tile in enumerate(tiles):
        original_file=(pack_dir/tile["source_original_png"]).resolve()
        if (pack_dir not in original_file.parents
            or not original_file.is_file()
            or sha256_file(original_file)!=tile["tile_preview_sha256"]):
            raise ValueError("S99 original S97 source PNG changed after expert annotation")
        xyxy=tile["xyxy_source"]
        original=cv2.imread(str(original_file),cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape!=(
            xyxy[3]-xyxy[1],xyxy[2]-xyxy[0]
        ):
            raise ValueError("S99 S97 source pixel region shape mismatch")
        rel_orig=f"gallery/roi{ti:03d}_original.png"
        _write_private(out_dir/rel_orig,original_file.read_bytes())
        overlays={}
        for ci,(name,components,typed) in enumerate(candidates):
            drawn=_overlay(original,xyxy,components,typed,human)
            pair=np.concatenate([
                cv2.cvtColor(original,cv2.COLOR_GRAY2BGR),drawn
            ],axis=1)
            image_name=f"gallery/roi{ti:03d}_candidate{ci:02d}_comparison.png"
            ok,encoded=cv2.imencode(".png",pair,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:
                raise OSError("S99 native source morphology posthoc PNG encode failed")
            _write_private(out_dir/image_name,encoded.tobytes())
            overlays[name]={
                "source_original_left_machine_and_human_right":image_name,
                "image_sha256":sha256_file(out_dir/image_name),
                "left_source_pixels_pixel_exact":True,
                "human_markup_shown_only_after_sealed_reference":True,
                "colored_S95_lines_are_machine_hypotheses_not_physical_GT":True,
            }
        cases.append({
            "original_source_tile_id":tile["tile_id"],
            "source_xyxy":xyxy,
            "original_png":rel_orig,
            "original_png_sha256":sha256_file(out_dir/rel_orig),
            "overlays":overlays,
        })
    headings=(
        "<h1>S99: сравнение классов S95 с независимой разметкой по оригинальным BMP</h1>"
        "<p>Этот HTML создаётся только ПОСЛЕ самостоятельной разметки S97. "
        "Операторский интерфейс S97 не показывает машинные результаты.</p>"
        "<p>На сравнении <b>слева неизменённый оригинал</b>, справа он же "
        "с наблюдёнными центрлиниями S86, окрашенными по категориям S95, "
        "и человеческими линиями поверх. Розовый — экспертная 2D линия; "
        "голубой — машинный трекоподобный след; зелёный — машинный завиток; "
        "жёлто-оранжевый — реперный крест; красный — артефакт/пересечение "
        "(точная категория указана в JSON). Серый — неизвестно. "
        "Подписи H:TRACK/H:COIL/H:ARTIFACT/H:CROSS/H:BUNDLE обозначают "
        "ТОЛЬКО независимый человеческий визуальный класс; "
        "H:UNCERTAIN не входит в количественную оценку.</p>"
        "<p><b>Не является подтверждением типов частиц, столкновений или "
        "полной точности всех пикселей изображения.</b> Показано только "
        "фиксированное число первых заранее выбранных ROI.</p>"
    )
    sections=[]
    for item in cases:
        sections.append(
            "<section><h2>"+html.escape(item["original_source_tile_id"])+
            " — ROI "+html.escape(str(item["source_xyxy"]))+"</h2>"
        )
        for name,_,_ in candidates:
            row=item["overlays"][name]
            sections.append(
                "<h3>"+html.escape(name)+"</h3>"
                "<img alt='Original source on left and machine class overlay on right' "
                "src='"+html.escape(row["source_original_left_machine_and_human_right"])+
                "'/>"
            )
        sections.append("</section>")
    page=(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>S99 — независимое сравнение морфологических классов</title>"
        "<style>body{font:14px system-ui;background:#f8f8f8;color:#242424;margin:20px}"
        "p{max-width:1050px;line-height:1.4}img{max-width:100%;height:auto}"
        "section{padding:15px 0;border-top:1px solid #aaa}</style>"
        +headings+"".join(sections)+"</html>"
    )
    _write_private(out_dir/"index.html",page.encode("utf-8"))
    report={
        "schema":"S99_POSTHOC_VISUAL_CLASS_AGREEMENT_GALLERY_V1",
        "status":"PRIVATE_POSTHOC_AFTER_S97_BLIND_SOURCE_REVIEW",
        "original_bmp_sha256":pack["source_bmp_sha256"],
        "source_tiles_previewed":len(cases),
        "total_source_tiles":len(pack["source_image_tiles"]),
        "source_order_sample_not_selected_by_best_model_score":True,
        "independent_annotation_editor_never_displays_S95":True,
        "original_source_pixel_per_image_sha256":True,
        "partial_manual_visible_2D_lines_not_particle_GT":True,
        "cases":cases,
    }
    _write_private(out_dir/"posthoc_gallery_manifest.json",_json(report))
    return report
