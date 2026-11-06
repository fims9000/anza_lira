"""S97 POST-HOC error visualizations, generated only AFTER independent labels.

The original-only S97 expert editor NEVER shows machine predictions. After
the human reference JSON is frozen, this module exports LOCAL comparison
maps: cyan = observed S86 skeleton, green = human positive line covered
within a fixed tolerance, magenta = human positive line that S86 missed.
Orange/gray (if shown) are excluded artifacts/uncertain references.

The map is diagnostic, not full negative-GT segmentation. Missing area is
a *conditional human-positive stroke* with no nearby observed centerline.
No inferred S92/S96 bridges or physics labels count as detected pixels.
"""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import html
import json
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .bitmap_tiles import sha256_file
from .s92_curve_review import _write_private,_json
from .s97_manual_reference import (
    POSITIVE_KINDS,_point_array,_resample_polyline,_source_observed_points
)

SCHEMA="S97_POSTHOC_HUMAN_POSITIVE_MISSED_SPAN_MAPS_V1"


def _is_within(tile,point):
    x0,y0,x1,y1=tile["xyxy_source"]
    return x0<=point[0]<x1 and y0<=point[1]<y1


def _candidate_tiles(pack,max_tiles):
    if not isinstance(max_tiles,int) or not 1<=max_tiles<=128:
        raise ValueError("Invalid S97 diagnostic preview tile cap")
    # NOT chosen by model performance: stable source tile ordering.
    return pack["source_image_tiles"][:max_tiles]


def _safe_model_tag(name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}",name):
        raise ValueError("Unsafe model label for private map export")
    return name


def export_posthoc_error_maps(
    *,
    pack_dir:Path,
    pack:dict,
    reference_strokes:list[dict],
    candidate_runs:list[tuple[str,list[dict]]],
    out_dir:Path,
    radius_px:float=2.,
    max_tiles:int=12,
    source_reference_sha256:str|None=None,
):
    pack_dir=Path(pack_dir).resolve()
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (out_dir.exists() or out_dir==repo or repo in out_dir.parents
        or out_dir==pack_dir or pack_dir in out_dir.parents):
        raise ValueError("Posthoc visual review needs new PRIVATE directory")
    if not isinstance(candidate_runs,list) or not 1<=len(candidate_runs)<=8:
        raise ValueError("S97 visual export supports 1..8 model runs")
    if len({name for name,_ in candidate_runs})!=len(candidate_runs):
        raise ValueError("Duplicate model in S97 visual error map")
    if not .5<=radius_px<=10.:
        raise ValueError("Invalid observed-reference tolerance")
    tiles=_candidate_tiles(pack,max_tiles)
    if not tiles:raise ValueError("Source annotation image has no original tiles")
    names=[_safe_model_tag(name) for name,_ in candidate_runs]
    observed=[]
    trees=[]
    for name,components in candidate_runs:
        coordinates=_source_observed_points(components)
        observed.append(coordinates)
        trees.append(cKDTree(coordinates) if len(coordinates) else None)

    # No evaluating model-specific winners when choosing reference subset.
    resampled=[]
    for row in reference_strokes:
        points,_=_resample_polyline(_point_array(row["points_xy_source_px"]))
        if len(points)>200000:
            raise ValueError("Unbounded positive reference trace preview")
        resampled.append((row,points))
    out_dir.mkdir(parents=True,mode=0o700)
    out_dir.chmod(0o700)
    (out_dir/"images").mkdir(mode=0o700)
    cards=[]
    for index,tile in enumerate(tiles):
        path=pack_dir/tile["source_original_png"]
        if (not path.is_file()
            or sha256_file(path)!=tile["tile_preview_sha256"]):
            raise ValueError("Original photo tile changed after independent annotation")
        original=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        x0,y0,x1,y1=tile["xyxy_source"]
        if original is None or original.shape!=(y1-y0,x1-x0):
            raise ValueError("Source S97 original ROI no longer pixel-exact")
        rootname=f"t{index:04d}"
        orig_rel=f"images/{rootname}_ORIGINAL.png"
        _write_private(out_dir/orig_rel,path.read_bytes())
        card={
            "source_tile_id":tile["tile_id"],
            "xyxy_original_source":tile["xyxy_source"],
            "unmodified_original_png":orig_rel,
            "model_maps":{},
        }
        for name,pts,tree in zip(names,observed,trees):
            photo=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
            # Draw exact real S86 source centerline pixels in muted cyan.
            if len(pts):
                within=(pts[:,0]>=x0)&(pts[:,0]<x1)&(pts[:,1]>=y0)&(pts[:,1]<y1)
                local=np.rint(pts[within]-np.array([x0,y0])).astype(int)
                if len(local):
                    xx=np.clip(local[:,0],0,photo.shape[1]-1)
                    yy=np.clip(local[:,1],0,photo.shape[0]-1)
                    photo[yy,xx]=(240,180,65)
            covered=missed=0
            for stroke,samples in resampled:
                relevant=(samples[:,0]>=x0)&(samples[:,0]<x1)&(
                    samples[:,1]>=y0)&(samples[:,1]<y1
                )
                q=samples[relevant]
                if not len(q):continue
                source_xy=np.rint(q-np.array([x0,y0])).astype(int)
                if (stroke["kind"] in POSITIVE_KINDS
                    and stroke["certainty"]=="CLEAR"):
                    if tree is None:
                        hits=np.zeros(len(q),bool)
                    else:
                        hits=tree.query(q,k=1,workers=1)[0]<=radius_px+1e-9
                    for xy,yes in zip(source_xy,hits):
                        cx=int(np.clip(xy[0],0,photo.shape[1]-1))
                        cy=int(np.clip(xy[1],0,photo.shape[0]-1))
                        # OpenCV BGR: green observed, magenta missing.
                        cv2.circle(photo,(cx,cy),1,
                                   (60,220,80) if yes else (215,55,245),
                                   -1,lineType=cv2.LINE_8)
                    covered+=int(hits.sum())
                    missed+=int((~hits).sum())
                else:
                    for xy in source_xy[::max(1,len(source_xy)//200)]:
                        cx=int(np.clip(xy[0],0,photo.shape[1]-1))
                        cy=int(np.clip(xy[1],0,photo.shape[0]-1))
                        photo[cy,cx]=(45,150,255) if stroke["kind"]=="ARTIFACT" else (185,185,185)
            relative=f"images/{rootname}_{name}_REVIEW.png"
            ok,blob=cv2.imencode(".png",photo,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:
                raise OSError("Cannot write S97 observed-vs-human error map")
            _write_private(out_dir/relative,blob.tobytes())
            card["model_maps"][name]={
                "image":relative,
                "human_positive_samples_covered_in_this_tile":covered,
                "human_positive_samples_missed_in_this_tile":missed,
                "preview_only_not_full_metric_denominator":True,
            }
        cards.append(card)
    report={
        "schema":SCHEMA,
        "status":"POSTHOC_VISUAL_ERROR_MAPS_AFTER_INDEPENDENT_ANNOTATION",
        "original_source_bmp_sha256":pack["source_bmp_sha256"],
        "source_reference_pack_sha256":source_reference_sha256,
        "source_tiles_displayed":len(cards),
        "source_tiles_total":len(pack["source_image_tiles"]),
        "tile_selection":"FIRST_IN_SOURCE_COORDINATE_ORDER_NO_MACHINE_SCORE_FILTER",
        "models":names,
        "fixed_observed_centerline_distance_radius_px":radius_px,
        "original_tiles_unchanged":True,
        "error_map_refers_only_to_independently_annotated_positive_lines":True,
        "posthoc_overlay_never_shown_during_original_reference_annotation":True,
        "no_unreviewed_negative_pixel_GT":True,
        "model_inferred_S92_S96_bridges_not_observed":True,
        "cards":cards,
    }
    headings=["<th>Оригинал</th>"]+[
        "<th>"+html.escape(name)+" — исходная детекция</th>" for name in names
    ]
    table=[]
    for item in cards:
        imgs=["<td><img src='"+item["unmodified_original_png"]+
              "' alt='Original BMP pixels'/></td>"]
        imgs.extend("<td><img src='"+item["model_maps"][name]["image"]+
                    "' alt='Observed S86 and separate human positive/absent strokes'/></td>"
                    for name in names)
        table.append(
            "<h3>Плитка "+html.escape(item["source_tile_id"])+
            " — координаты "+html.escape(str(item["xyxy_original_source"]))+"</h3>"
            "<table><tr>"+"".join(headings)+"</tr><tr>"+
            "".join(imgs)+"</tr></table>"
        )
    page=(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>S97: независимые визуальные пропуски</title>"
        "<style>body{font:14px system-ui;background:#12171e;color:#ecf1f5;margin:20px}"
        "table{border-collapse:collapse}th,td{border:1px solid #66717c;padding:8px}"
        "img{max-width:620px;height:auto}p{max-width:1000px}</style>"
        "<h1>S97: обнаруженные и пропущенные участки независимой разметки</h1>"
        "<p>Слабая бирюза — наблюдаемые центральные пиксели детектора S86; "
        "зелёный — независимый человеческий трек найден в радиусе допуска; "
        "пурпурный — независимый человеческий трек не покрыт детектором. "
        "Оранжевый — размеченный артефакт, серый — другие/неуверенные линии, "
        "не участвующие в метрике. "
        "Непомеченные машинные пиксели не считаются ложными автоматически.</p>"
        "<p>**ДИАГНОСТИКА ПОСЛЕ СБОРА ЭТАЛОНА**, а не независимый интерфейс разметки. "
        "Выборка плиток ограничена. Истинные физические частицы и нейтронные "
        "столкновения не подтверждены. Полнота всего кадра не определена.</p>"
        +"".join(table)+"</html>"
    )
    _write_private(out_dir/"index.html",page.encode("utf8"))
    _write_private(out_dir/"posthoc_error_map_report.json",_json(report))
    return report
