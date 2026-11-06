"""S98 PRIVATE POST-HOC original-film gap/fragmentation review.

The S97 expert annotation is already complete and SHA-frozen before this
module can be called. Only now the reviewer may compare the SAME original
source pixels with S86 fragment hypotheses and omitted S92/S96 proposals.

No S97 source original-only viewer file is touched or overwritten.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from hashlib import sha256
import html

import cv2
import numpy as np

from .bitmap_tiles import sha256_file
from .s92_curve_review import _write_private,_json


def _cell(value):
    if value is None:
        value="Не оценивалось"
    if isinstance(value,float):
        value=f"{value:.4f}"
    return html.escape(str(value),quote=True)


def _tile_for_original_anchor(pack_dir,pack,anchor):
    xy=np.asarray(anchor,float)
    if xy.shape!=(2,) or not np.isfinite(xy).all():
        raise ValueError("S98 invalid independent source anchor")
    for tile in pack["source_image_tiles"]:
        x0,y0,x1,y1=tile["xyxy_source"]
        if not (x0<=xy[0]<x1 and y0<=xy[1]<y1):
            continue
        file=(pack_dir/tile["source_original_png"]).resolve()
        if (pack_dir not in file.parents or not file.is_file()
            or sha256_file(file)!=tile["tile_preview_sha256"]):
            raise ValueError("S98 original S97 photo tile missing/modified")
        gray=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(y1-y0,x1-x0):
            raise ValueError("S98 original S97 source pixels have changed")
        return gray,tile
    return None,None


def _crop_anchor(gray,tile,anchor,half=112):
    x0,y0,x1,y1=tile["xyxy_source"]
    x=float(anchor[0])-x0
    y=float(anchor[1])-y0
    cx=int(round(x))
    cy=int(round(y))
    xx0=max(0,cx-half)
    yy0=max(0,cy-half)
    xx1=min(gray.shape[1],cx+half)
    yy1=min(gray.shape[0],cy+half)
    if xx1<=xx0 or yy1<=yy0:
        raise ValueError("S98 crop lies outside original S97 source tile")
    return gray[yy0:yy1,xx0:xx1],(x0+xx0,y0+yy0,x0+xx1,y0+yy1)


def _draw_observed_line(out,pts,top_left,color):
    arr=np.asarray(pts,float)
    if (arr.ndim!=2 or arr.shape[1]!=2 or not np.isfinite(arr).all()
        or len(arr)<2):
        raise ValueError("S98 observed line path invalid")
    p=np.rint(arr-np.asarray(top_left,float)).astype(np.int32)
    cv2.polylines(out,[p.reshape(-1,1,2)],False,color,2,lineType=cv2.LINE_AA)


def _build_missing_case_image(*,pack_dir,pack,candidate,reference_strokes,
                              observed_components,boundary):
    anchor=boundary["example_reference_boundary_anchor_xy_source"]
    original,tile=_tile_for_original_anchor(pack_dir,pack,anchor)
    if original is None:
        return None
    crop,xyxy=_crop_anchor(original,tile,anchor)
    overlay=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
    top_left=(xyxy[0],xyxy[1])
    source={x["component_id"]:x for x in observed_components}
    a,b=boundary["component_a"],boundary["component_b"]
    if a not in source or b not in source:
        raise ValueError("S98 boundary references non-source S86 component")
    for ident,color in ((a,(255,175,40)),(b,(80,210,70))):
        _draw_observed_line(overlay,source[ident]["points_xy"],top_left,color)
    strokes={row["stroke_id"]:row for row in reference_strokes}
    for sid in boundary["supporting_human_stroke_ids"]:
        if sid not in strokes:
            raise ValueError("S98 missing frozen human positive source stroke")
        _draw_observed_line(overlay,strokes[sid]["points_xy_source_px"],
                            top_left,(210,30,240))
    q=np.rint(np.asarray(anchor,float)-np.asarray(top_left,float)).astype(int)
    cv2.circle(overlay,(int(q[0]),int(q[1])),7,(20,30,220),2,lineType=cv2.LINE_AA)
    joined=np.concatenate((cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR),overlay),axis=1)
    return joined,{
        "tile_id":tile["tile_id"],
        "source_tile_sha256":tile["tile_preview_sha256"],
        "crop_xyxy_original_source":list(xyxy),
        "original_left_unchanged":True,
        "overlay_right_not_observed_physics_or_ground_truth":True,
    }


def export_s98_private_visual_report(
    *,
    pack_dir:Path,
    pack:dict,
    human:list[dict],
    observed_by_name:dict[str,list[dict]],
    output_dir:Path,
    report:dict,
    max_visual_case_count:int=16,
):
    pack_dir=Path(pack_dir).resolve()
    output_dir=Path(output_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (output_dir.exists() or output_dir==repo or repo in output_dir.parents
        or output_dir==pack_dir or pack_dir in output_dir.parents):
        raise ValueError("S98 visual report needs new PRIVATE output")
    if type(max_visual_case_count) is not int or not 0<=max_visual_case_count<=64:
        raise ValueError("S98 preview limit must be 0..64 per candidate")
    if len(observed_by_name)!=len(report["candidates"]):
        raise ValueError("S98 cannot silently omit a source candidate visual view")
    output_dir.mkdir(parents=True,mode=0o700)
    output_dir.chmod(0o700)
    (output_dir/"cases").mkdir(mode=0o700)
    previews={}
    for i,trial in enumerate(report["candidates"]):
        name=trial["model_name"]
        if name not in observed_by_name:
            raise ValueError("No matched S86 source pixels for S98 model "+name)
        recovery=trial["reference_first_model_link_recovery"]
        # Do not manufacture a "missed machine link" if no S92/S96 proposal
        # pack was supplied. Such links are NOT_EVALUATED, not model failures.
        if recovery is None:
            previews[name]={
                "posthoc_cases":[],"status":"NO_S92_S96_PACK_NOT_EVALUATED",
                "candidate_missing_link_claims_generated":False,
            }
            continue
        missing=recovery["reference_boundaries_missed_by_all_selected_models"]
        by_id={x["stroke_id"]:x for x in human}
        visible=[]
        for j,boundary in enumerate(missing[:max_visual_case_count]):
            image_payload=_build_missing_case_image(
                pack_dir=pack_dir,pack=pack,candidate=name,
                reference_strokes=human,observed_components=observed_by_name[name],
                boundary=boundary
            )
            if image_payload is None:
                visible.append({
                    "pair_id":boundary["pair_id"],
                    "component_a":boundary["component_a"],
                    "component_b":boundary["component_b"],
                    "status":"NO_ORIGINAL_ROI_TILE_FOR_BOUNDARY_ANCHOR"
                })
                continue
            png,receipt=image_payload
            location=f"cases/model{i:02d}_case{j:03d}_{boundary['pair_id']}.png"
            ok,buf=cv2.imencode(".png",png,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:
                raise RuntimeError("S98 posthoc source crop cannot be encoded")
            _write_private(output_dir/location,buf.tobytes())
            visible.append({
                "pair_id":boundary["pair_id"],
                "component_a":boundary["component_a"],
                "component_b":boundary["component_b"],
                "human_stroke_ids":boundary["supporting_human_stroke_ids"],
                "visible_human_instance_ids":boundary["supporting_human_visual_instance_ids"],
                "photo":location,
                "photo_sha256":sha256_file(output_dir/location),
                "status":"POSTHOC_CONDITIONAL_VISUAL_BOUNDARY_NOT_SELECTED",
                **receipt,
            })
        previews[name]={
            "status":"POSTHOC_FROZEN_HUMAN_VISUAL_TRACE_BOUNDARIES_ONLY",
            "total_reference_boundaries_unselected":len(missing),
            "posthoc_cases":visible,
            "source_order_sampling_first_k_no_machine_score_ranking":True,
            "only_first_k_previewed_not_full_metric_population":True,
        }
    report["posthoc_visual_case_preview"]={
        "per_model":previews,
        "not_part_of_original_only_human_editor":True,
        "all_images_private_source_derived":True,
        "source_photo_SHA":pack["source_bmp_sha256"],
        "source_ROIs_unmodified":True,
        "preview_is_not_independent_real_H50Z_quality_evidence":True,
    }
    rows=[]
    for trial in report["candidates"]:
        name=trial["model_name"]
        cond=trial["reference_first_source_piece_boundaries"]
        recovery=trial["reference_first_model_link_recovery"]
        support=trial["human_source_trace_per_stroke_results"]
        above=sum(
            x["unique_source_components_near_this_human_trace"]>1 for x in support
        )
        selected=(
            recovery["union_selected_conditional_coverage"]
            if recovery is not None else None
        )
        rows.append(
            "<tr><td>"+_cell(name)+"</td><td>"+_cell(
                trial["observed_source_centerline_positive_coverage"]
            )+"</td><td>"+_cell(
                cond["assessable_unique_detected_piece_adjacencies"]
            )+"</td><td>"+_cell(above)+"</td><td>"+_cell(
                selected
            )+"</td><td>"+_cell(
                trial["source_S86_skipped_tiles"]
            )+"</td></tr>"
        )
    parts=[
        "<!doctype html><html lang='ru'><meta charset='utf-8'>",
        "<title>S98 — независимая целостность наблюдаемых 2D линий</title>",
        "<style>body{font:14px system-ui,sans-serif;background:#fafafa;color:#24272b;margin:24px}"
        "p{max-width:1100px;line-height:1.48}table{border-collapse:collapse}"
        "th,td{border:1px solid #a2a8af;padding:8px;text-align:left}"
        "img{max-width:100%;height:auto;border:1px solid #979ca4}"
        "section{margin:28px 0;border-top:1px solid #bbb;padding-top:12px}"
        ".case{max-width:1100px;margin:14px 0}small{color:#616976}"
        "</style>",
        "<h1>S98 — разрывы и продолжения линий по независимой разметке</h1>",
        "<p>Эксперт заранее размечает видимые 2D линии по оригинальной "
        "фотографии S97 БЕЗ машинного оверлея. После сохранения исходной разметки "
        "S98 считает только отдельные наблюдаемые компоненты S86 и возможные "
        "границы между ними вдоль одной вручную нарисованной линии.</p>",
        "<p><b>Это не оценка физических частиц.</b> Если модель пропустила "
        "трассу целиком, у неё будет ноль обнаруженных фрагментов и ноль "
        "оцениваемых границ — это НЕ означает идеальную целостность. "
        "Сравнивайте число фрагментов ТОЛЬКО вместе с покрытием независимых "
        "человеческих позитивных линий и числом пропущенных участков.</p>",
        "<table><tr><th>Модель</th><th>Покрытие уверенных линий S97</th>"
        "<th>Оцениваемые межфрагментные границы</th>"
        "<th>Линий с несколькими наблюдаемыми компонентами</th>"
        "<th>Доля границ, для которых выбран S92/S96</th>"
        "<th>Пропущенных тайлов S86</th></tr>",
        *rows,"</table>",
        "<p><small>Показатель доли для S92/S96 пуст, если такие пакеты не "
        "переданы, либо если нет оцениваемых независимых границ. "
        "Источник — один снимок: независимый event-level доверительный "
        "интервал не вычисляется. Реперы и сомнительные линии исключены.</small></p>",
        "<p>На изображениях ниже: слева неизменённые исходные пиксели, "
        "справа те же пиксели с наблюдаемым фрагментом A (голубой), "
        "фрагментом B (зелёный), человеческой линией (пурпурный) и "
        "центром границы (красный круг). Машинная склейка НЕ доказана физически. "
        "Это диагностический просмотр ПОСЛЕ независимой разметки.</p>",
    ]
    for trial in report["candidates"]:
        name=trial["model_name"]
        parts.append("<section><h2>"+_cell(name)+"</h2>")
        recovery=trial["reference_first_model_link_recovery"]
        if recovery is None:
            parts.append("<p>S92/S96 не предоставлены: качество выбора "
                         "соединений не оценивалось.</p></section>")
            continue
        parts.append("<p>Число оцениваемых границ: "+_cell(
            recovery["reference_first_unique_source_piece_boundaries"]
        )+"; выбранные соединения покрывают долю "+_cell(
            recovery["union_selected_conditional_coverage"]
        )+". Это только наблюдаемые фрагменты одной визуально "
        "экспертно размеченной 2D линии, НЕ полная физическая полнота.</p>")
        examples=previews[name]["posthoc_cases"]
        if not examples:
            parts.append("<p>Нет пропущенных выбранными алгоритмами "
                         "границ, доступных для этого послеразметочного превью.</p>")
        for example in examples:
            parts.append("<div class='case'><h3>"+
                         _cell(example["component_a"])+" ↔ "+
                         _cell(example["component_b"])+"</h3>")
            if "photo" in example:
                parts.append("<img src='"+_cell(example["photo"])+"' "
                             "alt='Source BMP left, observed machine parts and human trace on right' />")
                parts.append("<p><small>Оригинальные координаты: "+
                             _cell(example["crop_xyxy_original_source"])+
                             "</small></p>")
            else:
                parts.append("<p>Исходный ROI не включает центр этой "
                             "геометрической границы, метаданные сохранены.</p>")
            parts.append("</div>")
        parts.append("</section>")
    parts.append("</html>")
    _write_private(output_dir/"index.html","\n".join(parts).encode("utf8"))
    _write_private(output_dir/"report.json",_json(report))
    return report
