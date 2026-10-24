"""Self-contained human-review workflow for P0 fragment joins.

All labels are local human annotations. No AI prediction is treated as truth.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = "bubble_chamber_pair_review_v1"
LABELS = frozenset({"same_track", "different_track", "uncertain"})


def pair_key(proposal: dict[str, Any]) -> str:
    """Order-independent unique endpoint-pair identifier within one frame."""
    endpoints = sorted((
        (int(proposal["segment_A"]), str(proposal["endpoint_A"])),
        (int(proposal["segment_B"]), str(proposal["endpoint_B"])),
    ))
    if any(side not in ("A", "B") for _, side in endpoints):
        raise ValueError("invalid endpoint label")
    if endpoints[0] == endpoints[1]:
        raise ValueError("identical endpoints")
    return f"{endpoints[0][0]}:{endpoints[0][1]}|{endpoints[1][0]}:{endpoints[1][1]}"


def validate_labels(result: dict[str, Any], label_bundle: dict[str, Any]) -> dict[str, int]:
    """Strictly validate a reviewed label bundle against the exact P0 result.

    This does not evaluate predictions: it only guarantees referential integrity.
    """
    if label_bundle.get("schema") != SCHEMA:
        raise ValueError("unsupported label schema")
    if label_bundle.get("source_sha256_grayscale_pixels") != result.get("source_sha256_grayscale_pixels"):
        raise ValueError("image SHA mismatch; do not transfer labels to another image")
    if label_bundle.get("pipeline") != result.get("pipeline"):
        raise ValueError("pilot pipeline mismatch")
    if label_bundle.get("result_sha256") != result_digest(result):
        raise ValueError("result content mismatch: segment identities may have changed")
    expected = {pair_key(p) for p in result["join_proposals"]}
    if len(expected) != len(result["join_proposals"]):
        raise ValueError("duplicate candidate IDs in the source result")
    labels = label_bundle.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    counts = {item: 0 for item in sorted(LABELS)}
    used: set[str] = set()
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("label entries must be objects")
        key = item.get("pair_key")
        value = item.get("label")
        if key not in expected:
            raise ValueError(f"unknown candidate {key!r}")
        if key in used:
            raise ValueError(f"duplicate candidate {key!r}")
        if value not in LABELS:
            raise ValueError(f"invalid label {value!r}")
        if not isinstance(item.get("comment", ""), str):
            raise ValueError("comment must be text")
        used.add(key)
        counts[value] += 1
    counts["unreviewed"] = len(expected) - len(used)
    return counts


def result_digest(result: dict[str, Any]) -> str:
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_review_html(result: dict[str, Any], image: np.ndarray, *, title: str = "Bubble chamber: track review") -> str:
    """Generate offline annotator HTML with frame-bound provenance and export.

    Source frame is loaded locally. Its SHA must match P0 grayscale pixel bytes.
    Frame and measurements are embedded in the HTML; no network calls required.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("source must be grayscale uint8")
    actual = hashlib.sha256(image.tobytes()).hexdigest()
    if actual != result.get("source_sha256_grayscale_pixels"):
        raise ValueError("result and original image SHA differ")
    if result.get("frame_shape_yx") != list(image.shape):
        raise ValueError("source dimensions do not match result")
    payload = {
        "schema": SCHEMA,
        "source_sha256_grayscale_pixels": actual,
        "pipeline": result["pipeline"],
        "result_sha256": result_digest(result),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "segments": [{"id": int(s["segment_id"]), "points": s["measurement_xy_px"], "fit_rmse": s["fit"]["residual_rms_px"]} for s in result["segments"]],
        "candidates": [{**p, "pair_key": pair_key(p)} for p in result["join_proposals"]],
    }
    if len({p["pair_key"] for p in payload["candidates"]}) != len(payload["candidates"]):
        raise ValueError("duplicate candidate IDs")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 91])
    if not ok:
        raise ValueError("image encode failed")
    background = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    # Escape title even when supplied by external documents.
    from html import escape
    return _HTML.replace("__TITLE__", escape(title)).replace("__FRAME_DATA_URI__", background).replace("__JSON_DATA__", encoded)


_HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme:dark; font:15px system-ui,Arial,sans-serif; background:#111720;color:#eaf0f7 }
body{margin:0;padding:20px;max-width:1400px;margin:auto}h1{font-size:22px}main{display:grid;grid-template-columns:minmax(250px,2fr) minmax(300px,1fr);gap:22px}
figure{margin:0}.imagebox{overflow:auto;max-height:85vh;border:1px solid #506077;background:#050609;border-radius:8px}
canvas{display:block;width:100%;height:auto;cursor:crosshair}aside{position:sticky;top:16px;align-self:start;max-height:95vh;overflow:auto}
fieldset{border:1px solid #536274;border-radius:8px;margin-bottom:12px;padding:15px}legend{padding:0 8px}
button,.button{cursor:pointer;color:#edf5ff;border:1px solid #55708c;border-radius:6px;background:#23374e;padding:10px 12px;margin:4px 3px}
button:hover{background:#315476}button:focus-visible,input:focus-visible{outline:2px solid #fff}
button.choice{font-weight:650}button.choice.selected{background:#246b58;border-color:#99e8c4}textarea,input[type=text]{background:#182231;color:#fff;border:1px solid #506077;padding:8px;border-radius:5px;width:90%}
.small{color:#a8b9cb;font-size:13px}#status{font-weight:bold}#detail{white-space:pre-wrap;font-variant-numeric:tabular-nums}
@media(max-width:850px){main{display:block}aside{position:static;max-height:none}.imagebox{max-height:70vh}}
</style></head><body>
<h1>__TITLE__</h1>
<p>Ручная экспертная разметка гипотез связности. Цветные линии — <strong>геометрические фрагменты</strong>, не подтверждённые частицы; предложение не является автоматическим соединением.</p>
<main><figure><div class="imagebox"><canvas id="overlay" aria-label="Фотография и выделенные кандидаты"></canvas></div><figcaption class="small">Показаны интерполированные 14 точек каждого фрагмента; красная перемычка — только рассматриваемая гипотеза.</figcaption></figure>
<aside><fieldset><legend>Выбор связи</legend>
<div id="status"></div><p id="detail"></p>
<button id="prev">← Назад</button><button id="next">Следующая →</button>
<p><label><input type="checkbox" id="onlyOpen"> Только ещё не размеченные</label></p>
</fieldset>
<fieldset><legend>Решение эксперта</legend>
<button class="choice" data-choice="same_track">Один трек</button>
<button class="choice" data-choice="different_track">Разные треки</button>
<button class="choice" data-choice="uncertain">Не уверен</button>
<p><label>Комментарий<br><textarea id="comment" rows="3" placeholder="Причина решения, особенности снимка"></textarea></label></p>
</fieldset>
<fieldset><legend>Разметка и перенос данных</legend>
<p><label>Обозначение эксперта<br><input id="annotator" type="text" placeholder="Физик / оператор" value=""></label></p>
<p><button id="save">Экспортировать JSON</button></p>
<p><label class="button">Импортировать JSON<input id="load" style="display:none" type="file" accept=".json,application/json"></label></p>
<p id="progress"></p><p class="small">Экспорт сохраняет только решения человека. Исходное изображение, версия сегментации и список кандидатов проверяются по хешам.</p>
</fieldset></aside></main>
<script id="source" type="application/json">__JSON_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('source').textContent);
const entries=data.candidates;
const saved=Object.create(null);
let index=0;
const el=id=>document.getElementById(id);
const canvas=el('overlay');canvas.width=data.width;canvas.height=data.height;
const ctx=canvas.getContext('2d');
const photo=new Image();photo.src='__FRAME_DATA_URI__';photo.onload=render;
const groups=new Map(data.segments.map(s=>[s.id,s]));
function endpoint(item,side){const pts=groups.get(item).points;return pts[side==='A'?0:pts.length-1]}
function line(points,color,width,opacity){if(!points.length)return;ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(const p of points.slice(1))ctx.lineTo(p[0],p[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=opacity;ctx.stroke();ctx.globalAlpha=1}
function render(){ctx.clearRect(0,0,canvas.width,canvas.height);if(photo.complete&&photo.naturalWidth)ctx.drawImage(photo,0,0,canvas.width,canvas.height);
  for(const s of data.segments)line(s.points,'#02a3ed',2,0.65);
  if(!entries.length){el('status').textContent='Нет кандидатов на соединение';el('detail').textContent='При другой настройке выделения фрагментов список может измениться.';el('progress').textContent='0 из 0';return}
  const p=entries[index];const a=groups.get(p.segment_A),b=groups.get(p.segment_B);
  if(a)line(a.points,'#f8ec40',5,1);if(b)line(b.points,'#2bf08b',5,1);
  if(a&&b){const start=endpoint(p.segment_A,p.endpoint_A),end=endpoint(p.segment_B,p.endpoint_B);line([start,end],'#ff4242',5,1);
    for(const v of [start,end]){ctx.beginPath();ctx.arc(v[0],v[1],7,0,Math.PI*2);ctx.fillStyle='#ff4242';ctx.fill()}}
  el('status').textContent='Предложение '+(index+1)+' / '+entries.length;
  el('detail').textContent=`${p.pair_key}\nРасстояние: ${p.gap_px} пикс.\nУглы: ${p.angle_A_deg}°, ${p.angle_B_deg}°\nНеоднозначно: ${p.ambiguous?'да':'нет'}\nРешение ИИ: не принято`;
  const label=saved[p.pair_key];el('comment').value=label?.comment||'';
  for(const button of document.querySelectorAll('button[data-choice]'))button.classList.toggle('selected',button.dataset.choice===label?.label);
  const reviewed=Object.keys(saved).length;el('progress').textContent=`Размечено ${reviewed} из ${entries.length}; пропущено ${entries.length-reviewed}`;
}
function move(delta){if(!entries.length)return;const open=el('onlyOpen').checked;
  if(!open){index=(index+delta+entries.length)%entries.length;render();return}
  for(let step=1;step<=entries.length;step++){const candidate=(index+delta*step+entries.length*10)%entries.length;
    if(!saved[entries[candidate].pair_key]){index=candidate;render();return}}
  render();}
function decide(label){if(!entries.length)return;const key=entries[index].pair_key;saved[key]={pair_key:key,label,comment:el('comment').value.trim()};render();move(1)}
function flushComment(){if(!entries.length)return;const key=entries[index].pair_key;if(saved[key]){saved[key].comment=el('comment').value.trim()}}
el('comment').addEventListener('change',flushComment);el('prev').onclick=()=>move(-1);el('next').onclick=()=>move(1);el('onlyOpen').onchange=()=>render();
for(const btn of document.querySelectorAll('button[data-choice]'))btn.onclick=()=>decide(btn.dataset.choice);
document.addEventListener('keydown',event=>{if(event.target.matches('textarea,input'))return;if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.key==='a')decide('same_track');if(event.key==='r')decide('different_track');if(event.key==='u')decide('uncertain')});
el('save').onclick=()=>{flushComment();const output={schema:data.schema,source_sha256_grayscale_pixels:data.source_sha256_grayscale_pixels,pipeline:data.pipeline,result_sha256:data.result_sha256,annotator:el('annotator').value.trim(),labels:Object.values(saved)};
 const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='bubble_labels_'+data.source_sha256_grayscale_pixels.slice(0,12)+'.json';a.click();URL.revokeObjectURL(url)};
el('load').addEventListener('change',async ev=>{const file=ev.target.files?.[0];if(!file)return;try{const payload=JSON.parse(await file.text());
 if(payload.schema!==data.schema||payload.source_sha256_grayscale_pixels!==data.source_sha256_grayscale_pixels||payload.result_sha256!==data.result_sha256||payload.pipeline!==data.pipeline)throw Error('Не совпадает снимок или версия результатов');
 const keys=new Set(entries.map(x=>x.pair_key));const loaded=Object.create(null);if(!Array.isArray(payload.labels))throw Error('Неверный формат меток');
 for(const item of payload.labels){if(!keys.has(item.pair_key)||loaded[item.pair_key]||!['same_track','different_track','uncertain'].includes(item.label))throw Error('Недопустимое или повторное решение');loaded[item.pair_key]={pair_key:item.pair_key,label:item.label,comment:typeof item.comment==='string'?item.comment:''}}
 for(const key of Object.keys(saved))delete saved[key];Object.assign(saved,loaded);el('annotator').value=payload.annotator||'';render();alert('Разметка загружена')
 }catch(error){alert('Импорт отклонён: '+error.message)}ev.target.value=''});
render();
</script></body></html>'''
