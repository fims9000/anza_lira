"""Human-reviewed sparse track-centerline labels from classical suggestions.

All unreviewed pixels remain UNKNOWN. A human may certify a candidate as a
particle-track centerline, an optical artifact, or uncertain. Proposals are
never ground truth, and 'reject as other track' is NOT a negative annotation.
"""
from __future__ import annotations

import base64
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .pilot import extract_skeleton_segments
from .ridges import frangi_dark_ridges

SCHEMA = 'bubble_human_track_centerline_review_v1'
ALLOWED = {'track_centerline', 'certified_artifact', 'unknown'}


def _digest(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()).hexdigest()


def propose_tile_segments(image: np.ndarray, source: dict, *, threshold: float = 0.16,
                          min_length_px: float = 24.0) -> dict[str, Any]:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError('expected grayscale uint8 tile')
    if source.get('label') not in {'start', *(f'f{i}' for i in range(1, 9))}:
        raise ValueError('source keypoint label unknown')
    if not {'parent','image_sha256','roi_original_xyxy','tile_sha256'}.issubset(source):
        raise ValueError('source provenance incomplete')
    mask, response = frangi_dark_ridges(image, response_fraction=threshold)
    skeleton = skeletonize(mask)
    parts = extract_skeleton_segments(skeleton, min_length_px=min_length_px)
    candidates = []
    x0,y0,_,_=source['roi_original_xyxy']
    for i, points_yx in enumerate(parts):
        pts = np.rint(points_yx[:, ::-1]).astype(int)
        # Keep full pixel geometry, not only a 14-click regression surrogate.
        arr=pts.tolist()
        candidates.append({'id':f'{source["tile_sha256"][:16]}:{i:04d}',
                           'centerline_local_xy_px':arr,
                           'centerline_source_xy_endpoints':[[arr[0][0]+x0,arr[0][1]+y0],
                                                           [arr[-1][0]+x0,arr[-1][1]+y0]],
                           'length_px':round(float(np.linalg.norm(np.diff(points_yx,axis=0),axis=1).sum()),3),
                           'origin':'fixed_frangi_p2_proposal_not_ground_truth'})
    out={'schema':SCHEMA, 'source':source, 'tile_shape_yx':list(image.shape),
         'pixels_sha256':sha256(image.tobytes()).hexdigest(),
         'candidate_method':'frangi_p2_fixed_fraction_0.16',
         'candidates':candidates,
         'label_semantics':'only manually confirmed centerline / artifact pixels are supervised',
         'unmarked_pixels':'UNKNOWN', 'automatic_labels_count':0}
    out['proposal_digest']=_digest(out)
    return out


def convert_review_to_sparse_mask(proposal: dict, review: dict) -> tuple[np.ndarray, dict]:
    if proposal.get('schema') != SCHEMA or review.get('schema') != SCHEMA:
        raise ValueError('unsupported review format')
    if review.get('proposal_digest') != proposal.get('proposal_digest'):
        raise ValueError('proposal digest mismatch: cannot transfer decisions to another crop')
    if proposal['proposal_digest'] != _digest({key:value for key,value in proposal.items() if key!='proposal_digest'}):
        raise ValueError('proposal payload was modified after generation')
    decisions = review.get('judgments')
    if not isinstance(decisions, list):
        raise ValueError('judgments must be an array')
    expected={p['id']:p for p in proposal['candidates']}
    seen=set()
    shape=tuple(proposal['tile_shape_yx'])
    positives=np.zeros(shape, dtype=np.uint8)
    artifacts=np.zeros(shape, dtype=np.uint8)
    counts={k:0 for k in ALLOWED}
    for item in decisions:
        key=item.get('candidate_id');value=item.get('decision')
        if key not in expected or key in seen or value not in ALLOWED:
            raise ValueError('unknown/duplicate candidate or invalid decision')
        seen.add(key);counts[value]+=1
        if value=='unknown':continue
        arr=np.asarray(expected[key]['centerline_local_xy_px'],dtype=np.int32).reshape(-1,1,2)
        if len(arr)<2:continue
        cv2.polylines(positives if value=='track_centerline' else artifacts,[arr],False,1,1,lineType=cv2.LINE_8)
    # Collision is not secretly resolved: uncertainty wins in contested pixels.
    conflict=(positives>0)&(artifacts>0)
    labels=np.full(shape,255,dtype=np.uint8)  # 255 is UNLABELLED, not background.
    labels[(artifacts>0)&~conflict]=0
    labels[(positives>0)&~conflict]=1
    stats={'reviewed_candidates':len(seen),'unreviewed_candidates':len(expected)-len(seen),
           'positive_centerline_pixels':int((labels==1).sum()),
           'negative_certified_artifact_pixels':int((labels==0).sum()),
           'unknown_pixels':int((labels==255).sum()),
           'conflicting_centerline_pixels_left_unknown':int(conflict.sum()),
           'counts':counts,'trainable_bce':bool((labels==1).any() and (labels==0).any()),
           'note':'255 pixels MUST be ignored by loss, not treated as background'}
    return labels,stats


def review_html(proposals: list[dict], images: list[np.ndarray]) -> str:
    if len(proposals)!=len(images) or not proposals:
        raise ValueError('source/proposal length mismatch')
    payload=[]
    for item,image in zip(proposals,images):
        if sha256(image.tobytes()).hexdigest()!=item['pixels_sha256'] or list(image.shape)!=item['tile_shape_yx']:
            raise ValueError('review image differs from proposed image')
        success,png=cv2.imencode('.png',image)
        if not success:raise ValueError('cannot encode review tile')
        data_uri='data:image/png;base64,'+base64.b64encode(png.tobytes()).decode()
        payload.append({**item,'image_data_uri':data_uri})
    safe=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return _HTML.replace('__PAYLOAD__',safe)


_HTML=r'''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bubble track segmentation review</title><style>
body{font:15px system-ui;background:#161b22;color:#e8ebee;margin:0;padding:20px}main{display:flex;gap:18px;flex-wrap:wrap}
#scene{background:#050607;border:1px solid #536070;max-height:82vh;overflow:auto}svg{max-width:75vw;height:auto;display:block}
button,select{background:#273b4d;color:white;border:1px solid #607d94;padding:10px;border-radius:7px;cursor:pointer;margin:3px}
button:focus-visible{outline:2px solid white}#info{max-width:340px;line-height:1.5}#status{white-space:pre-wrap}
.candidate{fill:none;stroke:#52d6ef;stroke-width:2;cursor:pointer;stroke-linejoin:round;stroke-linecap:round;opacity:.9}
.candidate.selected{stroke:#ffec65;stroke-width:4}.candidate.track_centerline{stroke:#3ded88}.candidate.certified_artifact{stroke:#fe6571}.candidate.unknown{stroke:#8b98a7}
</style></head><body><h2>Экспертная разметка треков пузырьковой камеры</h2>
<p>Кликните на предложенную линию. <strong>Зелёный = подтверждённый трек</strong>, красный = подтверждённый артефакт, серый = неизвестно. Непроверенное — НЕ фон. Никакого автоматического объединения линий.</p>
<main><div id="scene"><svg id="view" xmlns="http://www.w3.org/2000/svg"></svg></div><div id="info">
<p><label>Снимок <select id="tile"></select></label></p><p id="status"></p>
<p><button id="prev">←</button><button id="next">→</button></p>
<p><button id="positive">Подтвердить трек</button><button id="negative">Это артефакт</button><button id="uncertain">Неизвестно / другой трек</button></p>
<p><button id="save">Скачать решения JSON</button><label>Импорт решений <input id="load" type="file" accept=".json"/></label></p>
<p>Метки привязаны к исходному PNG и версии алгоритма через SHA-256. После проверки JSON преобразуется в частичную маску. Фон и другие треки остаются без меток.</p></div></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';const source=JSON.parse(document.getElementById('payload').textContent);let tileIndex=0,selected=null;
const decisions=Object.create(null);const q=id=>document.getElementById(id);const svg=q('view');
for(let i=0;i<source.length;i++){const opt=document.createElement('option');opt.value=i;opt.textContent=source[i].source.parent+' / '+source[i].source.label+' #'+source[i].source.source_index;q('tile').append(opt)}
const NS='http://www.w3.org/2000/svg';
function show(){const item=source[tileIndex];const [h,w]=item.tile_shape_yx;q('tile').value=String(tileIndex);svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width',w);svg.setAttribute('height',h);
 const bg=document.createElementNS(NS,'image');bg.setAttribute('href',item.image_data_uri);bg.setAttribute('width',w);bg.setAttribute('height',h);svg.append(bg);
 for(const c of item.candidates){const id=item.proposal_digest+':'+c.id;const line=document.createElementNS(NS,'polyline');line.setAttribute('points',c.centerline_local_xy_px.map(p=>p.join(',')).join(' '));line.setAttribute('class','candidate '+(decisions[id]||'')+(selected===id?' selected':''));line.onclick=()=>{selected=id;show()};svg.append(line)}
 const center=item.source.center_tile_xy;const circle=document.createElementNS(NS,'circle');circle.setAttribute('cx',center[0]);circle.setAttribute('cy',center[1]);circle.setAttribute('r',16);circle.setAttribute('stroke','#fbba45');circle.setAttribute('fill','none');circle.setAttribute('stroke-width',2);circle.setAttribute('pointer-events','none');svg.append(circle);
 q('status').textContent=`Кадр ${tileIndex+1}/${source.length}; фрагментов ${item.candidates.length}; выбран ${selected?.split(':').slice(-1)[0]||'нет'}\nПроверено: ${Object.keys(decisions).length}`;
}
function move(n){tileIndex=(tileIndex+n+source.length)%source.length;selected=null;show()}
function decide(value){if(!selected)return;decisions[selected]=value;show()}
q('tile').onchange=e=>{tileIndex=Number(e.target.value);selected=null;show()};q('prev').onclick=()=>move(-1);q('next').onclick=()=>move(1);
q('positive').onclick=()=>decide('track_centerline');q('negative').onclick=()=>decide('certified_artifact');q('uncertain').onclick=()=>decide('unknown');
q('save').onclick=()=>{const out=source.map(s=>({schema:s.schema,proposal_digest:s.proposal_digest,judgments:s.candidates.filter(c=>decisions[s.proposal_digest+':'+c.id]).map(c=>({candidate_id:c.id,decision:decisions[s.proposal_digest+':'+c.id]}))}));const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='bubble_dense_centerline_reviews.json';a.click();URL.revokeObjectURL(url)};
q('load').onchange=async e=>{try{const arr=JSON.parse(await e.target.files[0].text());if(!Array.isArray(arr)||arr.length!==source.length)throw Error('Другой набор');for(let i=0;i<arr.length;i++){if(arr[i].proposal_digest!==source[i].proposal_digest)throw Error('Неверные контрольные суммы');const ids=new Set(source[i].candidates.map(c=>c.id));for(const j of arr[i].judgments){if(!ids.has(j.candidate_id)||!['track_centerline','certified_artifact','unknown'].includes(j.decision))throw Error('Плохая метка');decisions[source[i].proposal_digest+':'+j.candidate_id]=j.decision}}show()}catch(err){alert(err.message)}};
show();
</script></body></html>'''
