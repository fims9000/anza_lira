"""S19 offline expert workbench for unlabelled H50Z photographs.

Create a source-locked review HTML and compile ONLY actual expert decisions into
0/1/255 masks + sparse pair labels. Classical proposals and START landmarks
are never automatically promoted to expert labels, nor do pixel probabilities
become instance-ground-truth. This module does not perform physical inference.
"""
from __future__ import annotations

import base64
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from .track_mask_review import SCHEMA as OLD_SCHEMA

SCHEMA = 's19_real_photo_sparse_expert_review_v1'
PACKAGE_SCHEMA = 's19_real_photo_review_package_v1'
LABELS = {'track_centerline', 'certified_artifact', 'unknown'}
STROKE_LABELS = {'track_centerline', 'certified_artifact'}
PAIR_LABELS = {'same_track', 'different_track', 'uncertain'}
_TILE_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_PARENT_RE = re.compile(r'^(H50Z_[0-9]{3}_[0-9]{2})_([0-9]{3})\.bmp$')


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(obj: Any) -> str:
    return sha256(_canonical(obj)).hexdigest()


def load_package(proposals_json: Path, image_root: Path) -> dict:
    """Validate original proposal digest AND byte identity of every tile."""
    path = Path(proposals_json)
    root = Path(image_root).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.get('schema') != 's15_real_photo_sparse_proposals_v1':
        raise ValueError('unsupported Frangi proposal source schema')
    records = raw.get('records')
    proposals = raw.get('proposals')
    if not isinstance(records, list) or not isinstance(proposals, list) or not records or len(records) != len(proposals):
        raise ValueError('records/proposals are missing or mismatched')
    tiles = []
    seen = set()
    for record, proposal in zip(records, proposals):
        if not isinstance(record, dict) or not isinstance(proposal, dict):
            raise ValueError('malformed source')
        name = record.get('crop_file')
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.png'):
            raise ValueError('crop file must be a local PNG with no path traversal')
        tile_id = Path(name).stem
        if not _TILE_RE.fullmatch(tile_id) or tile_id in seen:
            raise ValueError('duplicate or invalid crop ID')
        seen.add(tile_id)
        candidate_digest = proposal.get('proposal_digest')
        if candidate_digest != _sha({k: v for k, v in proposal.items() if k != 'proposal_digest'}):
            raise ValueError('Frangi proposal digest mismatch')
        if candidate_digest != record.get('proposal_digest') or proposal.get('schema') != OLD_SCHEMA:
            raise ValueError('source proposal provenance mismatch')
        source = proposal.get('source')
        if not isinstance(source, dict) or source.get('parent') != record.get('parent'):
            raise ValueError('parent provenance mismatch')
        parsed_photo = _PARENT_RE.fullmatch(source['parent'])
        if not parsed_photo:
            raise ValueError('H50Z source photo name does not identify a physical run/frame')
        run_id = parsed_photo.group(1)
        film_id = run_id.rsplit('_',1)[0]
        if not isinstance(source.get('image_sha256'), str) or len(source['image_sha256']) != 64:
            raise ValueError('source image digest invalid')
        if source.get('label') != 'start':
            raise ValueError('this package expects start-based ROIs, not certified tracks')
        image_path = (root / name).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError('source tile outside the source folder or missing')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.dtype != np.uint8 or list(image.shape) != proposal.get('tile_shape_yx'):
            raise ValueError('tile cannot be decoded as source grayscale shape')
        raw_sha = sha256(image.tobytes()).hexdigest()
        if any(raw_sha != s for s in [proposal.get('pixels_sha256'), source.get('tile_sha256'),
                                    record.get('crop_gray_pixels_sha256')]):
            raise ValueError('source tile pixels changed')
        box = source.get('roi_original_xyxy')
        h, w = image.shape
        if not (isinstance(box, list) and len(box) == 4 and
                all(isinstance(v, int) for v in box) and box[2]-box[0] == w and box[3]-box[1] == h):
            raise ValueError('source ROI dimensions invalid')
        candidates = proposal.get('candidates')
        if not isinstance(candidates, list):
            raise ValueError('no candidates list')
        ids = set()
        for candidate in candidates:
            cid = candidate.get('id')
            coords = candidate.get('centerline_local_xy_px')
            if not isinstance(cid, str) or cid in ids or not isinstance(coords, list) or len(coords) < 2:
                raise ValueError('duplicate/bad proposal candidate')
            ids.add(cid)
            for point in coords:
                if not _valid_xy(point, w, h):
                    raise ValueError('candidate coordinates out of range')
        tiles.append({'tile_id': tile_id, 'crop_file': name, 'proposal_digest': candidate_digest,
                      'source': source, 'tile_shape_yx': [h,w], 'candidates': candidates,
                      'pixels_sha256': raw_sha, 'parent_photo': source['parent'],
                      'run_id':run_id,'film_id':film_id})
    base = {'schema': PACKAGE_SCHEMA, 'tiles': tiles}
    base['package_sha256'] = _sha(base)
    return base


def _valid_xy(p: Any, w: int, h: int) -> bool:
    return (isinstance(p, list) and len(p) == 2 and
            all(isinstance(v, int) and not isinstance(v, bool) for v in p) and
            0 <= p[0] < w and 0 <= p[1] < h)


def empty_review(package: dict) -> dict:
    return {'schema': SCHEMA, 'package_sha256': package['package_sha256'],
            'entries': [{'tile_id': t['tile_id'], 'proposal_digest': t['proposal_digest'],
                         'decisions': [], 'strokes': [], 'links': []}
                        for t in package['tiles']]}


def _validate_entries(package: dict, reviews: dict) -> list[dict]:
    if not isinstance(reviews, dict) or reviews.get('schema') != SCHEMA or reviews.get('package_sha256') != package['package_sha256']:
        raise ValueError('review file does not match the source photo set')
    raw_entries = reviews.get('entries')
    if not isinstance(raw_entries, list):
        raise ValueError('review entries must be a list')
    allowed = {tile['tile_id']: tile for tile in package['tiles']}
    reviewed = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or entry.get('tile_id') not in allowed:
            raise ValueError('unknown source tile in reviews')
        tile_id = entry['tile_id']
        if tile_id in reviewed or entry.get('proposal_digest') != allowed[tile_id]['proposal_digest']:
            raise ValueError('duplicate or stale source tile in reviews')
        for field in ('decisions', 'strokes', 'links'):
            if not isinstance(entry.get(field), list):
                raise ValueError('review field missing: ' + field)
        reviewed[tile_id] = entry
    # Not-filled tiles are VALID and remain unknown; never converted to background.
    return [reviewed.get(t['tile_id'], {'tile_id': t['tile_id'], 'proposal_digest': t['proposal_digest'],
                                       'decisions': [], 'strokes': [], 'links': []})
            for t in package['tiles']]


def _paint(canvas: np.ndarray, coords: list[list[int]], radius: int) -> None:
    ar = np.asarray(coords, dtype=np.int32)
    if len(ar) > 1:
        cv2.polylines(canvas, [ar.reshape(-1,1,2)], False, 1, radius*2+1, lineType=cv2.LINE_8)
    for x,y in (ar[0], ar[-1]):
        cv2.circle(canvas, (int(x),int(y)), radius, 1, thickness=-1, lineType=cv2.LINE_8)


def _best_orient(a: list[list[int]], b: list[list[int]]) -> tuple[list, list]:
    """Geometry-only orientation; avoids using expert same/different decision."""
    pairs = []
    for ar in (False, True):
        for br in (False, True):
            lhs = a[::-1] if ar else a
            rhs = b[::-1] if br else b
            dist = float(np.linalg.norm(np.asarray(lhs[-1],float)-np.asarray(rhs[0],float)))
            pairs.append((dist,ar,br,lhs,rhs))
    return min(pairs, key=lambda x:(x[0],x[1],x[2]))[3:]


def compile_review(package: dict, reviews: dict, *, out_dir: Path | None = None,
                   image_root: Path | None = None) -> dict:
    """Validate every expert stroke/link and produce supervised masks without negatives by absence.

    If out_dir is set, files are written *only after all review entries validate*.
    Unlabelled tiles are retained as 'unlabeled'; no real test split is invented.
    """
    reviewed = _validate_entries(package, reviews)
    all_masks = []
    rows = []
    overview = []
    pairs_output = []
    for tile, entry in zip(package['tiles'], reviewed):
        h, w = tile['tile_shape_yx']
        candidates = {c['id']: c for c in tile['candidates']}
        labels = {}
        fg = np.zeros((h,w), np.uint8)
        bg = np.zeros((h,w), np.uint8)
        for d in entry['decisions']:
            if not isinstance(d,dict) or d.get('candidate_id') not in candidates or d.get('label') not in LABELS:
                raise ValueError('invalid candidate review decision')
            cid = d['candidate_id']
            if cid in labels:
                raise ValueError('duplicate candidate decision')
            labels[cid] = d['label']
            if labels[cid] in STROKE_LABELS:
                points = candidates[cid]['centerline_local_xy_px']
                _paint(fg if labels[cid] == 'track_centerline' else bg, points, 1)
        manual = {}
        for stroke in entry['strokes']:
            if not isinstance(stroke, dict) or not isinstance(stroke.get('id'), str):
                raise ValueError('bad manual stroke')
            sid = stroke['id']
            if not re.fullmatch('manual:'+re.escape(tile['tile_id'])+r':[0-9]+', sid) or sid in manual or sid in candidates:
                raise ValueError('duplicate or forged stroke ID')
            pts = stroke.get('points_xy')
            radius = stroke.get('radius_px')
            if (stroke.get('label') not in STROKE_LABELS or not isinstance(pts, list) or not (1 <= len(pts) <= 2000)
                or not isinstance(radius, int) or isinstance(radius,bool) or not (1 <= radius <= 8)
                or any(not _valid_xy(point, w, h) for point in pts)):
                raise ValueError('invalid expert stroke geometry/label')
            manual[sid] = stroke
            _paint(fg if stroke['label'] == 'track_centerline' else bg, pts, radius)
        conflict = (fg > 0) & (bg > 0)
        mask = np.full((h,w), 255, np.uint8)
        mask[(fg>0)&~conflict] = 1
        mask[(bg>0)&~conflict] = 0
        all_masks.append(mask)
        all_segments = {cid: (c['centerline_local_xy_px'], labels.get(cid,'unknown')) for cid,c in candidates.items()}
        all_segments.update({sid: (s['points_xy'],s['label']) for sid,s in manual.items()})
        seen_links = set()
        for link in entry['links']:
            if not isinstance(link,dict) or link.get('decision') not in PAIR_LABELS:
                raise ValueError('invalid expert link')
            a, b = link.get('a'), link.get('b')
            if not isinstance(a,str) or not isinstance(b,str) or a==b or a not in all_segments or b not in all_segments:
                raise ValueError('link refers to missing/different fragments')
            lk = tuple(sorted((a,b)))
            if lk in seen_links:
                raise ValueError('duplicate pair review')
            seen_links.add(lk)
            if link['decision']=='uncertain':
                continue
            ca, la = all_segments[a]
            cb, lb = all_segments[b]
            if la!='track_centerline' or lb!='track_centerline':
                raise ValueError('cannot train same/different labels for fragments not confirmed to be tracks')
            if len(ca)<3 or len(cb)<3:
                raise ValueError('reviewed relation needs at least three annotated points per fragment')
            left, right = _best_orient(ca,cb)
            if (np.linalg.norm(np.asarray(left[-1])-np.asarray(left[-3])) < 1e-6 or
                np.linalg.norm(np.asarray(right[2])-np.asarray(right[0])) < 1e-6):
                raise ValueError('degenerate reviewed fragment tangent')
            numbered = {token: k for k,token in enumerate(sorted(all_segments))}
            ox,oy=tile['source']['roi_original_xyxy'][:2]
            pairs_output.append({'schema':'s19_expert_track_pair_v1', 'film_id':tile['film_id'],
                                 'run_id':tile['run_id'],'frame_id':tile['parent_photo'],
                                 'scene_id':tile['tile_id'], 'left_segment':numbered[a],
                                 'right_segment':numbered[b], 'left_fragment_id':a,'right_fragment_id':b,
                                 'source_sha256':tile['pixels_sha256'],
                                 'left_points_xy':left,'right_points_xy':right,
                                 'source_photo_sha256':tile['source']['image_sha256'],
                                 'roi_original_xyxy':tile['source']['roi_original_xyxy'],
                                 'left_points_original_xy':[[x+ox,y+oy] for x,y in left],
                                 'right_points_original_xy':[[x+ox,y+oy] for x,y in right],
                                 'expert_label':link['decision'], 'orientation':'minimum_endpoint_distance_unverified',
                                 'verified_human':True})
        overview.append({'tile_id':tile['tile_id'], 'parent':tile['parent_photo'],
                         'confirmed_centerline_pixels':int((mask==1).sum()),
                         'certified_artifact_pixels':int((mask==0).sum()),
                         'unknown_pixels':int((mask==255).sum()),
                         'contested_pixels_left_unknown':int(conflict.sum()),
                         'expert_fragment_decisions':len(labels),
                         'expert_manual_strokes':len(manual),
                         'expert_pair_decisions':len(entry['links'])})
        # This avoids inventing real test labels: only two source parents from one run.
        parents = sorted({t['parent_photo'] for t in package['tiles']})
        split = 'train' if tile['parent_photo'] == parents[0] else ('val' if len(parents)>1 and tile['parent_photo'] == parents[1] else 'unlabeled')
        if not np.any((mask == 0) | (mask == 1)):
            split='unlabeled'  # genuinely no expert labels; keep all unknown
        rows.append({'film_id':tile['film_id'],'run_id':tile['run_id'],
                     'frame_id':tile['parent_photo'],'split':split,'image':tile['crop_file'],
                     'mask':f"{tile['tile_id']}_expert_sparse_mask.png" if split != 'unlabeled' else None,
                     'mask_data':mask})
    summary = {'schema':'s19_real_review_compilation_v1','source_package_sha256':package['package_sha256'],
               'real_photo_parent_count':len({t['parent_photo'] for t in package['tiles']}),
               'real_photo_runs':len({t['run_id'] for t in package['tiles']}),
               'real_dense_pixel_ground_truth':False,'independent_physical_run_test_available':False,
               'auto_certified_track_pixels':0,'auto_certified_track_links':0,
               'confirmed_positive_pixels':sum(x['confirmed_centerline_pixels'] for x in overview),
               'confirmed_negative_pixels':sum(x['certified_artifact_pixels'] for x in overview),
               'reviewed_relation_labels':len(pairs_output),
               'per_tile':overview,
               'safe_for_physical_automatic_merging':False,
               'trainable_real_image_baseline':False}
    train_parent={x['frame_id'] for x in rows if x['split']=='train'}
    val_parent={x['frame_id'] for x in rows if x['split']=='val'}
    def both_classes(split):
        selected = [x['mask_data'] for x in rows if x['split']==split]
        return bool(any((m==1).any() for m in selected) and any((m==0).any() for m in selected))
    trainable=bool(train_parent and val_parent and train_parent.isdisjoint(val_parent)
                   and both_classes('train') and both_classes('val'))
    summary['trainable_real_image_baseline']=trainable
    if out_dir is not None:
        if image_root is None:
            raise ValueError('image_root required when writing image/mask output')
        out = Path(out_dir)
        root = Path(image_root).resolve()
        # Verify original source PNG and digest again, do not trust copied metadata.
        for tile in package['tiles']:
            f=(root/tile['crop_file']).resolve()
            if not f.is_relative_to(root) or not f.is_file():
                raise ValueError('missing or unsafe source PNG')
            source_img=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
            if source_img is None or sha256(source_img.tobytes()).hexdigest()!=tile['pixels_sha256']:
                raise ValueError('real source PNG changed after review')
        out.mkdir(parents=True,exist_ok=True)
        entries=[]
        for row in rows:
            file=root/row['image']
            target=out/row['image']
            target.write_bytes(file.read_bytes())
            outrow={k:v for k,v in row.items() if k!='mask_data'}
            outrow['image_sha256']=sha256(target.read_bytes()).hexdigest()
            if row['mask'] is not None:
                path=out/row['mask']
                if not cv2.imwrite(str(path),row['mask_data']):
                    raise IOError('failed to write sparse expert mask')
                outrow['mask_sha256']=sha256(path.read_bytes()).hexdigest()
            entries.append(outrow)
        manifest={'schema':'s14_sparse_image_reviews_v1','entries':entries}
        # Always publish provenance and reviewed labels; only enable supervised
        # train path with source-photo-disjoint train AND validation.
        (out/'s19_review_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'s19_pair_labels.json').write_text(json.dumps({'schema':'s19_expert_pair_labels_v1','rows':pairs_output,
             'source_package_sha256':package['package_sha256'],'independent_run_calibrated':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'s14_sparse_image_reviews.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def make_offline_review_html(package:dict, image_root:Path)->str:
    root=Path(image_root).resolve()
    payload=[]
    for tile in package['tiles']:
        path=(root/tile['crop_file']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('image missing')
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None or sha256(image.tobytes()).hexdigest()!=tile['pixels_sha256']:
            raise ValueError('review page image does not match source')
        good,encoded=cv2.imencode('.png',image)
        if not good:raise IOError('PNG encoding failed')
        payload.append({'tile_id':tile['tile_id'], 'proposal_digest':tile['proposal_digest'],
                        'parent_photo':tile['parent_photo'],'image_data_uri':'data:image/png;base64,'+base64.b64encode(encoded.tobytes()).decode(),
                        'tile_shape_yx':tile['tile_shape_yx'], 'candidates':tile['candidates'],
                        'source_photo_hash':tile['source']['image_sha256'],
                        'roi_original_xyxy':tile['source']['roi_original_xyxy']})
    data={'schema':SCHEMA,'package_sha256':package['package_sha256'],'tiles':payload}
    safe=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return _HTML.replace('___JSON_PAYLOAD___',safe)


_HTML=r'''<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>S19 • Разметка треков H50Z</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111821;color:#f3f6fa}
header{padding:16px 24px;border-bottom:1px solid #374351}h1{font-size:20px;margin:0 0 5px}p{margin:6px 0;line-height:1.45}
main{display:grid;grid-template-columns:minmax(320px,1fr) 340px;gap:20px;padding:18px;align-items:start}svg{width:min(100%,760px);height:auto;background:#0b0e12;border:1px solid #415363;touch-action:none;display:block;user-select:none}
label{display:inline-block;margin:8px 0}select,input[type="number"],button{background:#233142;color:#f2f5fa;border:1px solid #52677d;border-radius:7px;padding:8px;font:inherit;cursor:pointer}button{margin:3px 2px}button:hover{background:#31506a}button.active{background:#24745c}button:focus-visible{outline:2px solid #e9ca65}
.panel{background:#1c2530;border-radius:10px;padding:14px;margin:0 0 12px}.panel h2{font-size:15px;margin:0 0 9px}#status,#selection,#pairStatus{white-space:pre-line;font-size:13px;color:#c8d7e7}#status{margin:10px 0}
.proposal{fill:none;stroke:#54d5ef;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round;cursor:pointer;vector-effect:non-scaling-stroke}
.proposal.track_centerline,.manual.track_centerline{stroke:#28e89a}.proposal.certified_artifact,.manual.certified_artifact{stroke:#ff7373}.proposal.selected,.manual.selected{stroke:#ffe68b;stroke-width:4}
.manual{fill:none;stroke-width:5;stroke-linejoin:round;stroke-linecap:round;cursor:pointer;vector-effect:non-scaling-stroke}
.preview{fill:none;stroke:#ffe68b;stroke-width:3;pointer-events:none}.hint{font-size:12px;color:#b3c5d6}.accent{color:#e7d28b}input[type=file]{max-width:100%}@media(max-width:920px){main{grid-template-columns:1fr}} </style></head>
<body><header><h1>Экспертная разметка H50Z · S19</h1>
<p>Зелёным отмечайте <b>видимую линию трека</b>, красным — <b>подтверждённый артефакт</b>. Неразмеченное остаётся <b>НЕИЗВЕСТНЫМ</b>. Реперные отметки и «start» — не доказательство принадлежности частицы.</p>
<p class="hint">Локальный HTML: никакие фотографии или ответы не отправляются на сервер. Сохраняйте файл JSON вручную. Разметка не является подтверждением типа частицы.</p></header>
<main><section><svg id="scene" xmlns="http://www.w3.org/2000/svg" aria-label="Фото и выделение линий"></svg><p class="hint">Для добавления пропущенной линии выберите «Дорисовать трек» и проведите мышью или пальцем вдоль неё. Для выделения предложенного фрагмента переключитесь в режим «Выбрать».</p></section>
<aside><div class="panel"><h2>1. Фотография</h2><label>Кадр <select id="tile"></select></label><div><button id="prev">← Назад</button><button id="next">Далее →</button></div><p id="status"></p></div>
<div class="panel"><h2>2. Подтвердить / добавить линию</h2><div><button id="select" class="active">Выбрать линию</button><button id="brushTrack">Дорисовать трек</button><button id="brushArtifact">Отметить артефакт</button></div>
<label>Радиус кисти <input type="number" id="radius" min="1" max="8" value="2" style="width:64px"/></label><p id="selection">Выделения пока нет</p>
<div><button id="markTrack">Это трек</button><button id="markArtifact">Это артефакт</button><button id="markUnknown">Неизвестно</button><button id="removeStroke">Удалить ручную линию</button><button id="undo">Отменить</button></div></div>
<div class="panel"><h2>3. Связность двух линий</h2><p class="hint">Сначала подтвердите обе линии как треки. Выберите первую и нажмите A, затем вторую и нажмите Б.</p>
<button id="setA">Назначить A</button><button id="setB">Назначить Б</button><p id="pairStatus"></p>
<button id="pairSame">Одна частица</button><button id="pairDifferent">Разные треки</button><button id="pairUncertain">Не уверен</button></div>
<div class="panel"><h2>4. Сохранение</h2><button id="save">Скачать экспертные решения JSON</button><label>Продолжить из JSON <input type="file" id="load" accept=".json,application/json"/></label><p class="hint">Файл привязан к хешам исходных фрагментов. Даже после экспорта все непроверенные пиксели останутся неизвестными.</p></div></aside></main>
<script id="payload" type="application/json">___JSON_PAYLOAD___</script><script>
'use strict';
const pack=JSON.parse(document.getElementById('payload').textContent), tiles=pack.tiles;
let tileIdx=0,selected=null,tool='select',drawing=null;
const states=tiles.map(()=>({decisions:{},strokes:[],links:[],a:null,b:null,undo:[],counter:0}));
const byId=id=>document.getElementById(id), scene=byId('scene'), NS='http://www.w3.org/2000/svg';
for(let i=0;i<tiles.length;i++){const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${tiles[i].parent_photo} · ${tiles[i].tile_id.split('_').slice(-2).join('_')}`;byId('tile').append(o)}
const state=()=>states[tileIdx];
function allFragments(){const t=tiles[tileIdx],s=state();return [...t.candidates.map(c=>({id:c.id,label:s.decisions[c.id]||'unknown',points:c.centerline_local_xy_px,kind:'candidate'})),...s.strokes.map(c=>({id:c.id,label:c.label,points:c.points_xy,kind:'stroke'}))]}
function pushUndo(){const s=state();s.undo.push(JSON.stringify({decisions:s.decisions,strokes:s.strokes,links:s.links,a:s.a,b:s.b,counter:s.counter}));if(s.undo.length>40)s.undo.shift()}
function drawSvg(){const t=tiles[tileIdx],s=state(),[h,w]=t.tile_shape_yx;scene.replaceChildren();scene.setAttribute('viewBox',`0 0 ${w} ${h}`);scene.setAttribute('width',w);scene.setAttribute('height',h);
 const image=document.createElementNS(NS,'image');image.setAttribute('href',t.image_data_uri);image.setAttribute('width',w);image.setAttribute('height',h);scene.append(image);
 for(const f of allFragments()){const line=document.createElementNS(NS,'polyline');line.setAttribute('points',f.points.map(p=>p.join(',')).join(' '));line.setAttribute('class',(f.kind==='candidate'?'proposal ':'manual ')+f.label+(selected===f.id?' selected':''));line.dataset.fragment=f.id;line.onclick=event=>{if(tool!=='select')return;selected=f.id;drawSvg();event.stopPropagation()};scene.append(line)}
 const center=document.createElementNS(NS,'circle');center.setAttribute('cx',w/2);center.setAttribute('cy',h/2);center.setAttribute('r',14);center.setAttribute('stroke','#f3d460');center.setAttribute('stroke-width','2');center.setAttribute('fill','none');center.setAttribute('pointer-events','none');scene.append(center);
 if(drawing){const pre=document.createElementNS(NS,'polyline');pre.setAttribute('points',drawing.points.map(p=>p.join(',')).join(' '));pre.setAttribute('class','preview');scene.append(pre)}
 byId('tile').value=String(tileIdx);let f=allFragments(),decided=f.filter(x=>x.label!=='unknown').length;
 byId('status').textContent=`Фрагмент ${tileIdx+1}/${tiles.length} · исходно предложено ${t.candidates.length}\nПодтверждено объектов: ${decided}; вручную нарисовано: ${s.strokes.length}; проверено пар: ${s.links.length}`;
 byId('selection').textContent=selected?`Выбрано ${selected} · ${f.find(x=>x.id===selected)?.label||'нет'}`:'Выделения пока нет';
 byId('pairStatus').textContent=`A: ${s.a||'не выбрано'}\nБ: ${s.b||'не выбрано'}`;
 for(const btn of ['select','brushTrack','brushArtifact'])byId(btn).classList.toggle('active',tool===btn);
}
function changeTile(index){tileIdx=(index+tiles.length)%tiles.length;selected=null;drawing=null;drawSvg()}
function mark(label){const f=allFragments().find(x=>x.id===selected);if(!f)return alert('Сначала выберите линию');const s=state();if(f.kind==='stroke')return alert('Для исправления ручной линии отмените действие и нарисуйте её заново');pushUndo();s.decisions[f.id]=label;if(label!=='track_centerline'){s.links=s.links.filter(x=>x.a!==f.id&&x.b!==f.id);if(s.a===f.id)s.a=null;if(s.b===f.id)s.b=null}drawSvg()}
function assign(side){const f=allFragments().find(x=>x.id===selected);if(!f||f.label!=='track_centerline')return alert('Подтвердите выбранный фрагмент как трек');state()[side]=selected;drawSvg()}
function pair(decision){const s=state(),f=allFragments();if(!s.a||!s.b||s.a===s.b)return alert('Выберите две разные линии A и Б');if(f.find(x=>x.id===s.a)?.label!=='track_centerline'||f.find(x=>x.id===s.b)?.label!=='track_centerline')return alert('Оба фрагмента должны быть подтверждёнными треками');pushUndo();const key=[s.a,s.b].sort().join('|');s.links=s.links.filter(x=>[x.a,x.b].sort().join('|')!==key);s.links.push({a:s.a,b:s.b,decision});s.a=null;s.b=null;drawSvg()}
function where(e){const t=tiles[tileIdx],b=scene.getBoundingClientRect(),w=t.tile_shape_yx[1],h=t.tile_shape_yx[0];const x=Math.floor((e.clientX-b.left)/b.width*w),y=Math.floor((e.clientY-b.top)/b.height*h);return [Math.max(0,Math.min(w-1,x)),Math.max(0,Math.min(h-1,y))]}
scene.addEventListener('pointerdown',e=>{if(tool==='select')return;if(e.button!==undefined&&e.button!==0)return;const p=where(e);drawing={points:[p],label:tool==='brushTrack'?'track_centerline':'certified_artifact'};scene.setPointerCapture(e.pointerId);drawSvg()});
scene.addEventListener('pointermove',e=>{if(!drawing)return;const p=where(e),a=drawing.points;if((p[0]-a[a.length-1][0])**2+(p[1]-a[a.length-1][1])**2>=2){if(a.length<2000)a.push(p);drawSvg()}});
function endDraw(){if(!drawing)return;const s=state();const pts=drawing.points,label=drawing.label;drawing=null;if(pts.length>2000)return;pushUndo();const id=`manual:${tiles[tileIdx].tile_id}:${s.counter++}`;s.strokes.push({id,label,radius_px:Math.max(1,Math.min(8,Number(byId('radius').value)||2)),points_xy:pts});selected=id;drawSvg()}
scene.addEventListener('pointerup',endDraw);scene.addEventListener('pointercancel',()=>{drawing=null;drawSvg()});
byId('tile').onchange=e=>changeTile(Number(e.target.value));byId('prev').onclick=()=>changeTile(tileIdx-1);byId('next').onclick=()=>changeTile(tileIdx+1);
for(const id of ['select','brushTrack','brushArtifact'])byId(id).onclick=()=>{tool=id;drawSvg()};
byId('markTrack').onclick=()=>mark('track_centerline');byId('markArtifact').onclick=()=>mark('certified_artifact');byId('markUnknown').onclick=()=>mark('unknown');
byId('removeStroke').onclick=()=>{const s=state(),f=s.strokes.find(x=>x.id===selected);if(!f)return alert('Выберите вручную нарисованную линию');pushUndo();s.strokes=s.strokes.filter(x=>x.id!==f.id);s.links=s.links.filter(x=>x.a!==f.id&&x.b!==f.id);if(s.a===f.id)s.a=null;if(s.b===f.id)s.b=null;selected=null;drawSvg()};
byId('setA').onclick=()=>assign('a');byId('setB').onclick=()=>assign('b');
byId('pairSame').onclick=()=>pair('same_track');byId('pairDifferent').onclick=()=>pair('different_track');byId('pairUncertain').onclick=()=>pair('uncertain');
byId('undo').onclick=()=>{const s=state();if(!s.undo.length)return;const previous=JSON.parse(s.undo.pop());Object.assign(s,previous);selected=null;drawSvg()};
function exported(){return {schema:pack.schema,package_sha256:pack.package_sha256,entries:states.map((s,i)=>({tile_id:tiles[i].tile_id,proposal_digest:tiles[i].proposal_digest,decisions:Object.entries(s.decisions).map(([candidate_id,label])=>({candidate_id,label})),strokes:s.strokes,links:s.links}))}}
byId('save').onclick=()=>{const content=JSON.stringify(exported(),null,2),blob=new Blob([content],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='S19_H50Z_expert_reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500)};
byId('load').onchange=async e=>{try{const file=e.target.files?.[0];if(!file)return;const obj=JSON.parse(await file.text());if(obj.schema!==pack.schema||obj.package_sha256!==pack.package_sha256||!Array.isArray(obj.entries))throw Error('Контрольная сумма набора не совпадает');const parsed=states.map(()=>({decisions:{},strokes:[],links:[],a:null,b:null,undo:[],counter:0})),byTile=new Map();for(const v of obj.entries){if(byTile.has(v.tile_id)||!tiles.some(t=>t.tile_id===v.tile_id))throw Error('Дублированный или посторонний кадр');byTile.set(v.tile_id,v)};
 for(let i=0;i<tiles.length;i++){const t=tiles[i],v=byTile.get(t.tile_id);if(!v)continue;if(v.proposal_digest!==t.proposal_digest)throw Error('Другая версия сегментов');const ids=new Set(t.candidates.map(c=>c.id));if(!Array.isArray(v.decisions)||!Array.isArray(v.strokes)||!Array.isArray(v.links))throw Error('Неверный формат');for(const d of v.decisions){if(!ids.has(d.candidate_id)||!['track_centerline','certified_artifact','unknown'].includes(d.label))throw Error('Неизвестная метка');parsed[i].decisions[d.candidate_id]=d.label};for(const s of v.strokes){if(!/^manual:[a-zA-Z0-9_-]+:[0-9]+$/.test(s.id)||!s.id.startsWith('manual:'+t.tile_id+':')||!['track_centerline','certified_artifact'].includes(s.label)||!Array.isArray(s.points_xy)||!Number.isInteger(s.radius_px)||s.radius_px<1||s.radius_px>8)throw Error('Неверная линия');parsed[i].strokes.push(s);parsed[i].counter=Math.max(parsed[i].counter,Number(s.id.split(':').pop())+1)};parsed[i].links=v.links;}
 for(let i=0;i<states.length;i++)Object.assign(states[i],parsed[i]);selected=null;drawSvg()}catch(e){alert('Импорт отклонён: '+e.message)}};
drawSvg();
</script></body></html>'''
