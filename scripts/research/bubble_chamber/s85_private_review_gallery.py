#!/usr/bin/env python3
"""PRIVATE offline H50Z 2D proposal gallery for subsequent physicist review.

No fake physical GT, no raw source coordinates in public source, and no
photographs are written anywhere in the tracked repository. Genuine images are
never embedded in scientific metric JSON; all joins remain UNKNOWN.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85
from scripts.research.bubble_chamber.s85_release_manager import _safe_directory, preflight, _write_once
from bubble_chamber.s85_fragment_ranker import ARMS,model_sha256


PALETTE={
    'source_line':(240,210,20),  # BGR cyan
    'extra_proposal':(170,55,245), # BGR pink / violet
    'UNKNOWN_join':(70,225,120), # BGR green
}


def render_overlay(gray:np.ndarray, pred:dict, *,title:str,source_count:int)->np.ndarray:
    """Only decorate image-derived hypotheses, never claim correct particle IDs."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Need original uint8 192px machine-only pixels')
    if not isinstance(source_count,int) or source_count<0 or source_count>len(pred['polylines']):
        raise ValueError('Invalid frozen base line count')
    canvas=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for i,curve in enumerate(pred['polylines']):
        points=np.asarray(curve,float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
            raise ValueError('Corrupted image-derived line')
        vertices=np.clip(np.rint(points),0,191).astype(np.int32)
        cv2.polylines(canvas,[vertices.reshape(-1,1,2)],False,
                      PALETTE['source_line'] if i<source_count else PALETTE['extra_proposal'],
                      1,cv2.LINE_AA)
        if i>=source_count and len(vertices)>0:
            cv2.circle(canvas,tuple(vertices[len(vertices)//2]),2,PALETTE['extra_proposal'],-1)
    for edge in pred['chosen']:
        a=np.asarray(pred['polylines'][edge.i][edge.end_i],float)
        b=np.asarray(pred['polylines'][edge.j][edge.end_j],float)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('Invalid edge coordinates')
        cv2.line(canvas,tuple(np.clip(np.rint(a),0,191).astype(int)),
                 tuple(np.clip(np.rint(b),0,191).astype(int)),
                 PALETTE['UNKNOWN_join'],1,cv2.LINE_AA)
    scale=cv2.resize(canvas,(576,576),interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(scale,(0,0),(575,52),(12,22,33),-1)
    cv2.putText(scale,title[:52],(10,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(scale,'2D MACHINE HYPOTHESES / NO PHYSICS GT',(10,43),
                cv2.FONT_HERSHEY_SIMPLEX,.44,(245,245,245),1,cv2.LINE_AA)
    return scale


def offline_html(rows):
    """Self-contained static HTML: no external images, scripts, login, cookies."""
    cards=[]
    for idx,item in enumerate(rows,1):
        ims=''.join(
           '<article class="card"><h3>'+html.escape(v['display'])+'</h3>'+
           '<img loading="lazy" src="'+html.escape(v['image'])+'" alt="Only UNKNOWN machine proposals"/>'+
           f'<p>{v["lines"]} линии · {v["extra"]} дополнительных · {v["joins"]} связей UNKNOWN</p></article>'
           for v in item['variants'])
        cards.append(f'<section class="scene" data-photo="{html.escape(item["photo_group"])}">'+
           '<h2>Участок '+str(idx).zfill(2)+' · фотография '+html.escape(item['photo_group'])+'</h2>'+
           '<div class="grid">'+ims+'</div></section>')
    return '''<!doctype html><html lang="ru"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ANZA–LIRA S85 · только машинные гипотезы H50Z</title>
<style>body{font:15px/1.5 system-ui,sans-serif;background:#0e1726;color:#e4ecf5;margin:0}
main{max-width:1550px;margin:auto;padding:25px 19px 90px}h1{font-size:clamp(24px,3vw,36px)}
h2{font-size:18px}h3{font-size:15px}.warn{border-left:4px solid #73d8c9;background:#1a3044;padding:18px;border-radius:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:13px}
.card{background:#1b2b3f;border:1px solid #364b63;border-radius:9px;padding:12px}
.card img{width:100%;height:auto;display:block}.card p{font-size:13px;color:#b8cbdc}
.scene{margin:32px 0 40px}small{color:#b8cbdc}select{background:#243447;color:white;padding:8px;border-radius:6px}
</style></head><body><main><h1>ANZA–LIRA · S85</h1>
<div class="warn"><strong>Это не доказанные физические треки.</strong> Все линии — 2D машинные предложения, все графовые связи UNKNOWN; экспертная GT отсутствует. Голубое — исходные линии S72, фиолетовое — дополнительно отобранные S85 кандидаты, зелёное — предложенные графом стыки. Граф не измеряет 3D, импульс или тип частицы. Фотографии относятся к одному физическому запуску. Архив ПРИВАТНЫЙ.</div>
<p>Показаны центральные 192×192 области оригинальных 512×512 H50Z фрагментов. Ни один кадр не загружается в интернет. Метки и оценки правильности физических событий не создавались.</p>
<label>Фильтр: <select id="photo"><option value="all">Все фотографии</option><option value="005">005</option><option value="006">006</option></select></label>
''' + '\n'.join(cards) + '''<p><small>Демонстрация является проверкой визуального поведения системы и не заменяет независимую экспертную разметку. Для обсуждения конкретного участка укажите номер на странице.</small></p></main>
<script>document.getElementById('photo').addEventListener('change',e=>{const v=e.target.value;document.querySelectorAll('.scene').forEach(x=>x.hidden=(v!=='all'&&x.dataset.photo!==v))})</script>
</body></html>'''


def build_gallery(visualmetro:Path,tiles:Path,models_path:Path,cal_path:Path,
                  out_dir:Path,*,max_per_photo:int|None=None):
    dest=_safe_directory(out_dir)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Private H50Z gallery output directory must be empty')
    readiness=preflight(visualmetro,tiles)
    if not readiness['ready_for_private_TRAIN']:
        raise RuntimeError('Private H50Z gallery requires authenticated data: '+','.join(readiness['blockers']))
    if not models_path.is_file() or not cal_path.is_file():
        raise FileNotFoundError('Frozen S85 PRIVATE TRAIN and CAL required for visualization')
    train=json.loads(models_path.read_text())
    cal=json.loads(cal_path.read_text())
    deps,templates,photos,frozen,ctx=s85._validate_private(visualmetro,tiles)
    s85._model_receipt(train,ctx)
    if (cal.get('phase')!='cal' or cal.get('smoke_NOT_CONFIRMATORY') or
        cal.get('frozen_TRAIN_models_sha256')!=s85.digest(models_path)):
        raise ValueError('Unfrozen source CAL / TRAIN model drift')
    for key in ('schema','locks','source_split_sha256','s72_source_code_locks'):
        if cal.get(key)!=train.get(key):raise ValueError('Frozen S85 model/source changed')
    scorer_factory=deps[4]
    items=[];counter=0;count_by_arm={a:dict(lines=0,extras=0,joins=0) for a in ARMS}
    (dest/'images').mkdir(parents=True,exist_ok=False)
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        direction=f'{source}_to_{photo}'
        scorer,cutoff=scorer_factory._fit_source(tiles,source,templates)
        old_policy=frozen['directions'][direction]
        if float(cutoff)!=float(old_policy['S56_frozen_CAL_cutoff']):
            raise ValueError('Source S56 scorer changed after original CAL')
        current=cal['directions'][direction]
        model=train['models'][direction]['model']
        if (current['original_graph_arm']!=old_policy['CAL_selected'] or
            current['train_model_sha256']!=model_sha256(model) or
            float(current['frozen_source_cutoff'])!=float(cutoff)):
            raise ValueError('S85 graph policy or private model differs from source-CAL')
        selection=current['CAL_decision']['CAL_selected']
        if selection not in ARMS:raise ValueError('Unknown CAL-selected algorithm')
        paths=photos[photo] if max_per_photo is None else photos[photo][:max_per_photo]
        for tile in paths:
            raw=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape!=(512,512):
                raise ValueError('Genuine 512px private H50Z crop required')
            image=raw[160:352,160:352].copy()
            preds=s85.predict_blind(image,scorer,cutoff,old_policy['CAL_selected'],model)
            base_count=len(preds['baseline']['polylines'])
            variants=[]
            for arm,display in (('baseline','S72: исходный'),
                                ('s85_learned_p85','S85: порог 0.85'),
                                ('s85_learned_p95','S85: порог 0.95')):
                p=preds[arm]
                counter+=1
                filename=f'proposal_{counter:04d}.jpg'
                output=render_overlay(image,p,title=display,source_count=base_count)
                if not cv2.imwrite(str(dest/'images'/filename),output,[cv2.IMWRITE_JPEG_QUALITY,92]):
                    raise OSError('Failed to create PRIVATE preview image')
                count_by_arm[arm]['lines']+=len(p['polylines'])
                count_by_arm[arm]['extras']+=int(p['added_pixel_instances'])
                count_by_arm[arm]['joins']+=len(p['chosen'])
                variants.append({'image':'images/'+filename,'display':display,
                                 'lines':len(p['polylines']),'extra':int(p['added_pixel_instances']),
                                 'joins':len(p['chosen']),'CAL_selected':arm==selection})
            items.append({'photo_group':photo,'variants':variants})
    (dest/'index.html').write_text(offline_html(items),encoding='utf8')
    report={'schema':'S85_PRIVATE_OFFLINE_39_MACHINE_UNKNOWN_GALLERY_V1',
            'experiment_id':s85.EXPERIMENT_ID,'genuine_H50Z_crops':len(items),
            'native_physics_GT_count':0,'all_machine_joins':'UNKNOWN',
            'physical_auto_joins_authorized':False,
            'model_sha256':s85.digest(models_path),'CAL_sha256':s85.digest(cal_path),
            'methods':count_by_arm,
            'CAL_selections':{k:v['CAL_decision']['CAL_selected'] for k,v in cal['directions'].items()},
            'contains_private_H50Z_rendered_pixels':True,'publish_on_public_GitHub':False}
    _write_once(dest/'PRIVATE_REVIEW_MANIFEST.json',report)
    (dest/'README_FIRST_RU.txt').write_text(
        'S85 PRIVATE H50Z machine review. Откройте index.html без интернета.\n'
        'НЕТ ЭКСПЕРТНОЙ GT. Голубое — оригинальные линии S72, фиолетовое — отобранные S85, зелёное — UNKNOWN связи.\n'
        'НЕ размещать изображения в публичном GitHub. Не делается 3D/импульс/тип частицы.\n',encoding='utf8')
    manifest={str(p.relative_to(dest)):sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
    _write_once(dest/'PRIVATE_SHA256_MANIFEST.json',manifest)
    return report


def main():
    a=argparse.ArgumentParser(description='PRIVATE offline HTML overlays: all native physics tracks UNKNOWN')
    a.add_argument('--visualmetro',type=Path,required=True)
    a.add_argument('--photo-tiles',type=Path,required=True)
    a.add_argument('--train-models',type=Path,required=True)
    a.add_argument('--calibration',type=Path,required=True)
    a.add_argument('--private-out-dir',type=Path,required=True)
    a.add_argument('--max-per-photo',type=int,default=None,help='PRIVATE smoke only; default all 39')
    p=a.parse_args()
    if p.max_per_photo is not None and p.max_per_photo<1:
        raise ValueError('Need at least one private crop per photo')
    result=build_gallery(p.visualmetro,p.photo_tiles,p.train_models,p.calibration,
                         p.private_out_dir,max_per_photo=p.max_per_photo)
    print(json.dumps({'preview_tiles':result['genuine_H50Z_crops'],'GT':'UNKNOWN',
           'private_review_index':str(p.private_out_dir/'index.html')},ensure_ascii=False))
if __name__=='__main__':main()
