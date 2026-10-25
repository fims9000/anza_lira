"""Display genuine operator-START path hypotheses as GHOSTS in S27 reviewer.

No proposal is inserted into exported S26 expert JSON. Experts must draw a
manual stroke themselves; a model-selected curve never becomes human truth.
"""
from __future__ import annotations
import json,re


def add_path_ghosts(html:str,s28:dict)->str:
    if s28.get('schema')!='s28_genuine_h50z_operator_anchor_to_unverified_2D_path_v1':
        raise ValueError('trusted source SHA-locked S28 proposals needed')
    if 's27-queue-json' not in html or 'id="annotations"' not in html:
        raise ValueError('S27 review source/annotation controls missing')
    match=re.search(r'<script type="application/json" id="payload">(.*?)</script>',html,re.S)
    if match is None:raise ValueError('original image payload missing')
    source=json.loads(match.group(1));ids={p['crop_id']:p['sha256'] for p in source['photos']}
    if len(source['photos'])!=39:raise ValueError('must preserve all 39 H50Z review photos')
    payload={}
    for rec in s28['records']:
        crop=rec['crop_id']
        if crop not in ids or ids[crop]!=rec['source_crop_sha256'] or crop in payload:
            raise ValueError('S28 source photo SHA mismatch or duplicate proposal')
        for family in ('classical_candidate_paths','original_AZUNet_assisted_candidate_paths'):
            for path in rec[family]:
                if path.get('human_confirmed_track') is not False or path.get('automatic_physical_join_allowed') is not False:
                    raise ValueError('unsafe pre-certified S28 proposal')
                points=path['path_points_original_xy']
                if not points or any(len(p)!=2 or not all(isinstance(v,(int,float)) and 0<=v<512 for v in p) for p in points):
                    raise ValueError('unbounded ghost centerline')
        payload[crop]={'start_xy':rec['genuine_OPERATOR_start_xy_in_crop'],
                      'photo_sha256':ids[crop],
                      'geometry':[[p['path_points_original_xy'] for p in rec[fam]] for fam in ('classical_candidate_paths','original_AZUNet_assisted_candidate_paths')]}
    if len(payload)!=9:raise ValueError('need all 9 authentic START landmarks')
    controls='''<section id="s28-controls" style="border:1px dashed #70aee8;border-radius:9px;padding:10px;margin:8px 0">
<strong>S28 · варианты продолжения от реального start (НЕ разметка)</strong>
<p class="help">Пунктир — только подсказка, выбранная алгоритмом по фото и ANZA. Он НЕ добавляется к ручной разметке и НЕ подтверждает частицу. Для подтверждения нарисуйте линию вручную обычным инструментом «Трек».</p>
<label>Показать вариант <select id="s28-choice"><option value="none">Без пунктирной подсказки</option><option value="az0">ANZA · вариант 1</option><option value="az1">ANZA · вариант 2</option><option value="az2">ANZA · вариант 3</option><option value="az3">ANZA · вариант 4</option><option value="classic0">Фильтры · вариант 1</option><option value="classic1">Фильтры · вариант 2</option><option value="classic2">Фильтры · вариант 3</option><option value="classic3">Фильтры · вариант 4</option></select></label>
<span id="s28-context" class="help"></span>
</section>'''
    if '<div id="scene">' not in html or '<canvas id="annotations"' not in html:raise ValueError('legacy scene missing')
    html=html.replace('<div id="scene">',controls+'\n<div id="scene">',1)
    html=html.replace('<canvas id="annotations"', '<svg id="s28-ghost" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></svg>\n<canvas id="annotations"',1)
    canonical='function update(){byId(\'background\').src=entries[active].image_modes[byId(\'mode\').value];redraw();}'
    if canonical not in html:raise ValueError('different S26 UI, review switch cannot be verified')
    html=html.replace(canonical,canonical.replace('redraw();}','redraw();if(typeof window.s28RenderGhost===\'function\')window.s28RenderGhost();}'),1)
    safe=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    newscript='''<script type="application/json" id="s28-payload">___PAYLOAD___</script>
<script>
'use strict';
const s28GhostMap=JSON.parse(document.getElementById('s28-payload').textContent);
window.s28RenderGhost=function(){
 const svg=byId('s28-ghost'),select=byId('s28-choice'),context=byId('s28-context');svg.replaceChildren();
 const crop=entries[active],rec=s28GhostMap[crop.crop_id];
 if(!rec){context.textContent='В этом участке нет операторской точки start. Все пиксели остаются неизвестными.';return;}
 if(rec.photo_sha256!==crop.sha256)throw Error('S28 source SHA mismatch');
 const ns='http://www.w3.org/2000/svg';
 const c=document.createElementNS(ns,'circle');c.setAttribute('cx',rec.start_xy[0]);c.setAttribute('cy',rec.start_xy[1]);c.setAttribute('r',7);c.setAttribute('stroke','#ffd86a');c.setAttribute('fill','none');c.setAttribute('stroke-width','2');svg.append(c);
 const mode=select.value;if(mode==='none'){context.textContent='Жёлтый круг — настоящая точка start; подтверждённых контуров нет.';return;}
 let fam=mode.startsWith('az')?1:0,index=Number(mode.replace(/\\D/g,''));let pts=rec.geometry[fam][index];
 if(!pts){context.textContent='Такого варианта нет';return;}
 const line=document.createElementNS(ns,'polyline');line.setAttribute('points',pts.map(p=>p.join(',')).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#ffdf69');line.setAttribute('stroke-width','2');line.setAttribute('stroke-dasharray','6 5');line.setAttribute('stroke-linecap','round');svg.append(line);
 context.textContent='Вариант '+(index+1)+' · только видимая пунктирная гипотеза, НЕ экспертная метка.';
};
byId('s28-choice').onchange=()=>window.s28RenderGhost();
window.s28RenderGhost();
</script>'''
    return html.replace('</body></html>',newscript.replace('___PAYLOAD___',safe)+'</body></html>')
