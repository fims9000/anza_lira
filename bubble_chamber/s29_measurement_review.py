"""Display source-locked UNVERIFIED 2D path diagnostics in the S28 reviewer.

No automatic model suggestion becomes an expert annotation or accepted join.
"""
from __future__ import annotations
import json,re


def add_candidate_metrics_panel(s28_html:str,measures:dict)->str:
    if 'id="s28-choice"' not in s28_html or 'window.s28RenderGhost=function()' not in s28_html:
        raise ValueError('S28 human-only review workbench expected')
    if measures.get('schema')!='s29_source_locked_unverified_h50z_path_measures_v1' or measures.get('expert_confirmed_paths')!=0:
        raise ValueError('unverified source S29 measurements required')
    match=re.search(r'<script type="application/json" id="s28-payload">(.*?)</script>',s28_html,re.S)
    if match is None:raise ValueError('S28 source proof unavailable')
    ghost=json.loads(match.group(1))
    data={}
    for row in measures['rows']:
        cid=row['crop_id'];method=row['method'];rank=row['hypothesis_rank']
        if cid not in ghost or ghost[cid]['photo_sha256']!=row['source_photo_crop_sha256'] or not 1<=rank<=4:
            raise ValueError('S29 path/photo SHA mismatch')
        if row.get('expert_approved') is not False or row.get('physics_use')!='REVIEW_ONLY':
            raise ValueError('cannot silently assign model labels as expert')
        fam='az' if method=='original_AZ_assisted' else 'classic'
        key=fam+str(rank-1)
        data.setdefault(cid,{})[key]={
            'dark_profile_fraction':row['photometry']['fraction_with_dark_center'],
            'apparent_width_px':row['photometry']['median_apparent_width_px'],
            'candidate_bubble_peaks_unverified':row['photometry']['candidate_bubble_peaks_unverified'],
            'fit_rms_px':row['geometry']['rms_centerline_residual_px'],
            'original_start_xy':row['geometry']['source_start_xy'],
            'original_end_xy':row['geometry']['source_end_xy'],
            'quadratic_parametric_xy':{'x':row['geometry']['x_coefficients_c_b_a'],
                                    'y':row['geometry']['y_coefficients_c_b_a']},
        }
        if len(data[cid])>8:raise ValueError('duplicate candidate measurement')
    if len(data)!=9 or any(len(v)!=8 for v in data.values()):
        raise ValueError('expected all 9 annotated START crops x 8 candidate hypotheses')
    controls='''<section id="s29-metrics-panel" style="border:1px solid #8cad96;border-radius:9px;padding:9px;margin:8px 0">
<strong>S29 · геометрия 2D и фотометрия неподтверждённой гипотезы</strong>
<p id="s29-values" class="help">Выберите пунктирный вариант S28. Значения только из пикселей и гипотезы модели, не физический импульс или тип частицы.</p>
</section>'''
    s28_html=s28_html.replace('<div id="scene">',controls+'\n<div id="scene">',1)
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    script='''<script type="application/json" id="s29-payload">__PAYLOAD__</script>
<script>
'use strict';
const s29Stats=JSON.parse(document.getElementById('s29-payload').textContent);
window.s29RenderMeasurements=function(){
 const target=byId('s29-values');const rec=entries[active],choice=byId('s28-choice').value;
 const m=s29Stats[rec.crop_id]?.[choice];
 if(!m){target.textContent='Нет операторского start или вариант не выбран. Все неподтверждённые линии остаются неизвестными.';return;}
 const fmt=(x,n=2)=>x===null?'нет уверенного сигнала':Number(x).toFixed(n);
 const formatQuadratic=(coef)=>coef.map(x=>Number(x).toFixed(2)).join(' ; ');
 target.textContent='Начало: '+m.original_start_xy.map(x=>fmt(x,1)).join(',')+' px · конец: '+m.original_end_xy.map(x=>fmt(x,1)).join(',')+' px'+
   ' | RMS параболы: '+fmt(m.fit_rms_px)+' px | видимый тёмный центр: '+fmt(100*m.dark_profile_fraction,1)+'%'+
   ' | ширина (пиксели): '+fmt(m.apparent_width_px,1)+
   ' | пики тёмного сигнала (НЕ подтверждённые пузырьки): '+m.candidate_bubble_peaks_unverified+
   ' | x(u)=[c,b,a]: '+formatQuadratic(m.quadratic_parametric_xy.x)+
   ' | y(u)=[c,b,a]: '+formatQuadratic(m.quadratic_parametric_xy.y)+
   '. Значения НЕ подтверждают частицу/импульс.';
};
const oldS28Renderer=window.s28RenderGhost;
window.s28RenderGhost=function(){oldS28Renderer();window.s29RenderMeasurements();};
window.s28RenderGhost();
</script>'''
    return s28_html.replace('</body></html>',script.replace('__PAYLOAD__',payload)+'</body></html>')
