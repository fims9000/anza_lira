"""Offline expert auditing UI. Exports source-scoped, reversible judgments.

This UI does not infer any physical particle ID itself and never modifies the
original H50Z raster or S109/S111 curve splines.
"""
from __future__ import annotations
import json,html


def make_review_html(roi, anchored,free,gaps,width=1200,height=1200,bridges=(),source_sha256=None,s111_sha256=None):
    catalog=[]
    for group,parts in [('anchored',anchored),('free',free)]:
        for z in parts:
            catalog.append({'id':z['path_id'],'kind':group,'curve_id':z.get('curve_id'),
                 'global_track_id':z.get('attached_to_s111_global_track_id'),
                 'bead_count':len(z['bubble_ids']),'length_px':z['length_px'],
                 'photo_support_fraction':z['mean_photo_mask_support'],
                 'score':z['score'],'status':z['status'],
                 'svg_path':z.get('inferred_spline',{}).get('svg_path') if z.get('inferred_spline') else None,
                 'polyline':z.get('preview_spline_xy'),'observed_bubbles':z['bubble_xy']})
    for z in gaps:
        catalog.append({'id':z['s111_edge_id'],'kind':'previous_S111_bridge',
                  'bead_count':z['midgap_photometric_dot_count'],
                  'length_px':z['gap_px'],'photo_support_fraction':z['midgap_S88_backed_dot_count']/max(1,z['midgap_photometric_dot_count']),
                  'status':z['review_priority_label'],'score':None,
                  'svg_path':None,'polyline':[], 'observed_bubbles':[],
                  'curve_ids':z['curve_ids']})
    for z in bridges:
        spline=z.get('candidate_inferred_spline') or {}
        catalog.append({'id':z['bridge_id'],'kind':'intergroup_bridge',
                   'global_track_id':' + '.join(z['source_groups']),
                   'bead_count':z['shared_dot_count'],
                   'length_px':0,'photo_support_fraction':z.get('photo_audit',{}).get('source_ridge_fraction',0),
                   'score':None,'status':z['status'],
                   'svg_path':spline.get('svg_path'),
                   'polyline':[],'observed_bubbles':z['source_dot_xy']})
    payload=json.dumps(catalog,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    # Relative paths keep the HTML usable after extracting the entire ZIP.
    source=f'../../s108_results_all/{roi}/06_SOURCE_ORIGINAL_GRAY.png'
    return '''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>S112 — проверка центров пузырьков и векторных треков</title>
<style>
:root{font-family:Arial,sans-serif;color:#1a2732;background:#f4f6f8}
main{display:grid;grid-template-columns:minmax(530px, min(68vw, 870px)) minmax(300px,1fr);gap:18px;align-items:start;padding:18px}
#stage{position:relative;max-width:100%;aspect-ratio:1;background:#777;border-radius:8px;overflow:hidden}
#stage img,#stage svg{position:absolute;inset:0;width:100%;height:100%}
#stage svg{pointer-events:none}#stage svg .clickable{pointer-events:stroke;cursor:pointer}
#stage svg .dots{pointer-events:none}
.panel{background:#fff;padding:16px;border-radius:10px;box-shadow:0 1px 6px #d3d6da}
small{color:#5a6975}button{padding:8px 10px;border:1px solid #c2cbd1;background:white;border-radius:6px;cursor:pointer}
button:hover{background:#eaf2fb}button.active{background:#183f65;color:white}
#items{height:385px;overflow:auto;border:1px solid #ddd;border-radius:5px;margin:8px 0}
#items button{display:block;width:100%;text-align:left;border:0;border-radius:0;border-bottom:1px solid #eee}
#items button.review-yes{border-left:5px solid #0f9864}#items button.review-no{border-left:5px solid #e25349}
section p{line-height:1.4}.legend{font-size:13px}
</style></head><body><main><div class="panel">
<h2>ANZA-LIRA S112 · '''+html.escape(roi)+'''</h2>
<p>Тёмный снимок — оригинальные пиксели. <b style="color:#d48700">Оранжевые</b> — продолжения старых кривых, <b style="color:#c62ad0">розовые</b> — неоднозначные,
<b style="color:green">зелёные</b> — новые независимые цепочки только на проверку. Линии пунктирные: это <strong>геометрические гипотезы</strong>, не сфотографированные пиксели. Точки на них — фотографические детекции, но пока не подтверждённые физические пузырьки.</p>
<div id="stage"><img alt="Исходный снимок H50Z" src="'''+source+'''"><svg id="overlay" viewBox="0 0 '''+str(width)+''' '''+str(height)+'''"></svg></div>
<p class="legend"><label><input id="toggleAnchored" type="checkbox" checked> Продолжения</label>
<label><input id="toggleFree" type="checkbox" checked> Новые цепочки</label>
<label><input id="toggleUncertain" type="checkbox" checked> Неоднозначные</label>
<label><input id="toggleBridges" type="checkbox" checked> Межгрупповые мосты</label>
<label><input id="toggleDots" type="checkbox" checked> Наблюдаемые точки выбранной цепочки</label></p>
<small>Не пытайтесь интерпретировать близость кривых на пересечении как доказанный трек: у каждого нового соединения нужна отдельная экспертиза.</small>
</div><aside class="panel"><h3>Экспертная проверка</h3><div id="counts"></div>
<input id="search" placeholder="Поиск ID или номера кривой" style="width:98%;padding:9px">
<div id="items"></div><div id="detail">Выберите кривую или соединение.</div>
<p><button id="confirm">Один физический трек</button><button id="reject">Разные / ложный трек</button>
<button id="unknown">Неясно</button></p>
<p><button id="export">Экспорт JSON решений</button><button id="import">Импорт JSON</button>
<input type="file" accept="application/json" id="file" hidden></p>
<small>Решения сохраняются только локально в экспортированном JSON. Автоматических физических ID этот просмотрщик не присваивает.</small>
</aside></main>
<script>
const ROI='''+json.dumps(roi)+''';const SOURCE_SHA='''+json.dumps(source_sha256)+''';const S111_SHA='''+json.dumps(s111_sha256)+''';const entries='''+payload+''';const NS='http://www.w3.org/2000/svg';
let chosen=null;const judgments={};const byid=Object.fromEntries(entries.map(e=>[e.id,e]));
function color(e){if(e.kind==='free')return'#24934d';if(e.kind==='intergroup_bridge')return e.status==='BEAD_SUPPORTED_INTERGROUP_BRIDGE_REVIEW_ONLY'?'#00a8d7':'#c04c5c';if(e.kind==='previous_S111_bridge')return'#888';return e.status.startsWith('AUTO_')?'#df8500':'#cf2bc1'}
function show(e){return (e.kind==='anchored'?document.getElementById('toggleAnchored').checked:e.kind==='free'?document.getElementById('toggleFree').checked:e.kind==='intergroup_bridge'?document.getElementById('toggleBridges').checked:false)&&
 (document.getElementById('toggleUncertain').checked||e.status.startsWith('AUTO_')||e.kind==='free')}
function render(){const svg=document.getElementById('overlay');svg.replaceChildren();
for(const e of entries){if(!show(e))continue;const path=document.createElementNS(NS,'path');
let d=e.svg_path;if(!d&&e.polyline?.length)d='M '+e.polyline.map((p,i)=>(i?'L ':'')+p[0]+','+p[1]).join(' ');
if(!d)continue;path.setAttribute('d',d);path.setAttribute('fill','none');path.setAttribute('stroke',color(e));
path.setAttribute('stroke-width',e.id===chosen?'4':'2');path.setAttribute('stroke-dasharray','7 5');
path.setAttribute('class','clickable');path.addEventListener('click',()=>select(e.id));svg.appendChild(path);
if(e.id===chosen&&document.getElementById('toggleDots').checked){for(const p of e.observed_bubbles){
const dot=document.createElementNS(NS,'circle');dot.setAttribute('cx',p[0]);dot.setAttribute('cy',p[1]);dot.setAttribute('r','3');
dot.setAttribute('fill','#00ccff');dot.setAttribute('class','dots');svg.appendChild(dot)}}}
}
function select(id){chosen=id;const e=byid[id];const judgment=judgments[id]||'uncertain';document.getElementById('detail').innerHTML=
'<h3>'+id+'</h3><p>'+e.kind+' · '+e.status+'</p><p>Пузырьков-кандидатов: '+e.bead_count+'; длина: '+e.length_px+' px; доля на S88: '+Number(e.photo_support_fraction).toFixed(2)+'</p><p>Группа S111: '+(e.global_track_id||'не установлена')+'</p><p>Ваше решение: '+judgment+'</p>';
render();renderList();}
function decision(label){if(!chosen)return;judgments[chosen]=label;select(chosen)}
function renderList(){const container=document.getElementById('items');container.replaceChildren();
const q=document.getElementById('search').value.toLowerCase();for(const e of entries.filter(x=>x.id.toLowerCase().includes(q)||(x.curve_id||'').toLowerCase().includes(q))){
const b=document.createElement('button');b.textContent=(e.kind==='free'?'Новый':'Фрагмент')+' · '+e.id+' · '+e.bead_count+' точек';
if(judgments[e.id]==='same')b.classList.add('review-yes');if(judgments[e.id]==='different')b.classList.add('review-no');
if(e.id===chosen)b.classList.add('active');b.onclick=()=>select(e.id);container.appendChild(b)}
const confirmed=Object.values(judgments).filter(x=>x==='same').length;const rejected=Object.values(judgments).filter(x=>x==='different').length;
document.getElementById('counts').textContent='Предложений: '+entries.length+'; подтверждено: '+confirmed+'; отклонено: '+rejected;}
document.getElementById('confirm').onclick=()=>decision('same');
document.getElementById('reject').onclick=()=>decision('different');
document.getElementById('unknown').onclick=()=>decision('uncertain');
for(const id of ['toggleAnchored','toggleFree','toggleUncertain','toggleBridges','toggleDots'])document.getElementById(id).onchange=render;
document.getElementById('search').oninput=renderList;
document.getElementById('export').onclick=()=>{const value=JSON.stringify({schema:'S112_EXPERT_BEAD_REVIEW_V1',source_roi:ROI,source_sha256:SOURCE_SHA,s111_global_tracks_sha256:S111_SHA,review_decisions:judgments},null,2);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([value],{type:'application/json'}));a.download=ROI+'_s112_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('import').onclick=()=>document.getElementById('file').click();
document.getElementById('file').onchange=async event=>{const file=event.target.files[0];if(!file)return;
const j=JSON.parse(await file.text());if(j.source_roi!==ROI||j.source_sha256!==SOURCE_SHA||j.s111_global_tracks_sha256!==S111_SHA){alert('Разметка другого источника или устаревшей версии S111');return}
for(const [id,v] of Object.entries(j.review_decisions||{}))if(byid[id]&&['same','different','uncertain'].includes(v))judgments[id]=v;
if(chosen)select(chosen);else renderList()};renderList();render();
</script></body></html>'''