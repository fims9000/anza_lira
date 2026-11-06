"""S95: offline expert review of all machine-suggested object classes.

Separate human visual status (track/artifact/unknown) from morphology
(crossing, fiducial X, bundle, coil...). No physical particle species/ID
or automatically approved same-track join is inferred.
"""
from __future__ import annotations
from collections import Counter
import json
import numpy as np

SCHEMA="S95_PRIVATE_HUMAN_OBJECT_CLASS_REVIEW_V1"
STATUSES=("TRACK_EVIDENCE","ARTIFACT_EVIDENCE","UNDECIDABLE")
MORPHOLOGIES=("REGULAR_LINE","CALIBRATION_CROSS","CROSSING",
              "BUNDLE_OR_OVERLAP","COILING_OR_SPIRAL","OTHER","UNKNOWN")


def _js(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":")).replace(
        "<","\\u003c").replace(">","\\u003e").replace("&","\\u0026"
    )


def build_review_html(*,source_report,source_report_sha256,
                      objects,components,image_shape_yx,small_shape_yx,
                      source_image_available:bool):
    if len(objects)!=len(components) or len(objects)>500000:
        raise ValueError("S95 review requires all source components")
    h,w=map(int,image_shape_yx)
    small_h,small_w=map(int,small_shape_yx)
    if min(h,w,small_h,small_w)<=0:
        raise ValueError("Invalid S95 source/display geometry")
    data=[]
    for item,comp in zip(objects,components):
        if item["component_id"]!=comp["component_id"]:
            raise ValueError("S95 review source ID mismatch")
        xy=np.asarray(comp["points_xy"],float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():
            raise ValueError("Bad S95 observed polyline")
        points=xy[np.linspace(0,len(xy)-1,min(100,len(xy))).astype(int)]
        points=np.round(points*np.array([small_w/w,small_h/h]),2)
        data.append({
            "id":item["component_id"],
            "machine":item["primary_review_category"],
            "tags":item["all_morphology_tags"],
            "points":points.tolist(),
        })
    meta={
        "schema":SCHEMA,
        "source_bmp_sha256":source_report["source_bmp_sha256"],
        "source_s95_report_sha256":source_report_sha256,
        "object_count":len(data),
    }
    doc=r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>S95 review — operator classes</title>
<style>
body{font:15px system-ui,sans-serif;background:#11161d;color:#eef3f7;margin:18px}
h1{font-size:21px}p{color:#cbd7e4;max-width:1060px}
main{display:grid;grid-template-columns:minmax(420px,3fr) minmax(290px,1fr);gap:16px}
svg{width:100%;height:auto;background:#293038;border:1px solid #667584}
.line{stroke:#48d3f1;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;fill:none;cursor:pointer}
.line.selected{stroke:#ffda66;stroke-width:5}
aside{border:1px solid #546673;padding:14px;border-radius:7px}
select,input,textarea,button{font:14px system-ui;color:#f0f5fa;background:#283544;
border:1px solid #667;padding:7px;border-radius:4px;margin-top:7px;max-width:100%}
textarea{width:94%;height:80px}button{cursor:pointer}label{display:block;margin-top:9px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style>
<h1>S95: все наблюдаемые линии, отдельно оцениваем класс и форму</h1>
<p>Это машинные гипотезы, а не доказанные типы частиц. Выберите фрагмент на снимке
или в списке и отдельно укажите: трек, артефакт или неясно; обычная линия,
калибровочный крест, пересечение, пучок, завиток или иное. Сохраните собственный
JSON для повторного открытия. Неразмеченное остаётся UNKNOWN, не становится фоном.
Нельзя заключать по 2D завитку, что причиной было столкновение нейтрона.</p>
<main><div>
<label>Страница кандидатов <select id="page"></select></label>
<label>ID <select id="item"></select></label>
<svg id="scene" viewBox="0 0 __WIDTH__ __HEIGHT__" xmlns="http://www.w3.org/2000/svg">
__BACKGROUND__<g id="lines"></g></svg>
</div><aside>
<h2>Решение эксперта</h2><div id="selected">Не выбран объект</div>
<label>Тип сигнала <select id="status"></select></label>
<label>Форма / структура <select id="morph"></select></label>
<label>Комментарий <textarea id="note" maxlength="800"></textarea></label>
<button id="save">Сохранить</button>
<button id="export">Скачать проверенные категории в JSON</button>
<label>Загрузить прежний файл <input id="import" type="file" accept=".json"></label>
<p id="progress"></p>
<p>Исходные пиксели неизменны. Ручной тип нельзя без проверки
использовать как доказанную физическую связь или причину взаимодействия.</p>
</aside></main>
<script>
'use strict';
const META=__META__;
const DATA=__DATA__;
const ST=__STATUS__;
const MOR=__MORPH__;
const PAGE=250;
const index=new Map(DATA.map(x=>[x.id,x]));
const edits=new Map();
let selected=null;
const byId=id=>document.getElementById(id);
const SVG='http://www.w3.org/2000/svg';
for(const [name,options] of [['status',ST],['morph',MOR]]){
 for(const value of options){
  const item=document.createElement('option');
  item.value=value;item.textContent=value;byId(name).appendChild(item);
 }
}
for(let i=0;i<Math.ceil(DATA.length/PAGE);i++){
 const opt=document.createElement('option');opt.value=String(i);
 opt.textContent=String(i+1)+' / '+String(Math.ceil(DATA.length/PAGE));
 byId('page').appendChild(opt);
}
function renderSelected(){
 const item=index.get(selected);
 if(!item){byId('selected').textContent='Нет кандидата';return;}
 const rev=edits.get(selected)||{status:'UNDECIDABLE',morphology:'UNKNOWN',note:''};
 byId('selected').textContent=item.id+'; machine: '+item.machine+'; '+item.tags.join(', ');
 byId('status').value=rev.status;byId('morph').value=rev.morphology;
 byId('note').value=rev.note;byId('item').value=selected;
}
function refresh(){
 const page=Number(byId('page').value)||0;
 const chunk=DATA.slice(page*PAGE,(page+1)*PAGE);
 const group=byId('lines');group.replaceChildren();
 const select=byId('item');select.replaceChildren();
 if(!selected||!chunk.some(x=>x.id===selected))selected=chunk[0]?.id||null;
 for(const item of chunk){
  const opt=document.createElement('option');opt.value=item.id;
  opt.textContent=item.id+' ('+item.machine+')';select.appendChild(opt);
  const path=document.createElementNS(SVG,'polyline');
  path.setAttribute('points',item.points.map(p=>p.join(',')).join(' '));
  path.setAttribute('class','line'+(item.id===selected?' selected':''));
  path.addEventListener('click',()=>{selected=item.id;refresh();});
  group.appendChild(path);
 }
 renderSelected();
 byId('progress').textContent='Проверено '+edits.size+' из '+DATA.length+' исходных кандидатов';
}
byId('page').addEventListener('change',()=>{selected=null;refresh();});
byId('item').addEventListener('change',()=>{selected=byId('item').value;refresh();});
byId('save').addEventListener('click',()=>{
 if(!selected)return;
 edits.set(selected,{component_id:selected,status:byId('status').value,
 morphology:byId('morph').value,note:byId('note').value.slice(0,800)});
 refresh();
});
byId('export').addEventListener('click',()=>{
 const obj={...META,decisions:[...edits.values()].sort((a,b)=>a.component_id.localeCompare(b.component_id))};
 const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob);const anchor=document.createElement('a');
 anchor.href=url;anchor.download='S95_human_review_PRIVATE.json';
 document.body.appendChild(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(url);
});
byId('import').addEventListener('change',async event=>{
 try{
  const payload=JSON.parse(await event.target.files[0].text());
  if(payload.schema!==META.schema||payload.source_bmp_sha256!==META.source_bmp_sha256
     ||payload.source_s95_report_sha256!==META.source_s95_report_sha256
     ||payload.object_count!==META.object_count||!Array.isArray(payload.decisions))
    throw Error('Выбран JSON от другой версии/фотографии');
  const next=new Map();
  for(const row of payload.decisions){
    if(!index.has(row.component_id)||next.has(row.component_id)
       ||!ST.includes(row.status)||!MOR.includes(row.morphology)
       ||typeof row.note!=='string'||row.note.length>800)
      throw Error('Неверный ID/тип или дублирующееся решение');
    next.set(row.component_id,row);
  }
  edits.clear();for(const [id,row] of next)edits.set(id,row);
  refresh();
 }catch(error){alert(String(error.message||error));}
});
refresh();
</script></html>"""
    background=(
        '<image href="S95_original_small.jpg" width="100%" height="100%"/>'
        if source_image_available else
        '<rect width="100%" height="100%" fill="#353c45"/>'
    )
    return (doc.replace("__WIDTH__",str(small_w))
               .replace("__HEIGHT__",str(small_h))
               .replace("__BACKGROUND__",background)
               .replace("__META__",_js(meta))
               .replace("__DATA__",_js(data))
               .replace("__STATUS__",_js(list(STATUSES)))
               .replace("__MORPH__",_js(list(MORPHOLOGIES))))


def compile_human_review(review,source_report,source_report_sha256,
                         source_objects):
    if not isinstance(review,dict) or review.get("schema")!=SCHEMA:
        raise ValueError("Unsupported human review schema")
    if (review.get("source_bmp_sha256")!=source_report["source_bmp_sha256"]
        or review.get("source_s95_report_sha256")!=source_report_sha256
        or review.get("object_count")!=len(source_objects)):
        raise ValueError("Human review belongs to a different S95 source/variant")
    byid={r["component_id"]:r for r in source_objects}
    if len(byid)!=len(source_objects):
        raise ValueError("Duplicate machine source IDs")
    values=review.get("decisions")
    if not isinstance(values,list):
        raise ValueError("Review decisions must be an array")
    seen=set();retained=[]
    for item in values:
        if not isinstance(item,dict):
            raise ValueError("Bad human decision")
        id=item.get("component_id")
        if id not in byid or id in seen:
            raise ValueError("Unknown or duplicated source component in human review")
        seen.add(id)
        status=item.get("status")
        morph=item.get("morphology")
        note=item.get("note","")
        if (status not in STATUSES or morph not in MORPHOLOGIES
            or not isinstance(note,str) or len(note)>800):
            raise ValueError("Invalid human status/morphology/note")
        retained.append({
            "component_id":id,
            "human_visual_status":status,
            "human_structure":morph,
            "human_note":note,
            "original_machine_category":byid[id]["primary_review_category"],
            "particle_species_identified":False,
            "same_particle_connection_confirmed":False,
        })
    return {
        "schema":SCHEMA,"status":"HUMAN_VISUAL_REVIEW_PROVENANCE_CHECKED",
        "source_bmp_sha256":review["source_bmp_sha256"],
        "source_s95_report_sha256":source_report_sha256,
        "all_source_objects":len(source_objects),
        "human_reviewed":len(retained),
        "unreviewed_remain_unknown":len(source_objects)-len(retained),
        "human_status_counts":dict(Counter(x["human_visual_status"] for x in retained)),
        "human_morphology_counts":dict(Counter(x["human_structure"] for x in retained)),
        "decisions":retained,
        "unreviewed_not_negative":True,
        "no_verified_particle_momentum_mass_or_neutron_cause":True,
    }
