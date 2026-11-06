"""S96 real-image junction review: independently label crossed/bundled arms.

Blinded reviewer sees original film next to source-observed arm endpoints,
NOT machine-predicted pairing. Four-arm choices encode all 3 perfect pairings
plus NONE/UNDECIDABLE. Expert labels are 2D visual hypotheses only; no source
mask is edited and no common physical particle/interaction is asserted.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .s92_curve_review import _write_private,_json

SCHEMA="S96_BLINDED_FOUR_ARM_VISUAL_REVIEW_V1"
STRUCTURES=("CROSSING","THREE_ARM_BRANCH","BUNDLE_OR_OVERLAP",
            "CALIBRATION_MARK_OR_ARTIFACT","UNDECIDABLE")
PAIRINGS=("AB|CD","AC|BD","AD|BC","NONE","UNDECIDABLE")


def _photo_case_id(source_sha,junction_id):
    return "j_"+sha256((source_sha+":"+junction_id).encode()).hexdigest()[:20]


def label_ports(junction):
    """Use camera polar order to give local independent A/B/C/D labels."""
    center=np.asarray(junction["center_xy_source"],float)
    ports=junction["ports"]
    order=sorted(ports,key=lambda p:(
        float(np.arctan2(p["point_xy"][1]-center[1],p["point_xy"][0]-center[0])),
        p["component_id"],p["port"]
    ))
    if len(ports)>26:
        raise ValueError("Too many displayed junction ports")
    return [{
        "label":chr(ord("A")+i),
        "component_id":p["component_id"],"port":p["port"],
        "source_endpoint_xy":p["point_xy"],
    } for i,p in enumerate(order)]


def pair_code_for_machine(junction,labels):
    if len(labels)!=4 or len(junction["provisional_pairs"])!=2:
        return None
    lookup={f'{p["component_id"]}:{p["port"]}':p["label"] for p in labels}
    groups=[]
    for edge in junction["provisional_pairs"]:
        a=lookup.get(edge["port_a"])
        b=lookup.get(edge["port_b"])
        if not a or not b:raise ValueError("Machine edge refers to unavailable port")
        groups.append("".join(sorted((a,b))))
    code="|".join(sorted(groups))
    if code not in PAIRINGS:
        raise ValueError("Invalid machine visual crossing pairing")
    return code


def _marked_junction_photo(bmp,center_xy,labels,crop_size):
    gray,rect=bmp.crop(tuple(center_xy),size=crop_size)
    left=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    right=left.copy()
    x0,y0,_,_=rect.original_xyxy
    center=np.rint(np.asarray(center_xy,float)-[x0,y0]).astype(int)
    cv2.circle(right,tuple(center),9,(160,50,255),1,lineType=cv2.LINE_AA)
    for p in labels:
        q=np.rint(np.asarray(p["source_endpoint_xy"],float)-[x0,y0]).astype(int)
        pos=tuple(q)
        cv2.circle(right,pos,5,(50,230,250),1,lineType=cv2.LINE_AA)
        cv2.putText(right,p["label"],(q[0]+7,q[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX,.62,(20,235,250),2,lineType=cv2.LINE_AA)
    # The machine grouping is NOT drawn here: avoid biasing the expert.
    return np.hstack((left,right)),list(rect.original_xyxy)


def _js(payload):
    return json.dumps(payload,ensure_ascii=False,separators=(",",":")).replace(
        "<","\\u003c").replace(">","\\u003e").replace("&","\\u0026"
    )


def _viewer(pack):
    rows=[{
        "case_id":x["case_id"],
        "ports":x["display_ports"],
        "file":x.get("relative_original_vs_arms_image"),
        "n":x["observed_arm_count"],
    } for x in pack["cases"]]
    meta={k:pack[k] for k in (
        "schema","source_bmp_sha256","source_s96_report_sha256",
        "source_image_available","sampled_cases",
    )}
    # Vanilla offline HTML: source pixels stay local and no network requests.
    source=r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>S96: blind 2D crossing and branch review</title>
<style>body{font:15px system-ui,sans-serif;margin:22px;background:#12181f;color:#ebf0f6}
img{max-width:100%;height:auto;border:1px solid #506173}
button,select,textarea{background:#28374a;color:#f6f8fb;border:1px solid #8492a6;
padding:8px;margin:6px;font:14px system-ui;border-radius:5px}
textarea{width:90%;height:60px}h2{font-size:20px}
label{display:block;margin:8px 0}p{max-width:1100px}</style>
<h2>S96: независимая проверка пересечения / пучка / разветвления</h2>
<p>СЛЕВА: неизменённый исходный участок. СПРАВА: те же пиксели, отмечены
координаты концов фрагментов A/B/C/D и центр области. Машинная
пара концов скрыта. Человек оценивает только видимую геометрию 2D:
варианты AB|CD, AC|BD и AD|BC означают две пары предполагаемых
продолжений. Это НЕ подтверждённые физические траектории.</p>
<p id="warning"></p><p id="summary"></p><div id="scene"></div>
<label>Структура
<select id="type">
<option value="UNDECIDABLE">Неясно</option>
<option value="CROSSING">Пересечение двух линий</option>
<option value="THREE_ARM_BRANCH">Трёхлучевой узел</option>
<option value="BUNDLE_OR_OVERLAP">Пучок / наложение</option>
<option value="CALIBRATION_MARK_OR_ARTIFACT">Репер / артефакт</option>
</select></label>
<label>Визуальное продолжение (только если A/B/C/D)
<select id="pair">
<option value="UNDECIDABLE">Неясно</option>
<option value="AB|CD">AB и CD</option>
<option value="AC|BD">AC и BD</option>
<option value="AD|BC">AD и BC</option>
<option value="NONE">Ни одна пара не подтверждается изображением</option>
</select></label>
<label>Примечание <textarea id="note" maxlength="600"></textarea></label>
<button id="previous">Назад</button>
<button id="save">Сохранить и далее</button>
<button id="next">Далее</button>
<button id="download">Скачать разметку JSON</button>
<label>Восстановить прежнюю сессию <input id="upload" type="file" accept=".json"></label>
<script>
'use strict';
const CASES=__CASES__,META=__META__,STRUCT=__STRUCT__,PAIRS=__PAIRS__;
const map=new Map();let current=0;
const el=x=>document.getElementById(x);
function display(){
 const box=el('scene');box.replaceChildren();
 const c=CASES[current];
 if(!c){box.textContent='Нет найденных узлов для экспертной проверки';return;}
 const title=document.createElement('h3');
 title.textContent=String(current+1)+' / '+CASES.length+' ('+c.case_id+')';
 box.appendChild(title);
 if(c.file){
   const pic=document.createElement('img');pic.src=c.file;
   pic.alt='Original versus labeled fragment endpoint positions; no suggested pairing';
   box.appendChild(pic);
 }else{
   const msg=document.createElement('p');msg.textContent='Нет исходного изображения: недопустима визуальная разметка';
   box.appendChild(msg);
 }
 const old=map.get(c.case_id)||{structure:'UNDECIDABLE',pairing:'UNDECIDABLE',note:''};
 el('type').value=old.structure;
 el('pair').value=old.pairing;
 el('pair').disabled=c.n!==4;
 el('note').value=old.note;
 el('summary').textContent='Решений '+map.size+' / '+CASES.length;
}
el('warning').textContent=META.source_image_available?
 'Машинные решения скрыты. Все ответы остаются локальными до скачивания JSON.':
 'Исходное фото НЕ предоставлено. Экспертные ответы отключены.';
el('previous').onclick=()=>{if(current>0)current--;display();};
el('next').onclick=()=>{if(current<CASES.length-1)current++;display();};
el('save').onclick=()=>{
 if(!META.source_image_available){alert('Нужен исходный BMP для подтверждения');return;}
 const c=CASES[current];if(!c)return;
 const pairing=c.n===4?el('pair').value:'UNDECIDABLE';
 map.set(c.case_id,{case_id:c.case_id,structure:el('type').value,
   pairing,note:el('note').value.slice(0,600)});
 if(current<CASES.length-1)current++;display();
};
el('download').onclick=()=>{
 const out={...META,decisions:[...map.values()]};
 const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
 const uri=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=uri;
 a.download='S96_human_junction_review_PRIVATE.json';
 document.body.append(a);a.click();a.remove();URL.revokeObjectURL(uri);
};
el('upload').onchange=async event=>{
 try{
   const data=JSON.parse(await event.target.files[0].text());
   for(const key of ['schema','source_bmp_sha256','source_s96_report_sha256','sampled_cases']){
    if(data[key]!==META[key])throw Error('Это другая фотография/версия S96');
   }
   if(!Array.isArray(data.decisions))throw Error('Неверный JSON');
   const lookup=new Map(CASES.map(x=>[x.case_id,x]));
   const loaded=new Map();
   for(const d of data.decisions){
    const c=lookup.get(d.case_id);
    if(!c||loaded.has(d.case_id)||!STRUCT.includes(d.structure)
       ||!PAIRS.includes(d.pairing)||typeof d.note!=='string'||d.note.length>600
       ||(c.n!==4&&d.pairing!=='UNDECIDABLE'))
      throw Error('Неизвестный объект, повтор или недопустимый класс');
    loaded.set(d.case_id,d);
   }
   map.clear();for(const [k,v] of loaded)map.set(k,v);display();
 }catch(e){alert(String(e.message||e));}
};
display();
</script></html>"""
    return (source.replace("__CASES__",_js(rows))
                  .replace("__META__",_js(meta))
                  .replace("__STRUCT__",_js(STRUCTURES))
                  .replace("__PAIRS__",_js(PAIRINGS)))


def prepare_blind_junction_review(*,junctions,report,source_sha,
                                  source_report_sha,bmp_path,out_dir,
                                  max_cases:int=32,crop_size:int=320):
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if out_dir.exists() or out_dir==repo or repo in out_dir.parents:
        raise ValueError("S96 blind review needs new PRIVATE directory")
    if not isinstance(max_cases,int) or not 1<=max_cases<=100:
        raise ValueError("Invalid S96 review case budget")
    if not isinstance(crop_size,int) or not 160<=crop_size<=512 or crop_size%16:
        raise ValueError("Invalid S96 review crop size")
    if source_sha!=report.get("source_bmp_sha256"):
        raise ValueError("S96 review/source BMP provenance mismatch")
    bmp=None
    if bmp_path is not None:
        bmp_path=Path(bmp_path).resolve()
        if sha256_file(bmp_path)!=source_sha:
            raise ValueError("Wrong original photograph for S96 review")
        bmp=GrayBMP8(bmp_path)
        if [bmp.height,bmp.width]!=report["image_shape_yx"]:
            raise ValueError("Wrong original photograph dimensions for S96 review")
    # Source-hash order prevents choosing only visually impressive crossings.
    picked=sorted(junctions,key=lambda x:sha256(
        (source_sha+":"+x["junction_id"]).encode()).hexdigest())[:max_cases]
    cases=[]
    out_dir.mkdir(mode=0o700,parents=True)
    out_dir.chmod(0o700)
    if bmp:
        (out_dir/"crops").mkdir(mode=0o700)
    for i,j in enumerate(picked):
        labels=label_ports(j)
        case={
            "case_id":_photo_case_id(source_sha,j["junction_id"]),
            "junction_id":j["junction_id"],
            "observed_arm_count":len(labels),
            "display_ports":labels,
            "machine_junction_status_HIDDEN_FROM_UI":j["junction_status"],
            "machine_pairing_HIDDEN_FROM_UI":pair_code_for_machine(j,labels),
        }
        if bmp:
            frame,xyxy=_marked_junction_photo(
                bmp,j["center_xy_source"],labels,crop_size
            )
            rel=f"crops/{i:03d}_{case['case_id']}.jpg"
            ok,buffer=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,92])
            if not ok:raise RuntimeError("Junction crop encoding failed")
            _write_private(out_dir/rel,buffer.tobytes())
            case["relative_original_vs_arms_image"]=rel
            case["crop_xyxy_source"]=xyxy
            case["crop_sha256"]=sha256_file(out_dir/rel)
        cases.append(case)
    pack={
        "schema":SCHEMA,"status":"SOURCE_LOCKED_BLINDED_2D_JUNCTION_REVIEW",
        "source_bmp_sha256":source_sha,
        "source_s96_report_sha256":source_report_sha,
        "source_image_available":bmp is not None,
        "all_machine_junctions":len(junctions),
        "sampled_cases":len(cases),
        "sampling":"DETERMINISTIC_SOURCE_HASH; no operator labels involved",
        "cases":cases,
        "physical_trajectory_identity_verified":False,
    }
    if bmp_path is not None and sha256_file(bmp_path)!=source_sha:
        raise ValueError("BMP changed during S96 review generation")
    _write_private(out_dir/"review_pack.json",_json(pack))
    _write_private(out_dir/"index.html",_viewer(pack).encode("utf8"))
    return pack


def compile_junction_review(pack,submitted):
    if (not isinstance(pack,dict) or not isinstance(submitted,dict)
        or pack.get("schema")!=SCHEMA or submitted.get("schema")!=SCHEMA):
        raise ValueError("Invalid S96 blind review schema")
    for key in ("source_bmp_sha256","source_s96_report_sha256",
                "sampled_cases","source_image_available"):
        if submitted.get(key)!=pack.get(key):
            raise ValueError("S96 human review references another source/variant")
    values=submitted.get("decisions")
    if not isinstance(values,list):
        raise ValueError("S96 human reviewer decisions missing")
    if not pack["source_image_available"] and values:
        raise ValueError("No original photo: cannot accept expert visual junction labels")
    cases={x["case_id"]:x for x in pack["cases"]}
    if len(cases)!=len(pack["cases"]):
        raise ValueError("Duplicate S96 visual cases")
    decisions=[]
    seen=set()
    for v in values:
        if not isinstance(v,dict):
            raise ValueError("Bad S96 human review record")
        cid=v.get("case_id")
        if cid not in cases or cid in seen:
            raise ValueError("Unknown or duplicated S96 case ID")
        seen.add(cid)
        structure=v.get("structure")
        pairing=v.get("pairing")
        note=v.get("note","")
        if (structure not in STRUCTURES or pairing not in PAIRINGS
            or not isinstance(note,str) or len(note)>600
            or (cases[cid]["observed_arm_count"]!=4 and pairing!="UNDECIDABLE")):
            raise ValueError("Invalid S96 human visual category or pairing")
        machine=cases[cid]["machine_pairing_HIDDEN_FROM_UI"]
        decisions.append({
            "case_id":cid,
            "human_structure":structure,
            "human_four_arm_pairing":pairing,
            "machine_pairing":machine,
            "human_machine_visual_pairing_agreement":(
                bool(pairing==machine) if pairing!="UNDECIDABLE"
                and machine is not None else None
            ),
            "note":note,
            "same_particle_physics_verified":False,
        })
    decided=[x for x in decisions if x["human_machine_visual_pairing_agreement"] is not None]
    return {
        "schema":SCHEMA,"status":"SOURCE_LOCKED_2D_VISUAL_JUNCTION_REVIEW_COMPILED",
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s96_report_sha256":pack["source_s96_report_sha256"],
        "sampled_cases":len(pack["cases"]),
        "reviewed_cases":len(decisions),
        "unreviewed_cases_unknown":len(pack["cases"])-len(decisions),
        "human_structure_counts":dict(Counter(x["human_structure"] for x in decisions)),
        "reviewed_machine_proposed_decidable_four_arm_cases":len(decided),
        "machine_human_visual_pairing_agreement_count":sum(
            x["human_machine_visual_pairing_agreement"] is True for x in decided
        ),
        "machine_human_visual_pairing_agreement_rate":(
            sum(x["human_machine_visual_pairing_agreement"] is True for x in decided)
            /len(decided) if decided else None
        ),
        "not_physical_track_or_collision_accuracy":True,
        "no_verified_native_real_H50Z_GT":True,
        "never_convert_unreviewed_to_negative":True,
        "decisions":decisions,
    }
