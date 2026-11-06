"""S95 visual-gap audit: separate human continuity judgments from S92 ranking.

Source S92 gap proposals may contain ambiguous/rival/blocked links. Review
exactly those image-level *suggestions* with blinded local crops, then measure
only conditional visual correctness among EXPLICITLY reviewed proposals.

Never label absent/untested links as negative, claim full track recall, infer
neutron interactions, or authorize physical auto-joining.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .s92_curve_review import _write_private,_json

SCHEMA="S95_PRIVATE_VISUAL_GAP_REVIEW_V1"
REVIEW_CHOICES=("APPEARS_SAME_2D_TRACE","APPEARS_DIFFERENT_2D_TRACES","UNDECIDABLE")
REASON_CHOICES=("VISIBLE_PIECES_ALIGN","CROSSING_OR_BUNDLE",
                "MISSING_PIXEL_EVIDENCE","IMAGE_ARTIFACT","OTHER","UNKNOWN")


def _stratum(row):
    status=row["review_decision"]
    if status=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
        return "SELECTED_FOR_REVIEW"
    if status=="AMBIGUOUS_PARTNER_ABSTAIN":
        return "AMBIGUOUS_ALTERNATIVE"
    if status in ("THIRD_FRAGMENT_NEAR_GAP_ABSTAIN","KEYPOINT_REGION_ABSTAIN_NOT_A_TRACK_JOIN"):
        return "RISK_ABSTENTION"
    return "OTHER_REJECTED_CANDIDATE"


def select_visual_cases(proposals:list[dict],*,max_cases:int=48,seed:str="s95"):
    """Deterministic equal-stratum sampling, NEVER cherry-pick an expected GT."""
    if not isinstance(max_cases,int) or not 4<=max_cases<=128:
        raise ValueError("Must request 4..128 blinded 2D gap review cases")
    known=set()
    bygroup=defaultdict(list)
    for p in proposals:
        ident=p.get("proposal_id")
        if not isinstance(ident,str) or not ident or ident in known:
            raise ValueError("Missing or duplicate S92 gap proposal ID")
        known.add(ident)
        if p.get("status")!="UNKNOWN" or p.get("physical_auto_join") is not False:
            raise ValueError("Unsafe S92 gap identity/physical provenance")
        if not isinstance(p.get("review_decision"),str):
            raise ValueError("Missing S92 gap decision")
        bygroup[_stratum(p)].append(p)
    for key in bygroup:
        bygroup[key].sort(
            key=lambda row:sha256((seed+":"+row["proposal_id"]).encode()).hexdigest()
        )
    active=[g for g in ("SELECTED_FOR_REVIEW","AMBIGUOUS_ALTERNATIVE",
                        "RISK_ABSTENTION","OTHER_REJECTED_CANDIDATE")
            if bygroup.get(g)]
    selected=[]
    while len(selected)<max_cases and active:
        upcoming=[]
        for stratum in active:
            if bygroup[stratum] and len(selected)<max_cases:
                selected.append(bygroup[stratum].pop(0))
            if bygroup[stratum]:
                upcoming.append(stratum)
        active=upcoming
    return selected


def _safe_link_view(original,pa,pb,bridge,top_left):
    out=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
    offset=np.asarray(top_left,float)
    for pts,color in ((pa,(235,220,40)),(pb,(40,220,80))):
        loc=np.rint(np.asarray(pts,float)-offset).astype("int32")
        cv2.polylines(out,[loc.reshape(-1,1,2)],False,color,2,cv2.LINE_AA)
    loc=np.rint(np.asarray(bridge,float)-offset).astype("int32")
    for k in range(0,len(loc)-1,4):
        j=min(len(loc)-1,k+2)
        if j>k:
            cv2.line(out,tuple(loc[k]),tuple(loc[j]),(230,50,235),2,cv2.LINE_AA)
    return out


def build_visual_gap_pack(*,s95_dir:Path,s92_dir:Path,
                          components:list[dict],bmp_path:Path|None,
                          out_dir:Path,max_cases:int=48,crop_size:int=320):
    """Export source-locked, *blinded* paired original/overlay gap crops.

    The reviewer does not see machine ranking/stratum/score in the HTML UI.
    No-BMP packs exist for geometry debugging only; a human visual judgment
    CANNOT be accepted into accuracy summaries without source BMP crops.
    """
    s95_dir=Path(s95_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (out_dir==repo or repo in out_dir.parents or out_dir.exists()
        or out_dir==s95_dir or out_dir==s92_dir
        or s92_dir in out_dir.parents or s95_dir in out_dir.parents
        or out_dir in s92_dir.parents or out_dir in s95_dir.parents):
        raise ValueError("Gap review requires new PRIVATE destination outside sources")
    if not isinstance(crop_size,int) or not 160<=crop_size<=512 or crop_size%16:
        raise ValueError("Gap crop size must be 160..512, multiple of 16")
    source=s95_dir/"report.json"
    s95=json.loads(source.read_text(encoding="utf-8"))
    proposals_file=s92_dir/"gap_proposals_unknown.ndjson"
    raw_proposals=[json.loads(line) for line in proposals_file.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if s95.get("source_s92_report_sha256")!=sha256_file(s92_dir/"report.json"):
        raise ValueError("S95 and S92 reports do not share a frozen provenance")
    if not isinstance(components,list) or not components:
        raise ValueError("S95 gap review needs known original S86 observed geometries")
    paths={x["component_id"]:np.asarray(x["points_xy"],float) for x in components}
    if len(paths)!=len(components):
        raise ValueError("Duplicate observed component IDs")
    proposals=select_visual_cases(raw_proposals,max_cases=max_cases)
    for p in proposals:
        if p["component_a"] not in paths or p["component_b"] not in paths:
            raise ValueError("S92 proposal references missing S86 observed component")
    bmp=None
    bmp_sha=s95["source_bmp_sha256"]
    if bmp_path is not None:
        bmp_path=Path(bmp_path).resolve()
        if sha256_file(bmp_path)!=bmp_sha:
            raise ValueError("Original BMP differs from frozen S95 source SHA")
        bmp=GrayBMP8(bmp_path)
        if [bmp.height,bmp.width]!=s95["image_shape_yx"]:
            raise ValueError("S95 source dimensions mismatch")
    out_dir.mkdir(mode=0o700,parents=True)
    out_dir.chmod(0o700)
    cases=[]
    if bmp is not None:
        (out_dir/"cases").mkdir(mode=0o700)
    for n,p in enumerate(proposals):
        ident=p["proposal_id"]
        guide=np.asarray(p["inferred_bridge_points_xy"],float)
        if guide.shape[1:]!=(2,) or not np.isfinite(guide).all():
            raise ValueError("Nonfinite S92 inferred gap")
        case_id="gap_"+sha256((bmp_sha+":"+ident).encode()).hexdigest()[:20]
        sample={
            "case_id":case_id,
            "proposal_id":ident,
            "fragment_a":p["component_a"],
            "fragment_b":p["component_b"],
            "source_port_a":p["port_a"],
            "source_port_b":p["port_b"],
            "machine_stratum_HIDDEN_IN_BLIND_REVIEW":_stratum(p),
            "machine_decision_HIDDEN_IN_BLIND_REVIEW":p["review_decision"],
            "original_image_available":bmp is not None,
        }
        if bmp is not None:
            center=np.median(guide,axis=0)
            center=np.clip(center,[0,0],[bmp.width-1,bmp.height-1])
            crop,coords=bmp.crop(tuple(center.tolist()),size=crop_size)
            x0,y0,_,_=coords.original_xyxy
            overlay=_safe_link_view(crop,paths[p["component_a"]],
                                    paths[p["component_b"]],guide,[x0,y0])
            # S92 'gap' pixels are always just magenta dashes, not filled mask.
            paired=np.concatenate((cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR),
                                   overlay),axis=1)
            ok,encoded=cv2.imencode(".jpg",paired,[cv2.IMWRITE_JPEG_QUALITY,92])
            if not ok:
                raise RuntimeError("Failed to encode source-locked gap crop")
            relative=f"cases/{n:03d}_{case_id}.jpg"
            _write_private(out_dir/relative,encoded.tobytes())
            sample["relative_case_photo"]=relative
            sample["crop_xyxy_source"]=list(coords.original_xyxy)
            sample["case_photo_sha256"]=sha256_file(out_dir/relative)
        cases.append(sample)
    pack={
        "schema":SCHEMA,
        "status":"BLINDED_2D_APPEARANCE_REVIEW_PACK",
        "source_bmp_sha256":bmp_sha,
        "source_s95_report_sha256":sha256_file(source),
        "source_s92_gap_proposals_sha256":sha256_file(proposals_file),
        "image_shape_yx":s95["image_shape_yx"],
        "source_image_available":bmp is not None,
        "all_machine_gap_proposals":len(raw_proposals),
        "sampled_cases":len(cases),
        "selection_method":"HASHED_STRATIFIED_BALANCED_OVER_MACHINE_DECISION; NO_GT",
        "machine_stratum_full_counts":dict(Counter(_stratum(x) for x in raw_proposals)),
        "cases":cases,
        "human_track_identity_certified":False,
        "physics_auto_join_allowed":False,
    }
    if (sha256_file(source)!=pack["source_s95_report_sha256"]
        or sha256_file(proposals_file)!=pack["source_s92_gap_proposals_sha256"]):
        raise ValueError("Machine source changed during blind case generation")
    if bmp_path is not None and sha256_file(bmp_path)!=bmp_sha:
        raise ValueError("Source image changed during blind case generation")
    _write_private(out_dir/"review_pack.json",_json(pack))
    _write_private(out_dir/"index.html",_build_review_page(pack).encode("utf-8"))
    return pack


def _js(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":")).replace(
        "<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")


def _build_review_page(pack):
    if not pack["cases"]:
        return (
            "<!doctype html><html lang='ru'><meta charset='utf-8'>"
            "<title>No S92 gap proposals</title>"
            "<h1>Нет геометрических предложений S92 для проверки</h1>"
            "<p>Пустая выборка не доказывает отсутствие разрывов или "
            "ошибочных соединений на изображении.</p></html>"
        )
    client=[
        {"case_id":x["case_id"],
         "relative_case_photo":x.get("relative_case_photo")}
        for x in pack["cases"]
    ]
    metadata={
        "schema":SCHEMA,
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s95_report_sha256":pack["source_s95_report_sha256"],
        "source_s92_gap_proposals_sha256":pack["source_s92_gap_proposals_sha256"],
        "source_image_available":pack["source_image_available"],
    }
    source_ready=bool(pack["source_image_available"])
    doc=r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>S95: blind visual gap continuity review</title>
<style>body{font:15px system-ui;background:#11171f;color:#eaf1fa;margin:24px}
img{width:100%;max-width:1024px}button,select,textarea{font:15px system-ui;
margin:6px;padding:8px;background:#273445;color:#eaf1fa;border:1px solid #8391a1}
textarea{width:90%;height:70px}label{display:block;padding:5px}</style>
<h2>Проверка разрыва: одна видимая траектория или две?</h2>
<p>Слева — оригинальные пиксели, справа — та же область с двумя
наблюдёнными фрагментами и пунктиром возможного промежутка. Пунктир
не означает наблюдённую линию. Оценка только внешней 2D непрерывности:
одна частица, тип события и физическая связь здесь НЕ доказаны.</p>
<p id="warning"></p><p id="progress"></p><div id="frame"></div>
<label>Как выглядит связь
<select id="choice">
<option value="UNDECIDABLE">Неясно / не могу определить</option>
<option value="APPEARS_SAME_2D_TRACE">Похоже на одну непрерывную 2D линию</option>
<option value="APPEARS_DIFFERENT_2D_TRACES">Похоже на разные линии</option>
</select></label>
<label>Причина <select id="reason">
<option value="UNKNOWN">Неясно</option>
<option value="VISIBLE_PIECES_ALIGN">Продолжение выглядит согласованным</option>
<option value="CROSSING_OR_BUNDLE">Пересечение / пучок</option>
<option value="MISSING_PIXEL_EVIDENCE">Не хватает наблюдаемых пикселей</option>
<option value="IMAGE_ARTIFACT">Мусор / артефакт</option>
<option value="OTHER">Другое</option></select></label>
<label>Примечание <textarea id="note" maxlength="600"></textarea></label>
<button id="prev">Назад</button><button id="save">Сохранить и далее</button>
<button id="next">Далее</button><button id="export">Скачать приватный JSON</button>
<script>
'use strict';
const META=__META__,CASES=__CASES__,READY=__READY__;
const edits=new Map();let index=0;
const el=id=>document.getElementById(id);
function render(){
 const c=CASES[index];el('frame').replaceChildren();
 const title=document.createElement('h3');
 title.textContent=String(index+1)+' / '+CASES.length+' — '+c.case_id;
 el('frame').append(title);
 if(c.relative_case_photo){
  const img=document.createElement('img');img.src=c.relative_case_photo;
  img.alt='Настоящий 2D кадр, слева оригинал, справа две наблюдаемые линии и гипотетический пунктир';
  el('frame').append(img);
 }else{
  const msg=document.createElement('p');msg.textContent='НЕТ ОРИГИНАЛЬНОЙ ФОТОГРАФИИ: только геометрический программный аудит, не присваивать физическую разметку.';
  el('frame').append(msg);
 }
 const v=edits.get(c.case_id)||{choice:'UNDECIDABLE',reason:'UNKNOWN',note:''};
 el('choice').value=v.choice;el('reason').value=v.reason;el('note').value=v.note;
 el('progress').textContent='Сохранённых ответов '+edits.size+' / '+CASES.length;
}
el('warning').textContent=READY?'Рецензия на реальные наблюдаемые пиксели. Машинное решение скрыто.':
 'ФОТО НЕ ДОСТУПНО: можно сформировать отчёт, но визуальные экспертные решения без исходного BMP недопустимы.';
el('prev').onclick=()=>{if(index>0)index--;render();};
el('next').onclick=()=>{if(index+1<CASES.length)index++;render();};
el('save').onclick=()=>{
 if(!READY){alert('Без исходной фотографии экспертную оценку принимать нельзя');return;}
 const c=CASES[index],choice=el('choice').value,reason=el('reason').value;
 edits.set(c.case_id,{case_id:c.case_id,choice,reason,note:el('note').value.slice(0,600)});
 if(index+1<CASES.length)index++;render();
};
el('export').onclick=()=>{
 const file={...META,decisions:[...edits.values()]};
 const blob=new Blob([JSON.stringify(file,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),anchor=document.createElement('a');
 anchor.href=url;anchor.download='S95_gap_visual_review_PRIVATE.json';
 document.body.append(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(url);
};
render();
</script></html>"""
    return (doc.replace("__META__",_js(metadata)).replace("__CASES__",_js(client))
               .replace("__READY__","true" if source_ready else "false"))


def compile_visual_gap_review(pack:dict,review:dict)->dict:
    """Only reviewed S92 proposals enter denominators; abstain stays separate."""
    if (not isinstance(review,dict) or review.get("schema")!=SCHEMA
        or pack.get("schema")!=SCHEMA):
        raise ValueError("Missing/unsupported source-locked visual review schema")
    for key in ("source_bmp_sha256","source_s95_report_sha256",
                "source_s92_gap_proposals_sha256"):
        if review.get(key)!=pack.get(key):
            raise ValueError("Human gap review source SHA mismatch")
    if review.get("source_image_available") is not True or pack.get("source_image_available") is not True:
        if review.get("decisions"):
            raise ValueError("No original image: expert continuity judgments prohibited")
    allowed={x["case_id"]:x for x in pack["cases"]}
    if len(allowed)!=len(pack["cases"]):
        raise ValueError("Duplicated source gap cases")
    decisions=review.get("decisions")
    if not isinstance(decisions,list):
        raise ValueError("Human decisions missing")
    result=[];seen=set()
    for item in decisions:
        ident=item.get("case_id") if isinstance(item,dict) else None
        if ident not in allowed or ident in seen:
            raise ValueError("Unknown/duplicate gap case response")
        seen.add(ident)
        choice=item.get("choice")
        reason=item.get("reason")
        note=item.get("note","")
        if (choice not in REVIEW_CHOICES or reason not in REASON_CHOICES
            or not isinstance(note,str) or len(note)>600):
            raise ValueError("Invalid human visual continuity decision")
        case=allowed[ident]
        result.append({
            "case_id":ident,
            "proposal_id":case["proposal_id"],
            "machine_stratum":case["machine_stratum_HIDDEN_IN_BLIND_REVIEW"],
            "human_visual_choice":choice,"reason":reason,"note":note,
            "does_not_verify_same_physical_particle":True,
        })
    counts=Counter(x["human_visual_choice"] for x in result)
    accepted=[x for x in result if x["machine_stratum"]=="SELECTED_FOR_REVIEW"]
    rejected=[x for x in result if x["machine_stratum"]!="SELECTED_FOR_REVIEW"]
    decided=[x for x in accepted if x["human_visual_choice"]!="UNDECIDABLE"]
    known_diff=sum(x["human_visual_choice"]=="APPEARS_DIFFERENT_2D_TRACES" for x in decided)
    known_same=sum(x["human_visual_choice"]=="APPEARS_SAME_2D_TRACE" for x in decided)
    return {
        "schema":SCHEMA,
        "status":"SOURCE_LOCKED_BLINDED_VISUAL_APPEARANCE_ONLY",
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s95_report_sha256":pack["source_s95_report_sha256"],
        "source_s92_gap_proposals_sha256":pack["source_s92_gap_proposals_sha256"],
        "candidate_review_cases":len(pack["cases"]),
        "source_machine_proposals":pack["all_machine_gap_proposals"],
        "reviewed_cases":len(result),
        "unreviewed_cases_remain_unknown":len(pack["cases"])-len(result),
        "reviewed_visual_counts":dict(counts),
        "selected_machine_proposals_reviewed":len(accepted),
        "nonselected_machine_proposals_reviewed":len(rejected),
        "selected_decided_visual_count":len(decided),
        "selected_appears_same_2d_count":known_same,
        "selected_appears_different_2d_count":known_diff,
        "selected_apparent_false_join_fraction_among_decided":(
            known_diff/len(decided) if decided else None
        ),
        "sample_stratified_not_population_accuracy":True,
        "cannot_estimate_unproposed_true_link_recall":True,
        "cannot_certify_particle_identity_or_neutron_cause":True,
        "all_physical_joins_still_unapproved":True,
        "decisions":result,
    }
