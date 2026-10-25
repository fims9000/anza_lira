"""Source-locked, editable S31 machine-first review UI that preserves S29 human export.

Machine proposals, human local corrections, and expert certified ground truth
have different schemas. Browser data stays private until explicit JSON export.
"""
from __future__ import annotations
import json,re,hashlib


def embed_machine_drafts(html:str,draft:dict)->str:
    from .s31_machine_first import validate_draft_pack
    validate_draft_pack(draft)
    m=re.search(r'<script type="application/json" id="payload">(.*?)</script>',html,re.S)
    if not m:raise ValueError('Expected source S29 real photo payload')
    source=json.loads(m.group(1))
    src={r['crop_id']:r for r in source['photos']}
    if len(src)!=39 or set(src)!={r['crop_id'] for r in draft['photos']}:
        raise ValueError('Source images mismatch; refusing foreign photo proposals')
    for rec in draft['photos']:
        if src[rec['crop_id']]['sha256']!=rec['source_original_png_sha256_from_S29'] or src[rec['crop_id']]['photo']!=rec['photo_id']:
            raise ValueError('Source H50Z original crop SHA mismatch')
    if 'id="s28-ghost"' not in html or 'id="s29-values"' not in html or 'id="annotations"' not in html:
        raise ValueError('Expected full S29 review workbench')
    if 'id="s31-draft-payload"' in html:raise ValueError('Cannot insert machine draft twice')
    original="function update(){byId('background').src=entries[active].image_modes[byId('mode').value];redraw();if(typeof window.s28RenderGhost==='function')window.s28RenderGhost();}"
    if original not in html:raise ValueError('Source S29 update hook changed')
    html=html.replace(original,original.replace('window.s28RenderGhost();}','window.s28RenderGhost();if(typeof window.s31Render===\'function\')window.s31Render();}'),1)
    panel='''<section id="s31-panel" style="border:2px solid #e3ac77;border-radius:10px;padding:12px;margin:12px 0;background:#211e1d">
<h2 style="margin:2px 0 8px;font-size:1.15rem">S31 · машинная разметка: мы исправляем, физики проверят позже</h2>
<p class="help">Зелёная/голубая линия — <strong>черновик алгоритма, не подтверждённый трек</strong>. Не подписывайте эти данные как экспертные. Красные кандидаты могут быть реперами или царапинами. Все исходные 39 снимков сохранены; старый ручной экспорт S29 работает отдельно.</p>
<div class="tools">
<label>Кандидат<select id="s31-candidate" style="max-width:390px"></select></label>
<button id="s31-next">Следующий непроверенный</button>
<button id="s31-accept">Считаю треком (предварительно)</button>
<button id="s31-reject">Считаю артефактом (предварительно)</button>
<button id="s31-uncertain">Пока непонятно</button>
<button id="s31-edit">Редактировать узлы</button>
<button id="s31-clear">Сбросить решение</button>
</div>
<div class="tools"><label>Пара фрагментов (гипотеза соединения)<select id="s31-relation" style="max-width:390px"></select></label>
<button id="s31-rel-same">Предполагаю один трек</button><button id="s31-rel-diff">Предполагаю разные треки</button><button id="s31-rel-unsure">Не знаю</button><button id="s31-rel-clear">Сбросить пару</button></div>
<div class="tools"><button id="s31-export">Скачать машинную разметку + мои исправления JSON</button><label>Открыть исправления S31<input id="s31-import" type="file" accept=".json"></label></div>
<p class="help" id="s31-status">На снимке показаны неподтверждённые гипотезы.</p>
</section>'''
    html=html.replace('<div id="scene">',panel+'\n<div id="scene">',1)
    html=html.replace('<canvas id="annotations" width="512" height="512"></canvas>',
                      '<canvas id="annotations" width="512" height="512"></canvas>\n<svg id="s31-svg" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"></svg>',1)
    payload=json.dumps(draft,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    # The input machine draft file is static immutable provenance. All reviewer
    # operations live in a second separate JSON object and never touch S29.
    script=r'''<script type="application/json" id="s31-draft-payload">__DRAFT__</script>
<script>
'use strict';
const s31Pack=JSON.parse(document.getElementById('s31-draft-payload').textContent);
if(s31Pack.schema!=='s31_machine_first_h50z_draft_v1'||s31Pack.physical_joins_authorized!==false)throw Error('unsafe machine draft');
const s31Crops=new Map(s31Pack.photos.map(r=>[r.crop_id,r]));
const s31Drafts=new Map();
for(const r of s31Pack.photos){for(const p of r.drafts){if(s31Drafts.has(p.draft_id))throw Error('duplicate proposed draft');s31Drafts.set(p.draft_id,{crop:r.crop_id,proposal:p});}}
for(const r of s31Pack.photos){const source=entries.find(e=>e.crop_id===r.crop_id);if(!source||source.sha256!==r.source_original_png_sha256_from_S29)throw Error('wrong source pixel data');}
const s31Decisions={},s31RelationDecisions={};let s31Selected=null,s31RelationSelected=null,s31Edit=false,s31ActiveHandle=-1;
const s31Svg=byId('s31-svg');const ns31='http://www.w3.org/2000/svg';
function s31CurrentDraft(){return s31Selected?s31Drafts.get(s31Selected)?.proposal:null;}
function s31Saved(id){return s31Decisions[id]||null;}
function s31Polyline(p){return s31Saved(p.draft_id)?.edited_points_xy||p.points_xy;}
function s31SaveDecision(status,edit=null){if(!s31Selected)return;let last=s31Saved(s31Selected)||{};
 if(status==='RESET'){delete s31Decisions[s31Selected];}else{s31Decisions[s31Selected]={status,edited_points_xy:edit||last.edited_points_xy||null,review_origin:'LOCAL_USER_PROVISIONAL_NOT_PHYSICIST_CERTIFIED'};}
 window.s31Render();}
function s31Shape(tag,attrs){let e=document.createElementNS(ns31,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;}
function s31Stats(){let total=s31Drafts.size,nAccept=0,nReject=0,nUncertain=0,nEdit=0;
 for(const d of Object.values(s31Decisions)){if(d.status==='USER_ACCEPTED_PROVISIONAL')nAccept++;
 else if(d.status==='USER_REJECTED_PROVISIONAL')nReject++;else if(d.status==='USER_UNCERTAIN')nUncertain++;
 if(d.edited_points_xy)nEdit++;}
 return {total,nAccept,nReject,nUncertain,nEdit};}
function s31Render(){let rec=s31Crops.get(entries[active].crop_id),sel=byId('s31-candidate'),old=s31Selected;
 sel.replaceChildren();s31Svg.replaceChildren();
 let relSelect=byId('s31-relation'),oldRel=s31RelationSelected;relSelect.replaceChildren();
 for(const rel of (rec?.relation_drafts||[])){let opt=document.createElement('option');opt.value=rel.relation_id;opt.textContent=rel.candidate_a.slice(-6)+' / '+rel.candidate_b.slice(-6)+' | gap '+rel.endpoint_gap_px+'px';relSelect.append(opt);}
 s31RelationSelected=rec?.relation_drafts.some(r=>r.relation_id===oldRel)?oldRel:rec?.relation_drafts[0]?.relation_id||null;relSelect.value=s31RelationSelected||'';
 if(!rec){s31Selected=null;return;}
 for(const d of rec.drafts){let opt=document.createElement('option');opt.value=d.draft_id;
 opt.textContent=d.source_method+' #'+(d.source_hypothesis_index+1)+(d.cross_method_agreement_UNCALIBRATED?' · оба метода':'')+(d.duplicate_of_draft_id?' · повтор':'');sel.append(opt);}
 s31Selected=rec.drafts.some(p=>p.draft_id===old)?old:rec.drafts[0]?.draft_id||null;sel.value=s31Selected||'';
 for(const d of rec.drafts){let pp=s31Polyline(d);const decision=s31Saved(d.draft_id);
 const relation=rec.relation_drafts.find(r=>r.relation_id===s31RelationSelected);
 const highlighted=d.draft_id===s31Selected;
 const inRel=relation&&(relation.candidate_a===d.draft_id||relation.candidate_b===d.draft_id);
 const color=decision?.status==='USER_REJECTED_PROVISIONAL'?'#ff5757':decision?.status==='USER_ACCEPTED_PROVISIONAL'?'#8aff80':d.source_method==='LSD_line_piece'?'#ffa0c7':d.source_method==='classical'?'#35e8c4':'#92b3ff';
 const line=s31Shape('polyline',{points:pp.map(x=>x.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':highlighted?2.8:inRel?2.2:1.25,opacity:highlighted||inRel?1:.35,'stroke-dasharray':decision?.status==='USER_ACCEPTED_PROVISIONAL'?'none':'4 4'});s31Svg.append(line);}
 if(s31Edit&&s31Selected){let draft=s31CurrentDraft(),pts=s31Polyline(draft);
 let chosen=new Set([0,pts.length-1]);for(let i=8;i<pts.length-1;i+=8)chosen.add(i);
 for(const i of chosen){let pt=pts[i],c=s31Shape('circle',{cx:pt[0],cy:pt[1],r:5,stroke:'#fbdc68','stroke-width':1.4,fill:'#172333','data-index':i});
 c.style.pointerEvents='auto';c.addEventListener('pointerdown',e=>{e.stopPropagation();e.preventDefault();s31ActiveHandle=i;s31Svg.setPointerCapture(e.pointerId);});s31Svg.append(c);}
 }
 let stats=s31Stats(),recRisk=rec.stratum==='fiducial_control'?' ВНИМАНИЕ: контрольный репер, автоматические линии могут быть ложными.':'';
 let status=s31Selected?(s31Saved(s31Selected)?.status||'MACHINE_DRAFT_UNREVIEWED'):'Нет кандидатных линий';
 byId('s31-status').textContent='Участок '+rec.crop_id+' · черновиков '+rec.drafts.length+' · статус '+status+recRisk+' | Всего '+stats.total+' · локально принято '+stats.nAccept+' · отклонено '+stats.nReject+' · сомнительных '+stats.nUncertain+' · исправлены узлы '+stats.nEdit+'. Физиками ещё не проверено.';
 byId('s31-edit').textContent=s31Edit?'Завершить правку':'Редактировать узлы';
 byId('s31-svg').style.pointerEvents=s31Edit?'auto':'none';
}
window.s31Render=s31Render;
function s31ToXY(e){let bounds=s31Svg.getBoundingClientRect();return [Math.max(0,Math.min(511,512*(e.clientX-bounds.left)/bounds.width)),Math.max(0,Math.min(511,512*(e.clientY-bounds.top)/bounds.height))];}
s31Svg.addEventListener('pointermove',e=>{if(s31ActiveHandle<0||!s31Selected)return;
 let draft=s31CurrentDraft(),pts=s31Polyline(draft).map(p=>p.slice()),i=s31ActiveHandle,old=pts[i],now=s31ToXY(e);
 // Smooth local point correction, preserving relative neighbor topology.
 let dx=now[0]-old[0],dy=now[1]-old[1];for(let j=Math.max(0,i-8);j<=Math.min(pts.length-1,i+8);j++){
 let weight=Math.exp(-.5*((j-i)/3.5)**2);pts[j]=[Math.max(0,Math.min(511,pts[j][0]+weight*dx)),Math.max(0,Math.min(511,pts[j][1]+weight*dy))];}
 s31SaveDecision('USER_EDITED_PENDING_REVIEW',pts.map(p=>p.map(z=>Number(z.toFixed(2)))));
});
s31Svg.addEventListener('pointerup',()=>{s31ActiveHandle=-1;});s31Svg.addEventListener('pointercancel',()=>{s31ActiveHandle=-1;});
byId('s31-candidate').onchange=e=>{s31Selected=e.target.value;s31Render();};
byId('s31-relation').onchange=e=>{s31RelationSelected=e.target.value;s31Render();};
function s31SaveRelation(status){if(!s31RelationSelected)return;if(status==='RESET')delete s31RelationDecisions[s31RelationSelected];else s31RelationDecisions[s31RelationSelected]={status,review_origin:'LOCAL_USER_NOT_PHYSICIST_CERTIFIED'};s31Render();}
byId('s31-rel-same').onclick=()=>s31SaveRelation('LOCAL_SAME_PROVISIONAL');
byId('s31-rel-diff').onclick=()=>s31SaveRelation('LOCAL_DIFFERENT_PROVISIONAL');
byId('s31-rel-unsure').onclick=()=>s31SaveRelation('LOCAL_UNSURE');
byId('s31-rel-clear').onclick=()=>s31SaveRelation('RESET');
byId('s31-accept').onclick=()=>s31SaveDecision('USER_ACCEPTED_PROVISIONAL');
byId('s31-reject').onclick=()=>s31SaveDecision('USER_REJECTED_PROVISIONAL');
byId('s31-uncertain').onclick=()=>s31SaveDecision('USER_UNCERTAIN');
byId('s31-clear').onclick=()=>s31SaveDecision('RESET');
byId('s31-edit').onclick=()=>{s31Edit=!s31Edit;s31ActiveHandle=-1;s31Render();};
byId('s31-next').onclick=()=>{const all=s31Pack.photos.flatMap(r=>r.drafts.map(d=>({crop_id:r.crop_id,id:d.draft_id})));
 if(!all.length)return;let index=all.findIndex(q=>q.id===s31Selected);
 for(let k=1;k<=all.length;k++){const q=all[(index+k+all.length)%all.length];if(s31Saved(q.id))continue;
 let i=entries.findIndex(e=>e.crop_id===q.crop_id);if(i<0)throw Error('source missing');active=i;byId('crop').value=String(i);
 s31Selected=q.id;s31Edit=false;update();return;}
 byId('s31-status').textContent='У всех кандидатов есть локальное решение. Экспертная физическая проверка всё ещё необходима.';
};
function s31Export(){return {schema:'s31_machine_corrections_v1',machine_source_s28_sha256:s31Pack.source_S28_proposals_sha256,
 machine_source_s29_sha256:s31Pack.source_S29_reviewer_sha256,provenance:'LOCAL_USER_CORRECTIONS_NOT_PHYSICIST_CERTIFIED',
 all_unreviewed_stay_unknown:true,physical_joins_authorized:false,
 crops:s31Pack.photos.map(x=>({crop_id:x.crop_id,source_crop_sha256:x.source_original_png_sha256_from_S29,
  source_gray_pixels_sha256:x.source_gray_pixels_sha256})),decisions:JSON.parse(JSON.stringify(s31Decisions)),
  relation_decisions:JSON.parse(JSON.stringify(s31RelationDecisions))};}
byId('s31-export').onclick=()=>{const obj=s31Export();const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='S31_machine_labels_local_corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);};
byId('s31-import').onchange=async e=>{try{let f=e.target.files?.[0];if(!f)return;let data=JSON.parse(await f.text());
 if(data.schema!=='s31_machine_corrections_v1'||data.physical_joins_authorized!==false||data.all_unreviewed_stay_unknown!==true||data.machine_source_s28_sha256!==s31Pack.source_S28_proposals_sha256||data.machine_source_s29_sha256!==s31Pack.source_S29_reviewer_sha256)throw Error('Неправильная версия данных или источника');
 if(!Array.isArray(data.crops)||data.crops.length!==s31Pack.photos.length)throw Error('Неполные фото');
 for(let i=0;i<data.crops.length;i++){let c=data.crops[i],p=s31Pack.photos[i];if(c.crop_id!==p.crop_id||c.source_crop_sha256!==p.source_original_png_sha256_from_S29||c.source_gray_pixels_sha256!==p.source_gray_pixels_sha256)throw Error('Подменена исходная фотография');}
 let decisions=data.decisions;if(!decisions||typeof decisions!=='object'||Array.isArray(decisions))throw Error('Некорректные решения');
 for(const [id,d] of Object.entries(decisions)){if(!s31Drafts.has(id)||!['USER_ACCEPTED_PROVISIONAL','USER_REJECTED_PROVISIONAL','USER_UNCERTAIN','USER_EDITED_PENDING_REVIEW'].includes(d.status))throw Error('Неизвестный кандидат/статус');
 if(d.edited_points_xy!==null&&d.edited_points_xy!==undefined){if(!Array.isArray(d.edited_points_xy)||d.edited_points_xy.length!==s31Drafts.get(id).proposal.points_xy.length||d.edited_points_xy.some(x=>!Array.isArray(x)||x.length!==2||x.some(v=>!Number.isFinite(v)||v<0||v>511)))throw Error('Некорректная правка точек');}}
 let relations=data.relation_decisions||{};let validRelations=new Set(s31Pack.photos.flatMap(r=>r.relation_drafts.map(q=>q.relation_id)));
 for(const [rid,ans] of Object.entries(relations)){if(!validRelations.has(rid)||!['LOCAL_SAME_PROVISIONAL','LOCAL_DIFFERENT_PROVISIONAL','LOCAL_UNSURE'].includes(ans.status))throw Error('Неверная связь фрагментов');}
 for(const key of Object.keys(s31Decisions))delete s31Decisions[key];Object.assign(s31Decisions,decisions);
 for(const key of Object.keys(s31RelationDecisions))delete s31RelationDecisions[key];Object.assign(s31RelationDecisions,relations);s31Render();
 byId('s31-status').textContent='Сохранённые исправления загружены. Ни одно решение не является физической экспертизой.';
 }catch(err){byId('s31-status').textContent='Ошибка загрузки: '+err.message;}};
window.s31ExportCorrections=s31Export;window.s31Render();
</script>'''
    return html.replace('</body></html>',script.replace('__DRAFT__',payload)+'</body></html>')
