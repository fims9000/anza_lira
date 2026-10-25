"""Present real-image ANZA centerline evidence inside private S32 reviewer.

The S31/S32 local decision/export logic remains authoritative and untouched.
Machine scores are explicitly UNCALIBRATED; this does not certify a track.
"""
from __future__ import annotations
import json,re
from .s33_normalized_inference import SCHEMA


def inject_evidence(original_html:str,unverified_evidence:dict)->str:
    if unverified_evidence.get('schema')!=SCHEMA or unverified_evidence.get('physics_auto_join_authorized') is not False:
        raise ValueError('Unsafe S33 evidence schema or physical join claim')
    if unverified_evidence.get('merged_reviewer_drafts')!=133 or unverified_evidence.get('true_expert_H50Z_track_labels_added')!=0:
        raise ValueError('S33 must have exactly 133 UNVERIFIED source drafts')
    m=re.search(r'<script(?=[^>]*id="s31-draft-payload")(?=[^>]*type="application/json")[^>]*>(.*?)</script>',original_html,re.S)
    if m is None:raise ValueError('Original source S31 review draft not available')
    base=json.loads(m.group(1));expected={d['draft_id']:(r['crop_id'],r['photo_id']) for r in base['photos'] for d in r['drafts']}
    if len(expected)!=133 or base.get('physical_joins_authorized') is not False:
        raise ValueError('Original S31 draft not review-only or inconsistent')
    rows=unverified_evidence['rows']
    if len(rows)!=133 or len({r['draft_id'] for r in rows})!=133:
        raise ValueError('Missing or repeated S33 machine evidence')
    for r in rows:
        if expected.get(r['draft_id'])!=(r['crop_id'],r['source_photo']):
            raise ValueError('Foreign H50Z photo/candidate inserted')
        if r.get('expert_approved') is not False or r.get('physics_auto_merge_allowed') is not False:
            raise ValueError('Model evidence cannot approve physics')
    if 'id="s33-evidence-panel"' in original_html:raise ValueError('S33 already injected')
    if '</body></html>' not in original_html:raise ValueError('Unexpected original browser document')
    # Strip accidental free-form fields; keep only the locked candidate evidence.
    compact={r['draft_id']:{
        'source_photo':r['source_photo'],
        'source_method':r['source_method'],
        'quantile':r['AZUNet_quantile'],
        'local':r['AZUNet_local'],
        'difference':r['cross_normalization_disagreement_UNCALIBRATED']
    } for r in rows}
    payload=json.dumps(compact,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    addon='''<section id="s33-evidence-panel" style="padding:12px;margin:10px 0;border:1px solid #999;border-radius:10px">
<h3>S33 · сравнение двух нормализаций ANZA</h3>
<p><strong>Только машинная подсказка.</strong> Значения не откалиброваны, не являются вероятностями реального трека и не подтверждены физиками. Локальные решения и ручной экспорт S31/S32 не изменяются.</p>
<p id="s33-evidence-status">Выберите одну линию в редакторе.</p></section>
<script id="s33-evidence-source" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
(function(){
 const data=JSON.parse(document.getElementById('s33-evidence-source').textContent);
 const panel=document.getElementById('s33-evidence-panel');
 const target=document.getElementById('s31-panel');if(!target)throw Error('Missing original review panel');
 target.insertAdjacentElement('afterend',panel);
 function update(){
   const id=typeof s31Selected==='string'?s31Selected:null;
   const item=data[id];const label=document.getElementById('s33-evidence-status');
   if(!item){label.textContent='Для этого кандидата оценки AZUNet отсутствуют. Любое решение остаётся предварительным.';return;}
   const q=item.quantile,l=item.local;
   if(q?.status!=='UNVERIFIED_MODEL_EVIDENCE'||l?.status!=='UNVERIFIED_MODEL_EVIDENCE'){
      label.textContent='Недостаточно данных для оценки этой геометрии. Проверяйте линию на исходном BMP.';return;
   }
   label.textContent='Линия '+id.slice(-8)+' / '+item.source_method+
      ' | контраст AZUnet (квантиль) '+q.median_center_minus_flank.toFixed(3)+
      ' | контраст AZUnet (локальный) '+l.median_center_minus_flank.toFixed(3)+
      ' | разногласие '+Math.abs(q.median_center_minus_flank-l.median_center_minus_flank).toFixed(3)+
      '. Ни одно число не является вероятностью частицы.';
 }
 const old=window.s31Render;
 if(typeof old==='function')window.s31Render=(...args)=>{let v=old(...args);update();return v;};
 for(const id of ['s31-candidate','crop','s32-next-risk','s32-next-pair','s31-next']){
   let ctl=document.getElementById(id);if(ctl)ctl.addEventListener(id==='crop'||id==='s31-candidate'?'change':'click',update);
 }
 update();window.s33UpdateEvidence=update;
})();
</script>'''
    return original_html.replace('</body></html>',addon.replace('__PAYLOAD__',payload)+'</body></html>',1)
