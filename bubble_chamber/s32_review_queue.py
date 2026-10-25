"""S32 human review queue for S31 machine-only annotations; no new labels.

All scores are procedural priorities, NEVER probability of track identity.
Preserves S31 correction schema and every source proposal without mutation.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np
from .s31_machine_first import validate_draft_pack

SCHEMA='s32_machine_draft_risk_review_queue_v1'


def build_review_queue(drafts, source_file_sha256):
    validate_draft_pack(drafts)
    candidates=[];pairs=[]
    for crop in drafts['photos']:
        for d in crop['drafts']:
            pts=np.asarray(d['points_xy'],dtype=float)
            if pts.ndim!=2 or pts.shape[1]!=2 or not np.isfinite(pts).all():raise ValueError('Invalid source trace geometry')
            reasons=[];risk=0
            if crop['stratum']=='fiducial_control':risk+=5;reasons.append('fiducial control: likely reference mark')
            if d['duplicate_of_draft_id']:risk+=3;reasons.append('duplicate geometry hypothesis')
            if d.get('cross_method_agreement_UNCALIBRATED') is False:risk+=2;reasons.append('no independent method agreement')
            if d['source_method']=='LSD_line_piece':risk+=2;reasons.append('unanchored image segment')
            support=d.get('photometry_support_fraction_UNCALIBRATED')
            if support is not None and support<.45:risk+=2;reasons.append('low measured photometric support')
            if min(pts[:,0].min(),pts[:,1].min(),511-pts[:,0].max(),511-pts[:,1].max())<12:
                risk+=1;reasons.append('near tile edge')
            if not reasons:reasons=['requires first local review']
            candidates.append({'crop_id':crop['crop_id'],'photo_id':crop['photo_id'],'draft_id':d['draft_id'],
                  'risk_priority_NOT_PROBABILITY':risk,'reasons':reasons,
                  'machine_label':'UNKNOWN','human_certified':False})
        byid={d['draft_id']:d for d in crop['drafts']}
        for rel in crop.get('relation_drafts',[]):
            if rel['candidate_a'] not in byid or rel['candidate_b'] not in byid:
                raise ValueError('Relation mentions a candidate from another photo')
            a=np.asarray(byid[rel['candidate_a']]['points_xy'],float)
            b=np.asarray(byid[rel['candidate_b']]['points_xy'],float)
            ends=[(a[0],a[1]-a[0]),(a[-1],a[-1]-a[-2])]
            bend=[(b[0],b[1]-b[0]),(b[-1],b[-1]-b[-2])]
            aff=[]
            for pa,da in ends:
                for pb,db in bend:
                    delta=pb-pa;gap=float(np.linalg.norm(delta))
                    if gap<1e-6:continue
                    tangent=abs(float(np.dot(da,delta)/(max(np.linalg.norm(da),1e-8)*gap)))
                    tangent*=abs(float(np.dot(db,delta)/(max(np.linalg.norm(db),1e-8)*gap)))
                    aff.append(float(np.exp(-gap/40)*tangent))
            priority=max(aff,default=0.)
            pairs.append({'crop_id':crop['crop_id'],'photo_id':crop['photo_id'],
                          'relation_id':rel['relation_id'],'candidate_a':rel['candidate_a'],
                          'candidate_b':rel['candidate_b'],
                          'geometric_review_rank_NOT_PROBABILITY':round(priority,6),
                          'question':'same / different / unknown (only local provisional)',
                          'human_certified':False})
    candidates.sort(key=lambda x:(-x['risk_priority_NOT_PROBABILITY'],x['photo_id'],x['crop_id'],x['draft_id']))
    pairs.sort(key=lambda x:(-x['geometric_review_rank_NOT_PROBABILITY'],x['photo_id'],x['crop_id'],x['relation_id']))
    return {'schema':SCHEMA,'source_machine_drafts_file_sha256':source_file_sha256,
            'source_S28_sha256':drafts['source_S28_proposals_sha256'],
            'source_S29_sha256':drafts['source_S29_reviewer_sha256'],
            'original_photos':len(drafts['photos']), 'machine_drafts':len(candidates),
            'possible_pairs':len(pairs),'expert_certified_lines':0,'expert_certified_pairs':0,
            'priority_not_probability':True,'physical_automatic_joins_authorized':False,
            'candidate_questions':candidates,'relation_questions':pairs}


def inject_review_queue(html:str,drafts:dict,queue:dict)->str:
    if 'id="s31-draft-payload"' not in html or 'window.s31ExportCorrections' not in html:
        raise ValueError('S31 full review workbench required')
    m=re.search(r'<script[^>]*id="s31-draft-payload"[^>]*>(.*?)</script>',html,re.S)
    if not m or json.loads(m.group(1))!=drafts:
        raise ValueError('HTML embeds different machine draft proposal source')
    if queue.get('schema')!=SCHEMA or queue['source_S28_sha256']!=drafts['source_S28_proposals_sha256'] or queue['source_S29_sha256']!=drafts['source_S29_reviewer_sha256']:
        raise ValueError('Review queue provenance mismatch')
    if 'id="s32-review-priority"' in html:raise ValueError('Already enhanced by S32')
    panel='''<section id="s32-review-priority" style="border:1px solid #7fb5c5;border-radius:10px;margin:12px 0;padding:12px;background:#202b31">
<h2 style="font-size:1.12rem">S32 · очередь исправления машинной разметки</h2>
<p>Подсказки основаны на эвристическом риске: это НЕ вероятность настоящего трека и НЕ заключение физика. Существующий экспорт S31 сохраняется отдельно.</p>
<button id="s32-next-risk">Следующая сомнительная линия</button>
<button id="s32-next-pair">Следующая возможная связь</button>
<p id="s32-priority-status">Неоднозначные гипотезы ждут локальной проверки.</p></section>'''
    html=html.replace('<section id="s31-panel"',panel+'\n<section id="s31-panel"',1)
    encoded=json.dumps(queue,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')
    addition=r'''<script id="s32-review-payload" type="application/json">__SOURCE__</script>
<script>
'use strict';
const s32Review=JSON.parse(document.getElementById('s32-review-payload').textContent);
if(s32Review.schema!=='s32_machine_draft_risk_review_queue_v1'||s32Review.source_S28_sha256!==s31Pack.source_S28_proposals_sha256||s32Review.source_S29_sha256!==s31Pack.source_S29_reviewer_sha256)throw Error('S32 sources changed');
let s32PairCursor=0;
function s32ChooseCrop(cropId){let idx=entries.findIndex(p=>p.crop_id===cropId);if(idx<0)throw Error('S32 crop missing');active=idx;byId('crop').value=String(idx);}
byId('s32-next-risk').onclick=()=>{for(const q of s32Review.candidate_questions){
 if(Object.prototype.hasOwnProperty.call(s31Decisions,q.draft_id))continue;
 s32ChooseCrop(q.crop_id);s31Selected=q.draft_id;s31Edit=false;update();
 byId('s32-priority-status').textContent='Разметить вручную: '+q.draft_id+' · причины: '+q.reasons.join('; ')+'. Риск — лишь порядок проверки.';return;
 }byId('s32-priority-status').textContent='Все предложенные линии получили локальные решения; их по-прежнему нужно проверить физикам.';};
byId('s32-next-pair').onclick=()=>{for(let k=0;k<s32Review.relation_questions.length;k++){
 const q=s32Review.relation_questions[s32PairCursor++%s32Review.relation_questions.length];
 if(Object.prototype.hasOwnProperty.call(s31RelationDecisions,q.relation_id))continue;
 s32ChooseCrop(q.crop_id);s31RelationSelected=q.relation_id;s31Edit=false;update();
 byId('s32-priority-status').textContent='Проверьте связь '+q.candidate_a+' / '+q.candidate_b+' как один / разные / не знаю. Автоматического соединения нет.';return;
 }byId('s32-priority-status').textContent='Нет непроверенных пар в предложенной очереди.';};
</script>'''
    return html.replace('</body></html>',addition.replace('__SOURCE__',encoded)+'</body></html>')
