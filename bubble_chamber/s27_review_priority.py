"""S27 rank REAL H50Z photo questions from UNLABELLED model uncertainty.

No expert ground truth is created. All scores are heuristic, uncalibrated and
cannot authorize an automatic physical association or particle identity.
"""
from __future__ import annotations
from collections import defaultdict
import json,re
from pathlib import Path
import numpy as np

SCHEMA='s27_source_locked_heuristic_expert_crop_queue_v1'


def _pct_rank(values:list[float])->np.ndarray:
    v=np.asarray(values,float)
    if not np.isfinite(v).all():raise ValueError('nonfinite unverified model measurements')
    return np.argsort(np.argsort(v,kind='stable'),kind='stable')/max(len(v)-1,1)


def rank_questions(rows:list[dict],*,maximum:int|None=None)->dict:
    if not rows or len({r['source']['crop_id'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate H50Z source crops')
    if not {'005','006'}<=set(x['source']['photo_id'] for x in rows):
        raise ValueError('need both source photographs to avoid review queue collapse')
    n=len(rows)
    maximum=n if maximum is None else min(maximum,n)
    if maximum<1:raise ValueError('invalid question count')
    ambiguity=np.array([r['mixed_model_candidates_NO_TRUTH']['unverified_neural_disagreement_fraction'] for r in rows])
    sato=np.array([abs(r['mixed_model_candidates_NO_TRUTH']['unverified_consensus_fraction'] -
                   r['mixed_model_candidates_NO_TRUTH']['unverified_consensus_sato_overlap_fraction']) for r in rows])
    instability=np.array([np.mean([r['original_score_to_overlap'][name]['candidate_difference']['candidate_pixel_fraction_changed']
                     for name in ('unet','azunet')]) for r in rows])
    seam=np.array([np.mean([r['original_score_to_overlap'][name]['mean_boundary_jump_reduction_fraction']
                        for name in ('unet','azunet','guided_az')]) for r in rows])
    # No physical confidence: all four scalars are unsupervised/repeated output
    # diagnostics. A model agreeing with itself might still be wrong.
    score=.38*_pct_rank(ambiguity)+.24*_pct_rank(instability)+.18*_pct_rank(sato)+.20*_pct_rank(seam)
    by_photo=defaultdict(int);by_stratum=defaultdict(int)
    remaining=set(range(n));ordered=[]
    # A scientific review needs quiet and fiducial controls, not just high
    # model disagreement. Cover each stratum in EACH photo up front, then
    # let uncalibrated model difficulty rank the remaining questions.
    strata_order=('start_anchor_nearby','dense','faint','geometry','fiducial_control','quiet')
    control_first=[]
    for stratum in strata_order:
        for photo in ('005','006'):
            matches=[i for i in remaining if rows[i]['source']['photo_id']==photo and
                     rows[i]['source']['review_stratum']==stratum]
            if matches:
                first=max(matches,key=lambda i:(float(score[i]),rows[i]['source']['crop_id']))
                control_first.append(first)
                remaining.remove(first)
    # Limit the warm-up coverage to the requested review budget.
    initial=control_first[:maximum]
    remaining.update(control_first[maximum:])
    while remaining or initial:
        if len(ordered)>=maximum:break
        if initial:
            idx=initial.pop(0)
        else:
            def priority(i):
                z=rows[i]['source'];div=1+.8*by_photo[z['photo_id']]+1.1*by_stratum[z['review_stratum']]
                return (float(score[i])/div,z['crop_id'])
            idx=max(remaining,key=priority)
            remaining.remove(idx)
        src=rows[idx]['source']
        by_photo[src['photo_id']]+=1;by_stratum[src['review_stratum']]+=1
        subtype={'fiducial_control':'fiducial control: do NOT label whole photo as artifact',
                 'quiet':'quiet control: unmarked pixels remain UNKNOWN',
                 'start_anchor_nearby':'start anchor is not track mask'}.get(src['review_stratum'],'review track OR artifact line manually')
        ordered.append({'crop_id':src['crop_id'],'photo':src['photo_id'],
                        'stratum':src['review_stratum'],'source_crop_sha256':src['source_crop_sha256'],
                        'review_priority_heuristic_UNCALIBRATED':round(float(score[idx]),6),
                        'neural_disagreement_fraction_UNVERIFIED':round(float(ambiguity[idx]),6),
                        'segmentation_stability_delta_fraction_UNVERIFIED':round(float(instability[idx]),6),
                        'review_type':subtype,'human_verified_line_pixels':0,
                        'human_verified_relations':0,'automatic_physics_merges':0})
    return {'schema':SCHEMA,'questions':ordered,'source_crop_count':n,
       'source_photos':sorted(by_photo),'human_verified_real_track_pixels':0,
       'human_verified_real_track_links':0,
       'scores_are_probabilities':False,'model_train_labels_on_original_native_traces':0,
       'review_priority_is_not_track_confidence':True,
       'no_auto_join_or_physical_output':True}


def embed_priorities(existing_review_html:str,priority:dict)->str:
    """Preserve the exact S26 JSON export schema and the existing source photo data.

    The S27 queue is presentation-only and never serializes its predictions
    into annotation strokes. Existing saved S26 expert JSON remains valid.
    """
    if priority.get('schema')!=SCHEMA or not priority.get('no_auto_join_or_physical_output'):
        raise ValueError('source-verified S27 queue required')
    match=re.search(r'(<script type="application/json" id="payload">)(.*?)(</script>)',existing_review_html,re.S)
    if not match:raise ValueError('original S26 review payload missing')
    data=json.loads(match.group(2))
    if data.get('schema')!='s26_review_visual_payload_v1':raise ValueError('unexpected photo reviewer version')
    available={p['crop_id']:p for p in data['photos']}
    if set(available)!=set(q['crop_id'] for q in priority['questions']):
        raise ValueError('review queue not same set of real source photos')
    for question in priority['questions']:
        if available[question['crop_id']]['sha256']!=question['source_crop_sha256']:
            raise ValueError('source photo SHA differs from question')
    payload=json.dumps(priority,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    content='''<section id="s27-review-controls" style="border:1px solid #59758b;border-radius:10px;padding:12px;margin:10px 0">
<strong>Очередь S27 — ручная проверка без доверенных меток</strong>
<p class="help">Порядок вычислен по несогласию и нестабильности сетей. Это НЕ вероятность настоящего трека. Ничего не подтверждается автоматически.</p>
<button id="s27-review-next" type="button">Следующий участок на проверку</button>
<label>Порядок вопросов <select id="s27-review-choice"></select></label>
<p id="s27-review-context" class="help"></p>
</section>'''
    script=r'''<script type="application/json" id="s27-queue-json">___QUEUE___</script>
<script>
'use strict';
const s27=JSON.parse(document.getElementById('s27-queue-json').textContent);
if(s27.schema!=='s27_source_locked_heuristic_expert_crop_queue_v1'||!s27.no_auto_join_or_physical_output)throw Error('unsafe review priority');
const s27Ids=new Map(entries.map((e,i)=>[e.crop_id,{index:i,sha256:e.sha256}]));
for(const q of s27.questions){const x=s27Ids.get(q.crop_id);if(!x||x.sha256!==q.source_crop_sha256)throw Error('wrong source H50Z crop in queue');}
const s27Select=byId('s27-review-choice');
for(let i=0;i<s27.questions.length;i++){const q=s27.questions[i];const item=document.createElement('option');item.value=String(i);item.textContent=(i+1)+'. '+q.photo+' / '+q.stratum+' / '+q.crop_id.split('_').slice(-1)[0];s27Select.append(item);}
let s27Cursor=-1;
function s27Jump(i){s27Cursor=i;const q=s27.questions[i];active=s27Ids.get(q.crop_id).index;
 byId('crop').value=String(active);ongoing=null;update();s27Select.value=String(i);
 byId('s27-review-context').textContent='№'+(i+1)+' из '+s27.questions.length+' | '+q.crop_id+' | '+q.stratum+' | '+q.review_type+' | несогласие сетей: '+(100*q.neural_disagreement_fraction_UNVERIFIED).toFixed(2)+'% | Доверенных меток: '+annotations[q.crop_id].length+' штрихов. Отсутствие штриха = НЕИЗВЕСТНО.';}
byId('s27-review-next').onclick=()=>s27Jump((s27Cursor+1)%s27.questions.length);
s27Select.onchange=e=>s27Jump(Number(e.target.value));
</script>'''
    if '</body></html>' not in existing_review_html:raise ValueError('broken S26 HTML')
    return existing_review_html.replace('<div class="tools"><label>Фрагмент',content+'\n<div class="tools"><label>Фрагмент',1).replace('</body></html>',script.replace('___QUEUE___',payload)+'</body></html>')
