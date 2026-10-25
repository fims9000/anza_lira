"""S20: budgeted HUMAN review ordering for real H50Z Frangi proposals.

Ranking scores are geometry heuristics, NOT posterior probability, expert truth,
or approval to merge particle traces. Existing S19 package digest is preserved.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any

import numpy as np


def _endpoints(c):
    pts=np.asarray(c['centerline_local_xy_px'],float)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or not np.isfinite(pts).all():
        raise ValueError('Malformed source proposal points')
    return (pts[0],pts[1]-pts[0]),(pts[-1],pts[-1]-pts[-2])


def _edge_affinity(a,b,max_gap=60.) -> dict | None:
    """Only suggest a possible review, never infer SAME_TRACK.

    Geometry is local to one tile; shared source frame images must NOT be
    matched across projections here without optical calibration.
    """
    va=_endpoints(a);vb=_endpoints(b)
    possible=[]
    for ia,(pa,ta) in enumerate(va):
        for ib,(pb,tb) in enumerate(vb):
            gap=pb-pa;dist=float(np.linalg.norm(gap))
            if dist<=.01 or dist>max_gap:continue
            ta=ta/max(np.linalg.norm(ta),1e-8);tb=tb/max(np.linalg.norm(tb),1e-8)
            u=gap/dist
            # Axial equivalence allows either a/b orientation; this is *not* PID.
            aligned=float(abs(np.dot(ta,u))*abs(np.dot(tb,u)))
            score=math.exp(-dist/30.)*(.15+.85*aligned)
            possible.append((score,dist,aligned,ia,ib))
    if not possible:return None
    score,dist,aligned,ia,ib=max(possible)
    return {'candidate_a':a['id'],'candidate_b':b['id'],
            'score_unvalidated':round(score,6),'gap_px':round(dist,3),
            'tangent_alignment_unvalidated':round(aligned,6),
            'source_endpoints':[ia,ib],'needs_human_pair_label':True}


def rank_review_questions(package:dict, *, limit_candidates:int=37, limit_pairs:int=35) -> dict:
    if package.get('schema')!='s19_real_photo_review_package_v1' or not package.get('package_sha256'):
        raise ValueError('Provenance-verified S19 package expected')
    if not package.get('tiles') or not 1<=limit_candidates<=1000 or not 0<=limit_pairs<=1000:
        raise ValueError('Review question limits/tile input invalid')
    all_candidates=[];all_pairs=[];seen=set()
    for tile in package['tiles']:
        tile_id=tile['tile_id'];parent=tile['parent_photo']
        if tile_id in seen:raise ValueError('Duplicate tile')
        seen.add(tile_id)
        candidates=tile['candidates']
        pair_scores=defaultdict(list)
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                x=_edge_affinity(a,b)
                if x is not None:
                    pair_scores[a['id']].append(x['score_unvalidated'])
                    pair_scores[b['id']].append(x['score_unvalidated'])
                    all_pairs.append({'tile_id':tile_id,'parent_photo':parent,**x})
        for c in candidates:
            coords=c['centerline_local_xy_px'];length=float(c['length_px'])
            if len(coords)<2 or not np.isfinite(length) or length<=0:raise ValueError('Invalid source candidate')
            ambiguity=sorted(pair_scores[c['id']],reverse=True)
            # Balanced: take both long lines worth labeling and uncertain joins.
            score=.4*min(length,160)/160+.6*min(sum(ambiguity[:2]),1.)
            all_candidates.append({'tile_id':tile_id,'photo_id':parent,
                'candidate_id':c['id'],'length_px':round(length,2),
                'score_unvalidated':round(score,6),
                'nearby_pair_candidates':len(ambiguity),'expert_label':'UNKNOWN'})
    # Diverse labels: ensure questions distribute across source photographs/tiles.
    remaining=all_candidates[:];selected=[];by_tile=defaultdict(int);by_photo=defaultdict(int)
    while remaining and len(selected)<limit_candidates:
        best=max(remaining,key=lambda z:(z['score_unvalidated']/(1+.6*by_tile[z['tile_id']]+.4*by_photo[z['photo_id']]),z['candidate_id']))
        selected.append(best)
        by_tile[best['tile_id']]+=1;by_photo[best['photo_id']]+=1
        remaining.remove(best)
    pairs=sorted(all_pairs,key=lambda z:(-z['score_unvalidated'],z['tile_id'],z['candidate_a'],z['candidate_b']))[:limit_pairs]
    return {'schema':'s20_real_photo_review_priority_v1',
        'source_package_sha256':package['package_sha256'],
        'source_photo_ids':sorted({t['parent_photo'] for t in package['tiles']}),
        'all_candidates':len(all_candidates),'all_geometric_pair_suggestions':len(all_pairs),
        'candidate_questions':selected,'pair_questions':pairs,
        'expert_approved_lines':0,'expert_approved_connections':0,
        'calibrated_probability':False,
        'note':'Geometry-based review ordering only. Candidate lines and pairs remain UNKNOWN until signed human review.'}


def integrate_priority_ui(original_html:str,priority:dict) -> str:
    """Add next-question controls without modifying the S19 canonical JSON format."""
    if 'function changeTile(' not in original_html or '___JSON_PAYLOAD___' in original_html:
        raise ValueError('Expected compiled S19 workbench HTML')
    if priority.get('schema')!='s20_real_photo_review_priority_v1':
        raise ValueError('Wrong priority schema')
    safe=json.dumps(priority,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    add=r'''<script id="s20-priority-payload" type="application/json">___PRIORITY___</script>
<script>
'use strict';
const s20Priority=JSON.parse(document.getElementById('s20-priority-payload').textContent);
if(s20Priority.source_package_sha256!==pack.package_sha256)throw Error('Priority list refers to another source package');
const s20Box=document.createElement('div');s20Box.className='panel';
s20Box.innerHTML='<h2>0. Очередь для минимальной разметки</h2><p class="hint">Подсказки только по геометрии, не готовые ответы. Проверяйте линии, затем пары. Неразмеченное остаётся неизвестным.</p><button id="s20-next-line">Следующая линия</button> <button id="s20-next-pair">Следующая пара</button><p id="s20-review-note" class="hint"></p>';
document.querySelector('aside').prepend(s20Box);
let s20PairIndex=0;
function s20NextLine(){for(const q of s20Priority.candidate_questions){const idx=tiles.findIndex(t=>t.tile_id===q.tile_id);if(idx<0)continue;
 if(Object.prototype.hasOwnProperty.call(states[idx].decisions,q.candidate_id))continue;
 changeTile(idx);selected=q.candidate_id;tool='select';drawSvg();
 byId('s20-review-note').textContent='Подтвердите или отклоните линию '+q.candidate_id+'; затем снова нажмите «Следующая линия». Приоритет не является вероятностью трека.';return;}
 byId('s20-review-note').textContent='Все предложенные линии просмотрены. Пропущенные настоящие линии дорисуйте вручную.';}
function s20NextPair(){let q=null;let idx=-1;
 for(let step=0;step<s20Priority.pair_questions.length;step++){
  const test=s20Priority.pair_questions[s20PairIndex++%s20Priority.pair_questions.length];
  const candidateIdx=tiles.findIndex(t=>t.tile_id===test.tile_id);if(candidateIdx<0)continue;
  if(states[candidateIdx].links.some(link=>[link.a,link.b].sort().join('|')===[test.candidate_a,test.candidate_b].sort().join('|')))continue;
  q=test;idx=candidateIdx;break;
 }
 if(!q){byId('s20-review-note').textContent='В этом наборе нет непроверенных близких пар; любые другие пары можно выбрать вручную.';return;}
 changeTile(idx);selected=q.candidate_a;tool='select';drawSvg();
 byId('s20-review-note').textContent='Возможная пара: '+q.candidate_a+' и '+q.candidate_b+'. Сначала подтвердите ОБЕ линии как треки, затем назначьте A/Б и отметьте «Одна частица» или «Разные треки». Гипотеза не равна ответу эксперта.';}
byId('s20-next-line').onclick=s20NextLine;
byId('s20-next-pair').onclick=s20NextPair;
</script>'''
    if not original_html.endswith('</html>'):
        raise ValueError('Malformed original html')
    return original_html.replace('</body></html>',add.replace('___PRIORITY___',safe)+'</body></html>')
