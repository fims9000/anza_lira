"""Private browser overlay: prioritize 28 existing S31 pair questions, never certify.

Original S31/S32 provisional line and pair decision storage, reload/export and
manual editing remain unchanged. S35 only displays unverified scientific cues.
"""
from __future__ import annotations
import json,re


def overlay_s35_review(original_html:str,review_record:dict)->str:
    if review_record.get('schema')!='s35_private_real_h50z_pair_review_evidence_v1':
        raise ValueError('Wrong S35 source evidence schema')
    if review_record.get('physics_auto_join_allowed') is not False or review_record.get('human_confirmed_same_different_pairs')!=0:
        raise ValueError('Unverified evidence cannot assert real physics')
    rows=review_record.get('rows',[])
    if len(rows)!=28 or len({r['relation_id'] for r in rows})!=28:
        raise ValueError('Missing/duplicate original S31 possible link questions')
    m=re.search(r'<script(?=[^>]*id="s31-draft-payload")(?=[^>]*type="application/json")[^>]*>(.*?)</script>',original_html,re.S)
    if not m:raise ValueError('Missing original source S31 machine review JSON')
    source=json.loads(m.group(1))
    expected={r['relation_id']:(p['crop_id'],p['photo_id'],r['candidate_a'],r['candidate_b']) for p in source['photos'] for r in p['relation_drafts']}
    if len(expected)!=28 or source.get('physical_joins_authorized') is not False:
        raise ValueError('S31 source must have exactly 28 provisional pairs')
    for r in rows:
        if expected.get(r['relation_id'])!=(r['crop_id'],r['photo_id'],r['candidate_a'],r['candidate_b']):
            raise ValueError('Foreign/mismatched source physical relation')
        if r['status']!='UNKNOWN_REVIEW_REQUIRED' or r['physics_auto_join_allowed'] is not False or r['human_expert_label'] is not None:
            raise ValueError('Unsafe automatic relation labelling')
    if 'id="s35-review"' in original_html:raise ValueError('Duplicate S35 browser overlay')
    if '</body></html>' not in original_html:raise ValueError('Unexpected original HTML')
    compact={r['relation_id']:{'crop_id':r['crop_id'],
        'scores':r['provisional_scores_NOT_PHYSICAL_PROBABILITIES'],
        'disagreement':r['local_vs_ANZA_score_disagreement'],
        'gap':r['measured_endpoint_gap_px'],
        'ta':r['outgoing_tangent_a'],'tb':r['outgoing_tangent_b']} for r in rows}
    payload=json.dumps(compact,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    addon='''<section id="s35-review" style="border:1px solid #aa9a70;border-radius:8px;padding:12px;margin:14px 0">
<h3 style="margin-top:0">S35 · контроль возможных соединений фрагментов</h3>
<p>Это <strong>не подтверждения физических треков</strong>. Числа получены на искусственных разрывах и не откалиброваны для настоящих фотографий. Неизвестные связи должны оставаться для ручного решения.</p>
<div><button id="s35-next-uncertain">Следующая непроверенная спорная пара</button></div>
<div id="s35-link-status" aria-live="polite">Выберите пару фрагментов в оригинальном редакторе.</div>
</section>
<script id="s35-unverified-source" type="application/json">__S35_PAYLOAD__</script>
<script>
'use strict';
(function(){
 const info=JSON.parse(document.getElementById('s35-unverified-source').textContent);
 const ids=Object.keys(info).sort((a,b)=>info[b].disagreement-info[a].disagreement);
 const original=document.getElementById('s31-panel');
 if(!original)throw Error('Original S31 review panel missing');
 original.insertAdjacentElement('afterend',document.getElementById('s35-review'));
 function render(){
  const id=typeof s31RelationSelected==='string'?s31RelationSelected:null;
  const row=info[id],out=document.getElementById('s35-link-status');
  if(!row){out.textContent='Для выбранной пары S35 пока нет оценки. Все связи остаются непроверенными.';return;}
  const s=row.scores;
  const geo=s['geometry_UNVERIFIED_SYNTHETIC_SCORER'];
  const loc=s['geometry+local_UNVERIFIED_SYNTHETIC_SCORER'];
  const az=s['geometry+AZUNet_UNVERIFIED_SYNTHETIC_SCORER'];
  const state=s31RelationDecisions[id]?.status||'UNKNOWN_REVIEW_REQUIRED';
  out.textContent='Пара '+id.slice(-8)+' | gap '+row.gap.toFixed(1)+' px | геометрия '+geo.toFixed(3)+
    ' | локальный контраст '+loc.toFixed(3)+' | ANZA '+az.toFixed(3)+
    ' | разногласие '+row.disagreement.toFixed(3)+' | локальное решение '+state+
    '. Оценки НЕ являются вероятностями одной частицы.';
 }
 const previous=window.s31Render;
 if(typeof previous==='function')window.s31Render=(...args)=>{const response=previous(...args);render();return response;};
 ['s31-relation','crop'].forEach(id=>{let el=document.getElementById(id);if(el)el.addEventListener('change',render);});
 ['s32-next-pair','s35-next-uncertain','s31-rel-same','s31-rel-diff','s31-rel-unsure','s31-rel-clear'].forEach(id=>{
  let el=document.getElementById(id);if(el)el.addEventListener('click',()=>queueMicrotask(render));
 });
 document.getElementById('s35-next-uncertain').addEventListener('click',()=>{
  const item=ids.find(id=>!s31RelationDecisions[id]);
  if(!item){document.getElementById('s35-link-status').textContent='Все 28 пар имеют предварительное локальное решение. Подтверждение физиков ещё требуется.';return;}
  let i=entries.findIndex(p=>p.crop_id===info[item].crop_id);
  if(i<0)throw Error('S35 source crop does not exist in S31');
  active=i;document.getElementById('crop').value=String(i);
  update();s31RelationSelected=item;window.s31Render();
 });
 render();window.s35RefreshEvidence=render;
})();
</script>'''
    return original_html.replace('</body></html>',addon.replace('__S35_PAYLOAD__',payload)+'</body></html>',1)
