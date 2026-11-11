#!/usr/bin/env python3
"""Global S110 association over immutable S109 source-backed vector fragments.

Produces one ordered composite cubic-Bezier candidate per connected group,
separate inferred connectors, transparent review UI and reproducible diagnostics.
"""
from __future__ import annotations

import argparse,colorsys,html,json,hashlib,sys,shutil
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
from bubble_chamber.s110_global_association import associate,GlobalConfig

CASES=('005_start2_1200','005_start3_1200','005_start5_1200',
       '006_start1_1200','006_start2_1200','006_start3_1200')


def mask(path):
    x=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if x is None:raise FileNotFoundError(path)
    return x>0


def runs(source):
    a=np.asarray(source,bool)
    diff=np.diff(np.r_[False,a,False].astype(np.int8))
    return list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)))


def colors(i):
    h=(i*.61803398875+.025)%1
    rgb=colorsys.hsv_to_rgb(h,.90,.96)
    return '#'+''.join(f'{int(255*v):02x}' for v in rgb)


def frag_visual_paths(c,color):
    q=np.asarray(c['sampled_xy'],float)
    ok=np.asarray(c['sampled_source_supported'],bool)
    res=[]
    # Inferred spline spans must not be shown as PHOTO pixels.
    for yes,startend in ((True,runs(ok)),(False,runs(~ok))):
        for a,b in startend:
            if b-a<2:continue
            pts=' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in q[a:b:2])
            if b-a>2 and (b-a-1)%2:pts+=' '+f'{q[b-1,0]:.2f},{q[b-1,1]:.2f}'
            dash='' if yes else ' stroke-dasharray="4 5"'
            alpha='.92' if yes else '.33'
            res.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" opacity="{alpha}"{dash}/>')
    return '\n'.join(res)


def path_from_beziers_local(spans):
    if not spans:return ''
    f=lambda p:f'{p[0]:.3f},{p[1]:.3f}'
    result='M '+f(spans[0][0])
    for b in spans:result+=' C '+f(b[1])+' '+f(b[2])+' '+f(b[3])
    return result


def build_svg(data,source_gray,include_unmatched=False):
    curves={c['curve_id']:c for c in data['curves']}
    groups=data['association']['tracks']
    W=source_gray.shape[1];H=source_gray.shape[0]
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<image href="SOURCE_GRAYSCALE.png" x="0" y="0" width="{W}" height="{H}"/>',
           '<g fill="none" stroke-linecap="round" stroke-linejoin="round">']
    for i,t in enumerate(groups):
        col=colors(i)
        label=html.escape(t['global_track_id'])
        lines.append(f'<g data-group-id="{label}" data-curves="{len(t["member_curve_ids"])}" class="track">')
        lines.append(f'<title>{label}: {len(t["member_curve_ids"])} fragments; {t["inferred_connector_count"]} inferred joins</title>')
        # One editable cubic SVG path per global track candidate. It includes
        # inferred joins, and is therefore styled VERY faint/dashed. Bold
        # source-backed portions and yellow-coded reviewed joins are drawn in
        # separate layers below; no physical-ID certainty is implied.
        lines.append(f'<path d="{html.escape(t["composite_svg_path"])}" data-global-composite="true" fill="none" stroke="{col}" stroke-width=".6" stroke-dasharray="2 6" opacity=".10"/>')
        for entry in t['ordered_members']:
            c=curves[entry['curve_id']]
            lines.append(f'<g data-curve-id="{html.escape(c["curve_id"])}">')
            # Keep the actual editable cubic spline path; the bolder overlaid
            # polylines indicate which sampled points have source support.
            lines.append(f'<path d="{html.escape(c["svg_path"])}" data-geometry="cubic-B-spline" fill="none" stroke="{col}" stroke-width=".85" stroke-dasharray="3 5" opacity=".20"/>')
            lines.append(frag_visual_paths(c,col))
            lines.append('</g>')
        for b in t['inferred_connectors']:
            lines.append(f'<g data-edge-id="{b["candidate_id"]}" data-status="{b["status"]}">')
            lines.append(f'<path d="{b["connector_svg"]}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="5 5" opacity=".85"/>')
            lines.append('</g>')
        lines.append('</g>')
    if include_unmatched:
        lines.append('<g fill="none" stroke="#bf5c62" stroke-width="1.8" stroke-dasharray="3 7" opacity=".65">')
        for e in data['association']['edge_proposals']:
            if e['decision'] in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME'):continue
            d=path_from_beziers_local([e['bezier_cubic']])
            lines.append(f'<g data-edge-id="{e["candidate_id"]}" class="unmatched"><path d="{d}"/><path d="{d}" stroke-width="12" opacity="0"/></g>')
        lines.append('</g>')
    lines.append('</g></svg>')
    return '\n'.join(lines)


def render_png(data,gray,path):
    out=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    # Saturated track colors, alpha over gray so a dark track remains visible.
    overlay=out.copy();curves={c['curve_id']:c for c in data['curves']}
    for i,t in enumerate(data['association']['tracks']):
        col=colors(i)
        rgb=tuple(int(col[j:j+2],16) for j in (1,3,5))
        bgr=(rgb[2],rgb[1],rgb[0])
        for item in t['ordered_members']:
            c=curves[item['curve_id']]
            q=np.rint(np.asarray(c['sampled_xy'],float)).astype(np.int32)
            ok=np.asarray(c['sampled_source_supported'],bool)
            for a,b in runs(ok):
                if b-a<2:continue
                cv2.polylines(overlay,[q[a:b].reshape(-1,1,2)],False,bgr,2,cv2.LINE_AA)
        for gap in t['inferred_connectors']:
            q=np.rint(np.asarray(gap['sampled_xy'],float)).astype(np.int32)
            # dotted intervals: each isolated line is strictly hypothetical.
            for a in range(0,len(q)-1,14):
                b=min(len(q),a+7)
                if b-a>1:cv2.polylines(overlay,[q[a:b].reshape(-1,1,2)],False,bgr,2,cv2.LINE_AA)
    out=cv2.addWeighted(out,.16,overlay,.84,0)
    cv2.imwrite(str(path),out)


def review_html(data,svg):
    edges=data['association']['edge_proposals']
    # Keep the editor small. Data stays in a JSON file; only review metadata is
    # inlined. Exports can be reopened by --decisions at next run.
    rows=[{'id':e['candidate_id'],'curves':e['curve_ids'],'gap':e['gap_px'],
           'score':e['score'],'photo':e['photo_support_fraction'],'status':e['decision']}
          for e in edges]
    js=json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    body=svg.replace('<svg ','<svg id="map" ',1)
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"/>
<title>S110 — экспертная проверка группировок</title>
<style>body{margin:0;background:#f7f8fa;font:14px Arial,sans-serif;color:#172334}
header{background:white;border-bottom:1px solid #ddd;padding:12px 22px;position:sticky;top:0;z-index:4}
h1{font-size:20px;margin:0 0 6px}main{display:grid;grid-template-columns:minmax(600px,1200px) 340px;gap:15px;padding:14px}
#canvas{border:1px solid #d6dbe4;border-radius:6px;max-height:82vh;overflow:auto;background:white}
#canvas svg{width:100%;min-width:600px;height:auto;display:block}
#canvas g.track:hover{opacity:.8}#canvas g.track.selected{filter:drop-shadow(0 0 2px #f70)}
#canvas g[data-edge-id] path{pointer-events:stroke;cursor:pointer}
#canvas g[data-edge-id].selected path{stroke:#ff8b00;stroke-width:5;opacity:1}
aside{background:white;border:1px solid #ddd;border-radius:7px;padding:15px;overflow:auto;max-height:82vh}
.edge{padding:8px;border-bottom:1px solid #eee;cursor:pointer;font-size:12px}.edge:hover{background:#f2f5fa}.edge.active{background:#fff2d6}
.controls{display:flex;flex-wrap:wrap;gap:7px;padding:10px 0}button{padding:7px 10px;cursor:pointer;border:1px solid #bbc4d1;background:white;border-radius:5px}
#details{white-space:pre-wrap;font-size:12px;line-height:1.6}small{color:#647184}
</style>
<header><h1>S110: глобальные гипотезы треков</h1>
<div>Один цвет = одна <b>алгоритмическая</b> группа. Пунктир = интерполяция, НЕ наблюдаемые пузырьки.
<b>Ни один ID не подтверждён как физическая частица.</b></div>
<button id="export">Сохранить решения JSON</button>
<label>Загрузить решения <input type="file" id="import" accept="application/json"/></label></header>
<main><div id="canvas">__SVG__</div><aside><h3>Связи на проверку</h3>
<div id="selected">Выберите пунктирную связь или строку ниже.</div>
<div class="controls"><button data-v="same">Один трек</button>
<button data-v="different">Разные треки</button><button data-v="uncertain">Неясно</button>
<button id="clear">Сбросить оценку</button></div><small>Оценка НЕ меняет изображение без повторного запуска
`python ...s110_global_matching.py --decisions review_decisions.json`.</small>
<div id="edges"></div></aside></main><script>
const links=__EDGES__;const byid=Object.fromEntries(links.map(x=>[x.id,x]));
const decisions={};let selected=null;
const $=id=>document.getElementById(id);
function draw(){ $('edges').innerHTML=''; for(const e of links){const d=document.createElement('div');
d.className='edge'+(e.id===selected?' active':'');
d.textContent=e.curves.map(x=>x.slice(-13)).join(' + ')+' / '+e.gap.toFixed(1)+' px / '+e.score.toFixed(3)+' / '+(decisions[e.id]||e.status);
d.onclick=()=>pick(e.id);$('edges').appendChild(d);} }
function pick(id){selected=id;document.querySelectorAll('g[data-edge-id]').forEach(n=>n.classList.toggle('selected',n.getAttribute('data-edge-id')===id));
const e=byid[id];$('selected').textContent=e?JSON.stringify({...e,expert_decision:decisions[id]||'unreviewed'},null,2):'Нет выбора';draw()}
document.querySelectorAll('g[data-edge-id]').forEach(el=>el.addEventListener('click',ev=>{ev.stopPropagation();pick(el.getAttribute('data-edge-id'))}));
document.querySelectorAll('button[data-v]').forEach(el=>el.onclick=()=>{if(!selected)return;decisions[selected]=el.dataset.v;pick(selected)});
$('clear').onclick=()=>{if(selected){delete decisions[selected];pick(selected)}};
$('export').onclick=()=>{const data=JSON.stringify({schema:'ANZA_S110_EXPERT_LINK_REVIEW',source_roi:'__ROI__',review_decisions:decisions},null,2);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type:'application/json'}));a.download='s110_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('import').onchange=async ev=>{const f=ev.target.files?.[0];if(!f)return;try{const p=JSON.parse(await f.text());
Object.keys(decisions).forEach(k=>delete decisions[k]);
for(const [k,v] of Object.entries(p.review_decisions||{})){if(byid[k]&&['same','different','uncertain'].includes(v))decisions[k]=v;}
pick(selected);}catch(e){alert('Неверный JSON: '+e.message)}};draw();
</script></html>'''.replace('__SVG__',body).replace('__EDGES__',js).replace('__ROI__',data['source_roi'])


def run_case(root,out,roi,cfg,decisions=None):
    src=root/'s109_vector_results'/roi
    s108=root/'s108_results_all'/roi
    original=json.loads((src/'vector_tracks.json').read_text(encoding='utf-8'))
    original_gray=s108/'06_SOURCE_ORIGINAL_GRAY.png'
    gray=cv2.imread(str(original_gray),cv2.IMREAD_GRAYSCALE)
    mask=cv2.imread(str(s108/'01_S88_SOURCE_RIDGE_ONLY.png'),0)
    blocked=cv2.imread(str(s108/'07_BLOCKED_SOURCE.png'),0)
    if any(x is None for x in (gray,mask,blocked)):raise FileNotFoundError(f'Bad S108 inputs for {roi}')
    association=associate(original['curves'],mask>0,blocked>0,gray,cfg,decisions,original.get('proposed_links_for_review',[]))
    result={'schema':'ANZA_S110_GLOBAL_ASSOCIATION_v1','source_roi':roi,
            's109_sha256':hashlib.sha256((src/'vector_tracks.json').read_bytes()).hexdigest(),
            'source_sha256':original['source_sha256'],'config':dict(vars(cfg)),
            'source_evidence':'S109 cubic vector fragments and S88 ridge pixels',
            'curves':original['curves'],'association':association,
            'physical_particle_identity_proven':False,
            'all_automated_merges_reversible':True}
    d=out/roi;d.mkdir(parents=True,exist_ok=True)
    # For standalone SVG/HTML no external services are required.
    shutil.copy2(original_gray,d/'SOURCE_GRAYSCALE.png')
    svg=build_svg(result,gray)
    (d/'grouped_vector_tracks.svg').write_text(svg,encoding='utf-8')
    (d/'expert_global_review.html').write_text(review_html(result,build_svg(result,gray,include_unmatched=True)),encoding='utf-8')
    (d/'global_tracks.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    render_png(result,gray,d/'GLOBAL_TRACK_GROUPS_WITH_DASHED_LINKS.png')
    rep={'roi':roi,'association':association['association_report'],
         'candidate_generator':association['generator_audit'],'global_solver':association['solver_audit'],
         'not_physical_ground_truth':True}
    (d/'report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
    return rep


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--cases',nargs='*',default=list(CASES),choices=CASES)
    parser.add_argument('--decisions',type=Path,default=None,help='expert review JSON, one ROI or keyed by ROI')
    parser.add_argument('--max-gap',type=float,default=GlobalConfig.max_gap_px)
    args=parser.parse_args()
    out=args.out or args.root/'s110_global_results'
    cfg=GlobalConfig(max_gap_px=float(args.max_gap))
    review={}
    if args.decisions:
        a=json.loads(args.decisions.read_text(encoding='utf-8'))
        review=a.get('review_decisions',a)
        if a.get('source_roi') and len(args.cases)>1:
            review={a['source_roi']:review}
    out.mkdir(parents=True,exist_ok=True)
    reps=[]
    for roi in args.cases:
        decisions=review.get(roi,{}) if isinstance(review,dict) and any(k in CASES for k in review) else review
        # Single-ROI review JSON is edge_id: status; multi-ROI JSON is
        # roi: {edge_id: status} and is applied independently per image.
        decisions=decisions if isinstance(decisions,dict) else {}
        rep=run_case(args.root,out,roi,cfg,decisions)
        reps.append(rep)
        print(roi,rep['association'],flush=True)
    (out/'summary.json').write_text(json.dumps(reps,ensure_ascii=False,indent=2),encoding='utf-8')
    return reps

if __name__=='__main__':main()