"""Produce true cubic vector curve hypotheses from S105/S108 tracklet points.

Keep original single-frame pixel evidence and inferred geometry separated.
S109 is not an automatic particle identification solution and requires expert
review of crossing/branch associations.
"""
from __future__ import annotations
import argparse,hashlib,html,json,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np

BASE=Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
from bubble_chamber.s109_vector_tracks import VectorConfig,vectorize,tentative_links


def color_for(identifier):
    # Reproducible distinct highly visible hues on gray originals.
    h=int(hashlib.sha256(identifier.encode()).hexdigest()[:8],16)%180
    pixel=cv2.cvtColor(np.uint8([[[h,225,235]]]),cv2.COLOR_HSV2RGB)[0,0]
    return tuple(map(int,pixel))


def points_for(source,report):
    x0,y0,*_=report['original_roi_source_coordinates']
    if 'source_points_xy' in source:
        data=np.asarray(source['source_points_xy'],float)
        xy=data[:,:2]-np.array([x0,y0])
    elif 'points_xy_local_observed' in source:
        data=np.asarray(source['points_xy_local_observed'],float)
        xy=data[:,:2]
    else:
        raise ValueError('unsupported source point representation')
    return xy[data[:,2]>.5]


def make_svg(curves, links, gray_name='SOURCE_GRAYSCALE.png'):
    header=['<?xml version="1.0" encoding="UTF-8"?>',
      '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1200 1200" width="1200" height="1200">',
      '<image href="'+html.escape(gray_name)+'" x="0" y="0" width="1200" height="1200"/>',
      '<g fill="none" stroke-linecap="round" stroke-linejoin="round">']
    for c in curves:
        color='#%02x%02x%02x'%color_for(c['curve_id'])
        label=html.escape(c['curve_id'])
        path=c['svg_path']
        if not path:continue
        # Full mathematical cubic is a hypothesis, dashed unless existing
        # pixel evidence supports an individual sampled segment.
        header.append(f'<g data-track-id="{label}" data-status="{c["status"]}">')
        candidate_status='supported' if c['source_anchored'] else 'uncertain'
        support_frac=c['photo_support']['supported_sample_fraction']
        header.append(f'<title>{label}: {candidate_status}; photo fraction {support_frac:.2f}</title>')
        header.append(f'<path d="{path}" stroke="{color}" opacity="0.40" stroke-width="1.2" stroke-dasharray="4,5"/>')
        xy=np.asarray(c['sampled_xy'])
        supported=np.asarray(c['sampled_source_supported'],bool)
        if not c['source_anchored']:supported[:]=False
        runs=[];start=None
        for j,yes in enumerate(supported):
            if yes and start is None:start=j
            if start is not None and (not yes or j==len(supported)-1):
                end=j+1 if yes else j
                if end-start>=3:runs.append(xy[max(0,start-1):min(len(xy),end+1)])
                start=None
        for q in runs:
            ps=' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in q)
            header.append(f'<polyline points="{ps}" stroke="{color}" stroke-width="1.65" opacity=".92"/>')
        header.append('</g>')
    header.append('</g>')
    header.append('<g data-link-proposal="true" fill="none" stroke="#FFE846" stroke-width="1.1" stroke-dasharray="3,5" opacity="0.8">')
    for e in links:
        q=e['inferred_link_xy']
        header.append('<polyline points="'+' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in q)+'"/>')
    header.append('</g></svg>')
    return '\n'.join(header)+'\n'


def make_review_html(svg, case, source_sha, curves):
    # SVG is generated in this tool from model-owned numeric paths, not from
    # untrusted HTML. All reviewed decisions require explicit user clicks.
    raw=svg.replace('<?xml version="1.0" encoding="UTF-8"?>','',1)
    return """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>S109 — экспертная проверка векторных треков</title>
<style>
:root{font-family:Inter,Arial,sans-serif;color:#152030;background:#eff1f5}body{margin:0}
header{padding:14px 22px;background:white;border-bottom:1px solid #ccd3df;position:sticky;top:0;z-index:3}
h2{margin:0 0 7px;font-size:21px}button{padding:8px 13px;border:1px solid #98a5b5;border-radius:7px;cursor:pointer;background:white}
button:disabled{opacity:.45;cursor:not-allowed}button:hover:not(:disabled){background:#e6edfa}
main{display:grid;grid-template-columns:minmax(600px,1200px) minmax(250px,350px);gap:15px;padding:15px}
#canvas{background:white;border:1px solid #ccd3df;border-radius:8px;overflow:auto;max-height:85vh}
#canvas svg{display:block;width:100%;height:auto;min-width:650px}
#canvas g[data-track-id]{cursor:pointer}
#canvas g[data-track-id]:hover > path{stroke-width:3;opacity:.8}
#canvas g[data-track-id].selected > path{stroke:#ffb800;stroke-width:3;opacity:1}
#canvas g[data-track-id].selected polyline{stroke:#ffb800;stroke-width:3;opacity:1}
#canvas.hide-inferred g[data-track-id] > path{display:none} #canvas.hide-inferred g[data-link-proposal]{display:none}
#canvas.hide-inferred g[fill="none"] > polyline{opacity:1}
#canvas.hide-photo image{display:none}
aside{background:white;border:1px solid #ccd3df;border-radius:8px;padding:18px;align-self:start}
#selected{font-size:12px;overflow-wrap:anywhere;line-height:1.6}
small{color:#576477}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
label{margin-right:18px;font-size:13px}
</style></head><body><header><h2>Векторные гипотезы S109 — проверка связей</h2>
<div><label><input type="checkbox" id="onlyphoto"/> Скрыть предположительные участки</label>
<label><input type="checkbox" id="hidephoto"/> Скрыть фотографию</label>
<button id="download">Экспорт решений JSON</button></div></header>
<main><div id="canvas">"""+raw+"""</div>
<aside><h3>Два фрагмента одного трека?</h3>
<p><small>Выберите на изображении две цветные кривые. Одинаковый цвет не означает одинаковую частицу. Пересекающиеся треки могут относиться к разным частицам.</small></p>
<div id="selected">Пока ничего не выбрано</div><div class="actions">
<button data-judgment="same">Один трек</button><button data-judgment="different">Разные треки</button>
<button data-judgment="unknown">Неясно</button><button id="clear">Очистить выбор</button></div>
<p id="count">Решений: 0</p>
<small>Синие/цветные сплошные участки имеют поддержку фотомаски. Пунктир — интерполяция, НЕ наблюдение. Ни одна траектория пока не получила проверенный физический ID.</small>
</aside></main><script>
const run="""+json.dumps(case)+""",sha="""+json.dumps(source_sha)+""";
const selected=[];const decisions=[];const panel=document.querySelector('#canvas');
function refresh(){document.querySelector('#selected').textContent=selected.length?selected.join(' и '):'Пока ничего не выбрано';
 document.querySelectorAll('[data-track-id]').forEach(n=>n.classList.toggle('selected',selected.includes(n.dataset.trackId)));
 document.querySelectorAll('button[data-judgment]').forEach(n=>n.disabled=selected.length!==2);document.querySelector('#count').textContent='Решений: '+decisions.length}
document.querySelectorAll('g[data-track-id]').forEach(el=>el.addEventListener('click',evt=>{evt.preventDefault();
 const id=el.dataset.trackId;let idx=selected.indexOf(id);
 if(idx>=0)selected.splice(idx,1);else{if(selected.length>=2)selected.shift();selected.push(id)}refresh()}));
document.querySelectorAll('button[data-judgment]').forEach(el=>el.onclick=()=>{if(selected.length!==2)return;
 decisions.push({track_ids:[...selected].sort(),judgment:el.dataset.judgment,reviewer:'manual',
  source_sha256:sha,roi:run});selected.length=0;refresh()});
document.querySelector('#clear').onclick=()=>{selected.length=0;refresh()};
document.querySelector('#onlyphoto').onchange=e=>panel.classList.toggle('hide-inferred',e.target.checked);
document.querySelector('#hidephoto').onchange=e=>panel.classList.toggle('hide-photo',e.target.checked);
document.querySelector('#download').onclick=()=>{const obj={schema:'S109_EXPERT_PAIR_REVIEW_V1',
  roi:run,source_sha256:sha,decisions,labels_not_for_unreviewed_pixels:true};
 const data=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});const url=URL.createObjectURL(data);
 const a=document.createElement('a');a.href=url;a.download=run+'_expert_pairs.json';a.click();URL.revokeObjectURL(url)};
refresh();
</script></body></html>"""


def main_case(root:Path,case:str,outroot:Path,cfg:VectorConfig=VectorConfig()):
    source=root/'results'/case
    s108=root/'s108_results_all'/case
    report=json.loads((s108/'report.json').read_text())
    photo=cv2.imread(str(s108/'01_S88_SOURCE_RIDGE_ONLY.png'),cv2.IMREAD_GRAYSCALE)
    gray=cv2.imread(str(s108/'06_SOURCE_ORIGINAL_GRAY.png'),cv2.IMREAD_GRAYSCALE)
    blocked=cv2.imread(str(s108/'07_BLOCKED_SOURCE.png'),cv2.IMREAD_GRAYSCALE)
    if any(z is None for z in (photo,gray,blocked)):raise FileNotFoundError(case)
    photo=photo>0;blocked=blocked>0
    paths=json.loads((source/'trace_hypotheses.json').read_text())+json.loads((s108/'STRICT_CURVED_SOURCE_CANDIDATES_NOT_PHYSICAL_IDS.json').read_text())
    curves=[];fit_audits=[]
    for p in paths:
        pts=points_for(p,report)
        z,a=vectorize(p['visual_track_id'],pts,photo,blocked,cfg,gray=gray)
        curves+=z
        fit_audits.append({'path_id':p['visual_track_id'],'pieces':len(z),**a})
    links,links_report=tentative_links(curves,photo,blocked,cfg)
    od=outroot/case;od.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(od/'SOURCE_GRAYSCALE.png'),gray)
    svg=make_svg(curves,links)
    (od/'vector_tracks.svg').write_text(svg,encoding='utf-8')
    (od/'expert_vector_review.html').write_text(make_review_html(svg,case,report['source_sha256'],curves),encoding='utf-8')
    (od/'vector_tracks.json').write_text(json.dumps({
        'schema':'S109_CUBIC_BSPLINE_VISUAL_HYPOTHESES_NOT_PHYSICAL_IDS_V1',
        'source_roi':case,'source_sha256':report['source_sha256'],
        'original_roi_source_coordinates':report['original_roi_source_coordinates'],
        'curves':curves,'proposed_links_for_review':links},ensure_ascii=False),encoding='utf-8')
    output=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    good=sorted((c for c in curves if c['source_anchored']),key=lambda z:-z['spline_arc_length_px'])
    for c in good:
        rr,gg,bb=color_for(c['curve_id'])
        pts=np.asarray(c['sampled_xy'],float)
        solid=np.asarray(c['sampled_source_supported'],bool)
        draw=np.rint(pts).astype(np.int32)
        for i in range(1,len(draw)):
            if solid[i] and solid[i-1]:
                cv2.line(output,tuple(draw[i-1]),tuple(draw[i]),(bb,gg,rr),1,cv2.LINE_AA)
    cv2.imwrite(str(od/'VECTOR_SOURCE_BACKED_COLOR.png'),output)
    view=output.copy()
    for c in good:
        rr,gg,bb=color_for(c['curve_id'])
        coords=np.rint(np.asarray(c['sampled_xy'])).astype(np.int32)
        supported=np.asarray(c['sampled_source_supported'],bool)
        # Dashed geometry when observations do NOT support the spline.
        for j in range(1,len(coords)):
            if not (supported[j] and supported[j-1]) and j%7<3:
                cv2.line(view,tuple(coords[j-1]),tuple(coords[j]),(40,190,255),1,cv2.LINE_AA)
    cv2.imwrite(str(od/'VECTOR_WITH_INFERRED_DASHED_YELLOW.png'),view)
    new={'case':case,'input_s105_paths':report['S105_recomputed_curves'],
        'input_s108_paths':report['S108_extra_paths'],
        'candidate_cubic_splines':len(curves),
        'source_supported_curves':len(good),
        'uncertain_or_rejected_curve_fits':len(curves)-len(good),
        'source_supported_total_arc_px':round(sum(c['spline_arc_length_px'] for c in good),1),
        'mean_supported_sample_ratio':round(float(np.mean([c['photo_support']['supported_sample_fraction'] for c in good])) if good else 0,3),
        'candidate_review_links':len(links),
        'tentative_review_links':links_report['tentative_for_review'],
        'physical_particle_identification_claimed':False,
        'B_spline_does_not_infer_correct_track_association':True,
        'input_path_audit':fit_audits,'link_audit':links_report}
    (od/'report.json').write_text(json.dumps(new,indent=2,ensure_ascii=False),encoding='utf-8')
    return new


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BASE)
    parser.add_argument('--out',type=Path,default=BASE/'s109_vector_results')
    parser.add_argument('--cases',nargs='+',default=['005_start3_1200','006_start2_1200'])
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for case in args.cases:
        r=main_case(args.source,case,args.out)
        print(case,{k:r[k] for k in ('candidate_cubic_splines','source_supported_curves','candidate_review_links','tentative_review_links')},flush=True)
        reports.append(r)
    (args.out/'summary.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()