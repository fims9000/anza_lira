#!/usr/bin/env python3
"""Offline full-resolution reviewer for S112 points and classical LSD segments.

All photos, annotations, tile coordinates, and exported operator decisions are
PRIVATE. This script is NOT trained ANZA, expert ground truth, or particle IDs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bubble_chamber.bitmap_tiles import GrayBMP8, sha256_file
from scripts.research.bubble_chamber.s113_fullframe_qa import load_keypoints


def build(bmp_path:Path, s112_dir:Path, out:Path, labels_path:Path|None=None,
          max_linelets:int=16, max_tiles:int|None=None):
    """Review saved S112 tile proposals against their exact original BMP pixels."""
    out=Path(out)
    if out.resolve()==ROOT.resolve() or ROOT.resolve() in out.resolve().parents:
        raise ValueError('Private H50Z pixels must stay outside public checkout')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Review output must be a new empty private directory')
    bmp=GrayBMP8(bmp_path)
    raw=(s112_dir/'summary_PRIVATE.json').read_bytes()
    meta=json.loads(raw)
    source_sha=sha256_file(bmp_path)
    if meta['source_image_sha256']!=source_sha or meta['frame_size_wh']!=[bmp.width,bmp.height]:
        raise ValueError('S112 and BMP source provenance mismatch')
    if not meta['full_frame_tiled_pixel_coverage']:
        raise ValueError('Input S112 output has incomplete frame coverage')
    entries=meta['tiles']
    rows,cols=meta['grid_rows_cols']
    if len(entries)!=meta['tiles_processed'] or len(entries)!=rows*cols:
        raise ValueError('Missing or duplicate S112 image tiles')
    if max_tiles is not None and max_tiles<1: raise ValueError('Invalid max_tiles')
    selected=entries if max_tiles is None else entries[:max_tiles]
    names=set()
    digest=sha256(raw)
    for item in entries:
        row,col=item['row'],item['column']
        key=(row,col)
        if key in names:raise ValueError('Duplicate tile')
        names.add(key)
        name=f'candidates_R{row:02d}_C{col:02d}.json'
        if item['candidate_coordinates_private']!=name:raise ValueError('Unsafe candidate name')
        digest.update(name.encode())
        digest.update((s112_dir/name).read_bytes())
    if len(names)!=rows*cols:raise ValueError('Incomplete S112 grid')
    marks=load_keypoints(labels_path,bmp_path.name,bmp.width,bmp.height)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD) if max_linelets else None
    out.mkdir(parents=True,exist_ok=True)
    tiles=[]
    for item in selected:
        row,col=item['row'],item['column']
        x0,y0,x1,y1=map(int,item['xyxy_full'])
        size=x1-x0
        if size<64 or size!=y1-y0:raise ValueError('Invalid square ROI')
        gray,where=bmp.crop(((x0+x1)/2,(y0+y1)/2),size)
        if list(where.original_xyxy)!=[x0,y0,x1,y1]:
            raise ValueError('S112 source tile origin mismatch')
        img=f'R{row:02d}_C{col:02d}.jpg'
        if not cv2.imwrite(str(out/img),gray,[cv2.IMWRITE_JPEG_QUALITY,88]):
            raise OSError('Could not save preview')
        points=json.loads((s112_dir/item['candidate_coordinates_private']).read_text())
        if list(points['roi_xyxy_full'])!=[x0,y0,x1,y1]:
            raise ValueError('S112 candidate file belongs to another source ROI')
        beads=[]
        for p in points['proposals']:
            x,y=float(p['x_local']),float(p['y_local'])
            if not (0<=x<size and 0<=y<size):
                raise ValueError('S112 point outside ROI')
            if abs(x+x0-p['x_full'])>.015 or abs(y+y0-p['y_full'])>.015:
                raise ValueError('S112 point incorrectly transformed to full-frame coordinates')
            beads.append([round(x,3),round(y,3),round(float(p['score_uncalibrated']),4)])
        linelets=[]
        if lsd is not None:
            rawlines=lsd.detect(gray)[0]
            if rawlines is not None:
                lines=rawlines.reshape(-1,4)
                lines=sorted(lines,key=lambda a:-float(np.hypot(a[0]-a[2],a[1]-a[3])))
                linelets=[[round(float(k),2) for k in v] for v in lines[:max_linelets]]
        operator=[]
        for m in marks:
            x=float(m['x_px'])-x0;y=float(m['y_px'])-y0
            if 0<=x<size and 0<=y<size:
                operator.append([x,y,m['label']])
        tiles.append({'row':row,'col':col,'image':img,'xyxy':[x0,y0,x1,y1],
                      'beads':beads,'linelets':linelets,'operator':operator})
    audit={'schema':'PRIVATE_S113_FULL_TILE_REVIEW_V1',
           'source_sha256':source_sha,'s112_evidence_sha256':digest.hexdigest(),
           'operator_labels_sha256':sha256_file(labels_path) if labels_path else None,
           'source_name':bmp_path.name,'frame_wh':[bmp.width,bmp.height],
           'is_full_frame_review':len(tiles)==len(entries),
           'model':'UNSUPERVISED_S112_PHOTO_CANDIDATES_AND_CLASSICAL_LSD',
           'not_physical_track_GT':True,
           'geometric_AB_not_particle_direction':True,
           'tiles':tiles}
    (out/'evidence_PRIVATE.json').write_text(json.dumps(audit,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    (out/'index.html').write_text(render_html(audit),encoding='utf8')
    return {'tiles':len(tiles),'full_frame':len(tiles)==len(entries),
            'bead_candidates':sum(len(t['beads']) for t in tiles),
            'LSD_linelets':sum(len(t['linelets']) for t in tiles),
            'operator_marks':sum(len(t['operator']) for t in tiles),
            'source_sha256':source_sha}


def render_html(audit):
    data=json.dumps(audit,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>S113 private review</title>
<style>body{background:#181a20;color:#e2e6ee;font:14px system-ui}input,select,button{margin:5px;background:#303743;color:white;padding:5px}#canvas{position:relative;width:min(94vw,1000px);aspect-ratio:1}#canvas img,#canvas svg{position:absolute;top:0;left:0;width:100%;height:100%}p{max-width:1000px}</style>
<h2>H50Z: SOURCE / points / linelets / fiducials</h2>
<p>Точки S112 и LSD: НЕ физический GT. A/B — только геометрические концы, НЕ направление частицы. Розовые f1–f8 / зелёный start имеют операторское происхождение. Все выводы приватны.</p>
<select id="tile"></select><button id="prev">Предыдущая</button><button id="next">Следующая</button>
<label>Порог <input id="threshold" type="range" min=".78" max="1" step=".01" value=".78"></label><output id="th"></output>
<label><input id="beads" type="checkbox" checked>точки</label><label><input id="lsd" type="checkbox" checked>LSD A/B</label><label><input id="marks" type="checkbox" checked>f/start</label>
<button id="export">Экспорт решений</button><p id="detail"></p>
<div id="canvas"><img id="source"><svg id="svg"></svg></div>
<p>Нажатие на точку: UNKNOWN → ACCEPT → REJECT → UNKNOWN. Это черновые решения по машинным кандидатам, без автоматического обучения.</p>
<script id="evidence" type="application/json">'''+data+'''</script><script>
const D=JSON.parse(document.getElementById('evidence').textContent),$=x=>document.getElementById(x);
const state={},selector=$('tile'),svg=$('svg'),ns='http://www.w3.org/2000/svg';
D.tiles.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.text='R'+t.row+' C'+t.col+' — '+t.beads.length+' points';selector.add(o)});
const node=(tag,attrs)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e};
function render(){const t=D.tiles[Number(selector.value)],k=t.image,sz=t.xyxy[2]-t.xyxy[0];$('source').src=k;svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+sz+' '+sz);
$('th').textContent=Number($('threshold').value).toFixed(2);
if($('lsd').checked)t.linelets.forEach((v,j)=>{svg.appendChild(node('line',{x1:v[0],y1:v[1],x2:v[2],y2:v[3],stroke:'#35c9fc','stroke-width':1.3}));if(j<8){svg.appendChild(node('circle',{cx:v[0],cy:v[1],r:4,fill:'#42ef73'}));svg.appendChild(node('rect',{x:v[2]-3,y:v[3]-3,width:6,height:6,fill:'#ff4949'}))}});
if($('beads').checked)t.beads.forEach((v,j)=>{if(v[2]<Number($('threshold').value))return;const id=k+'#'+j,st=state[id]||'unknown';const c=node('circle',{cx:v[0],cy:v[1],r:3,fill:st==='accept'?'#30ee66':st==='reject'?'#ff4466':'#ffaa16'});c.addEventListener('click',()=>{state[id]=st==='unknown'?'accept':st==='accept'?'reject':'unknown';render()});svg.appendChild(c)});
if($('marks').checked)t.operator.forEach(p=>{svg.appendChild(node('circle',{cx:p[0],cy:p[1],r:15,stroke:p[2]==='start'?'lime':'magenta',fill:'none','stroke-width':2}));const x=node('text',{x:p[0]+17,y:p[1]-10,fill:'white','font-size':18,stroke:'black','stroke-width':3,'paint-order':'stroke'});x.textContent=p[2];svg.appendChild(x)});
$('detail').textContent='Full-frame xyxy: '+t.xyxy+' | S112 points: '+t.beads.length+' | LSD: '+t.linelets.length+' | physical ID: UNKNOWN'}
selector.onchange=render;$('prev').onclick=()=>{selector.value=Math.max(0,Number(selector.value)-1);render()};$('next').onclick=()=>{selector.value=Math.min(D.tiles.length-1,Number(selector.value)+1);render()};
['threshold','beads','lsd','marks'].forEach(k=>$(k).oninput=render);
$('export').onclick=()=>{const obj={schema:'S113_REVIEW_DECISIONS_V1',source_sha256:D.source_sha256,s112_evidence_sha256:D.s112_evidence_sha256,operator_labels_sha256:D.operator_labels_sha256,decisions:state};const url=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='PRIVATE_S113_review.json';a.click();URL.revokeObjectURL(url)};
selector.value=0;render();
</script></html>'''


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--bmp',required=True,type=Path)
    ap.add_argument('--s112',required=True,type=Path)
    ap.add_argument('--labels',type=Path)
    ap.add_argument('--private-out',required=True,type=Path)
    ap.add_argument('--max-linelets',type=int,default=16)
    ap.add_argument('--max-tiles',type=int)
    opts=ap.parse_args()
    print(json.dumps(build(opts.bmp,opts.s112,opts.private_out,opts.labels,opts.max_linelets,opts.max_tiles),indent=2,ensure_ascii=False))
