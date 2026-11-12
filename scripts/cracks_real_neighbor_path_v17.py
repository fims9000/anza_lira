#!/usr/bin/env python3
"""Real CRACKS v2: conditional bridge-geometry test using genuine adjacent F3 images.

The expert mask is used to select unambiguous lines, simulate a missing segment,
expose surviving endpoints, and score prediction AFTER selection. Hidden expert
pixels are not used by scoring. NOT an ANZA/LIRA learned-model E2E evaluation.
There are no defensible false-bridge negatives in this pilot.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scripts.cracks_real_oracle_gap_audit import paths_from_mask, sections, COLORS, split

OFFSETS=(-5.,-3.,0.,3.,5.)

def gray(archive:zipfile.ZipFile,name:str)->np.ndarray:
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.asarray(im.convert('L'),dtype=np.float32)/255.

def rgb(archive:zipfile.ZipFile,name:str)->np.ndarray:
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.asarray(im.convert('RGB'),dtype=np.uint8)

def gradient(image:np.ndarray)->np.ndarray:
    """Frozen generic horizontal-derivative proxy, NOT a trained fault detector."""
    g=np.abs(ndi.sobel(ndi.gaussian_filter(image.astype(np.float32),1.),axis=1))
    return g/(float(np.percentile(g,95))+1.e-6)

def curves(end0:tuple[int,int],end1:tuple[int,int],region:tuple[int,int,int,int]):
    """Five label-independent connecting curves with identical visible endpoints."""
    y0,x0=np.asarray(end0,float);y1,x1=np.asarray(end1,float)
    dy,dx=y1-y0,x1-x0
    length=float(np.hypot(dy,dx))
    if length<6:raise ValueError('anchors too close')
    t=np.linspace(0.,1.,max(20,int(3*max(abs(dx),abs(dy)))))
    y=y0+t*dy;x=x0+t*dx
    ymin,ymax,xmin,xmax=region
    m=(y>=ymin)&(y<ymax)&(x>=xmin)&(x<xmax)
    if m.sum()<8:raise ValueError('insufficient gap pixels')
    ny,nx=-dx/length,dy/length
    return [np.stack([y[m]+a*np.sin(np.pi*t[m])*ny,
                      x[m]+a*np.sin(np.pi*t[m])*nx]) for a in OFFSETS]

def candidate_scores(paths:list[np.ndarray],images:list[np.ndarray]):
    """Pure scoring: no expert labels, hidden paths or truth argument."""
    return np.asarray([[float(np.mean(ndi.map_coordinates(g,p,order=1,mode='nearest')))
                         for g in images] for p in paths],float)

def choose(scored:np.ndarray,method:str)->int:
    """Frozen choice: no threshold or rule adapted to real expert TEST."""
    if method=='straight':return OFFSETS.index(0.)
    if method=='central':v=scored[:,0]
    elif method=='adjacent':v=scored[:,1:3].mean(axis=1)
    elif method=='central_adjacent':v=scored[:,:3].mean(axis=1)
    elif method=='central_shuffled':v=np.stack(
        (scored[:,0],scored[:,3],scored[:,4]),axis=1).mean(axis=1)
    else:raise ValueError(method)
    return int(np.argmax(v))

def distance_to_expert(path:np.ndarray,expert:np.ndarray,dist=None):
    if dist is None:dist=ndi.distance_transform_edt(~expert)
    values=ndi.map_coordinates(dist,path,order=1,mode='nearest')
    return float(values.mean()),float((values<=2.).mean())

def normalized_corr(a:np.ndarray,b:np.ndarray)->float:
    x=a.ravel()[::31];y=b.ravel()[::31]
    if len(x)!=len(y) or np.std(x)<1e-8 or np.std(y)<1e-8:return float('nan')
    return float(np.corrcoef(x,y)[0,1])

def enumerate_gaps(sid:int,label:np.ndarray):
    confident=np.all(label==np.array(COLORS['blue'],dtype=np.uint8),axis=2)
    uncertain=np.all(label==np.array(COLORS['green'],dtype=np.uint8),axis=2)
    clean=confident & ~ndi.binary_dilation(uncertain,iterations=5)
    count=0
    for path in paths_from_mask(clean):
        if len(path)<60:continue
        mid=len(path)//2;middle=np.asarray(path[mid-6:mid+6])
        ymin,ymax=int(middle[:,0].min()-3),int(middle[:,0].max()+4)
        xmin,xmax=int(middle[:,1].min()-3),int(middle[:,1].max()+4)
        if ymin<0 or xmin<0 or ymax>clean.shape[0] or xmax>clean.shape[1]:continue
        if uncertain[ymin:ymax,xmin:xmax].any():continue
        full_line=np.zeros_like(clean)
        for point in path:full_line[point]=True
        support=ndi.binary_dilation(full_line,iterations=5)
        if (confident[ymin:ymax,xmin:xmax] & ~support[ymin:ymax,xmin:xmax]).any():continue
        inside=lambda p:ymin<=p[0]<ymax and xmin<=p[1]<xmax
        e0=next((p for p in reversed(path[:mid]) if not inside(p)),None)
        e1=next((p for p in path[mid:] if not inside(p)),None)
        if e0 is None or e1 is None:continue
        visible=clean.copy();visible[ymin:ymax,xmin:xmax]=False
        comp,_=ndi.label(visible,structure=np.ones((3,3),np.uint8))
        if comp[e0]==0 or comp[e1]==0 or comp[e0]==comp[e1]:continue
        gt=np.zeros_like(clean)
        for p in path:
            if inside(p):gt[p]=True
        if int(gt.sum())<5:continue
        try:candidates=curves(e0,e1,(ymin,ymax,xmin,xmax))
        except ValueError:continue
        yield count,candidates,gt,(ymin,ymax,xmin,xmax),e0,e1
        count+=1
        if count>=2:break

def audit(img_zip:Path,label_zip:Path,out:Path,csv_out:Path,max_sections:int)->dict:
    rows=[];alignment=[];counts=Counter()
    with zipfile.ZipFile(img_zip) as zi,zipfile.ZipFile(label_zip) as zl:
        ims=sections(zi);experts=sections(zl,expert=True)
        known=sorted(set(ims)&set(experts))[:max_sections]
        if not known:raise RuntimeError('no matching expert masks')
        gray_cache={};grad_cache={}
        def get_img(sid):
            if sid not in gray_cache:gray_cache[sid]=gray(zi,ims[sid])
            return gray_cache[sid]
        def get_grad(sid):
            if sid not in grad_cache:grad_cache[sid]=gradient(get_img(sid))
            return grad_cache[sid]
        all_ids=sorted(ims)
        for sid in known:
            label=rgb(zl,experts[sid]);central=get_img(sid)
            if label.shape[:2]!=central.shape:raise ValueError(f'section {sid} shape mismatch')
            if sid-1 not in ims or sid+1 not in ims:
                counts['missing_immediate_neighbors']+=1;continue
            if get_img(sid-1).shape!=central.shape or get_img(sid+1).shape!=central.shape:
                counts['unaligned_dimensions']+=1;continue
            far1=all_ids[(all_ids.index(sid)+107)%len(all_ids)]
            far2=all_ids[(all_ids.index(sid)+211)%len(all_ids)]
            if abs(far1-sid)<20 or abs(far2-sid)<20:
                raise AssertionError('sham control not sufficiently distant')
            alignment.append({'section':sid,
                'near_minus':normalized_corr(central,get_img(sid-1)),
                'near_plus':normalized_corr(central,get_img(sid+1)),
                'far_1':normalized_corr(central,get_img(far1)),
                'far_2':normalized_corr(central,get_img(far2))})
            grads=[get_grad(t) for t in (sid,sid-1,sid+1,far1,far2)]
            region_count=0
            for index,candidates,gt,region,e0,e1 in enumerate_gaps(sid,label):
                scores=candidate_scores(candidates,grads)
                dmap=ndi.distance_transform_edt(~gt)
                losses=np.asarray([distance_to_expert(p,gt,dmap)[0] for p in candidates])
                best=int(np.argmin(losses)) # evaluation-only oracle choice
                chosen={m:choose(scores,m) for m in (
                    'straight','central','adjacent','central_adjacent','central_shuffled')}
                region_count+=1;sp,block=split(sid)
                row={'section':sid,'block':block,'split':sp,'gap_index':index,
                     'region':json.dumps(region),'e0':json.dumps(e0),'e1':json.dumps(e1),
                     'oracle_best_offset':OFFSETS[best],
                     'oracle_best_distance':float(losses[best]),
                     'score_sha256':hashlib.sha256(np.round(scores,8).tobytes()).hexdigest()}
                for name,ind in chosen.items():
                    distance,precision=distance_to_expert(candidates[ind],gt,dmap)
                    row[f'{name}_offset']=OFFSETS[ind]
                    row[f'{name}_distance']=distance
                    row[f'{name}_precision_tol2']=precision
                    row[f'{name}_oracle_match']=int(ind==best)
                rows.append(row)
            counts['expert_sections_visited']+=1
            counts['valid_gaps']+=region_count
    if not rows:raise RuntimeError('zero eligible expert gaps')
    csv_out.parent.mkdir(parents=True,exist_ok=True)
    with csv_out.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    methods=('straight','central','adjacent','central_adjacent','central_shuffled')
    def summarize(arr):
        return {'N':len(arr),'methods':{m:{
            'mean_distance':float(np.mean([r[f'{m}_distance'] for r in arr])),
            'mean_precision_tol2':float(np.mean([r[f'{m}_precision_tol2'] for r in arr])),
            'oracle_best_choice_rate':float(np.mean([r[f'{m}_oracle_match'] for r in arr]))}
            for m in methods}}
    splits={s:summarize([r for r in rows if r['split']==s]) for s in
            ('development','calibration','test') if any(r['split']==s for r in rows)}
    result={'dataset':'CRACKS v2 DOI 10.5281/zenodo.13926822',
        'protocol':'CRACKS_V17_REAL_ADJACENT_GRADIENT_CONDITIONAL_PATH_EXPLORATORY',
        'experiment':'real image gradients, oracle-selected surviving endpoints',
        'is_full_ANZA_SBPP_LIRA_E2E':False,
        'uses_expert_hidden_gap_for_inference':False,
        'oracle_surviving_pair':True,
        'false_bridge_negatives_available':False,
        'claim':'frozen unsupervised conditional path-scoring pilot; NOT learned segmentation',
        'N_expert_sections':len(set(r['section'] for r in rows)),
        'N_gaps':len(rows),'counts':dict(counts),
        'alignment':{'N':len(alignment),
            'mean_near_corr':float(np.nanmean([.5*(r['near_minus']+r['near_plus']) for r in alignment])),
            'mean_far_corr':float(np.nanmean([.5*(r['far_1']+r['far_2']) for r in alignment]))},
        'by_split':splits,'all':summarize(rows),'offsets_px':OFFSETS,
        'reporting_note':'Five-section split reused after earlier exploration; purely descriptive, NOT unseen confirmatory TEST.',
        'cases_csv_sha256':hashlib.sha256(csv_out.read_bytes()).hexdigest()}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--out',default='results/cracks_real_controlled_v1/neighbor_v17.json')
    p.add_argument('--cases',default='results/cracks_real_controlled_v1/neighbor_v17_cases.csv')
    p.add_argument('--max-sections',type=int,default=400)
    a=p.parse_args()
    result=audit(Path(a.images),Path(a.annotations),Path(a.out),Path(a.cases),a.max_sections)
    print(json.dumps({k:v for k,v in result.items() if k not in ('all','by_split')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
