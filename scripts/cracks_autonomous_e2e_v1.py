#!/usr/bin/env python3
"""Autonomous CRACKS end-to-end baseline.

Crowd-only segmentation -> automatic endpoints -> candidate relations ->
NO_LINK or image-supported path. Expert masks are read ONLY after prediction.
This is not trained original ANZA/LIRA and does not identify physical faults.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from sklearn.ensemble import HistGradientBoostingClassifier

BLUE=(31,119,180)
ORANGE=(255,127,14)
SOURCE_VERSION="CRACKS_AUTONOMOUS_2D_BASELINE_V1"

def image_index(z):
    result={}
    for name in z.namelist():
        m=re.search(r"section_(\d+)\.png$",name,re.I)
        if m:result[int(m.group(1))]=name
    return result

def annotation_index(z):
    result=defaultdict(dict)
    for name in z.namelist():
        m=re.search(r"section_(\d+)\.png$",name,re.I)
        if m and len(name.split("/"))>=2:
            result[int(m.group(1))][name.split("/")[-2]]=name
    return result

def read_png(z,name,mode):
    with Image.open(io.BytesIO(z.read(name))) as im:
        return np.asarray(im.convert(mode))

def features(image):
    x=np.asarray(image,np.float32)
    if x.ndim==3:x=x.mean(axis=-1)
    if x.max()>1.5:x/=255.
    g1=ndi.gaussian_filter(x,1)
    g2=ndi.gaussian_filter(x,2)
    g4=ndi.gaussian_filter(x,4)
    dx=ndi.sobel(g1,axis=1)/8
    dy=ndi.sobel(g1,axis=0)/8
    hxx=ndi.gaussian_filter(g2,1,order=(0,2))
    hxy=ndi.gaussian_filter(g2,1,order=(1,1))
    hyy=ndi.gaussian_filter(g2,1,order=(2,0))
    t=np.sqrt((hxx-hyy)**2+4*hxy**2)
    a=(hxx+hyy+t)/2
    b=(hxx+hyy-t)/2
    return np.stack([x,g1,g2,g4,x-g2,g1-g4,dx,dy,
                     np.hypot(dx,dy),abs(dx),abs(dy),a,b,abs(a)-abs(b)],
                    axis=-1).astype(np.float32)

def train_crowd(images,labels,imgidx,maskidx,train_sections):
    x=[];y=[];totals={"positive":0,"negative":0,"masks":0}
    rng=np.random.default_rng(42)
    for sec in train_sections:
        if sec not in imgidx or sec not in maskidx:continue
        f=features(read_png(images,imgidx[sec],"L")).reshape(-1,14)
        crowd=sorted(k for k in maskidx[sec] if k.lower()!="expert")
        for ann in crowd[:6]:
            rgb=read_png(labels,maskidx[sec][ann],"RGB")
            for positive,color in ((1,BLUE),(0,ORANGE)):
                inds=np.flatnonzero(np.all(rgb==color,axis=-1))
                if len(inds)==0:continue
                chosen=rng.choice(inds,size=min(42,len(inds)),replace=False)
                x.append(f[chosen]);y.extend([positive]*len(chosen))
                totals["positive" if positive else "negative"]+=len(chosen)
            totals["masks"]+=1
    if len(set(y))!=2 or not x:raise RuntimeError("Missing explicit positive or negative crowd training pixels")
    model=HistGradientBoostingClassifier(max_iter=100,max_depth=4,
             max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,random_state=42)
    model.fit(np.concatenate(x),np.asarray(y,np.uint8))
    return model,totals

def calibrate(model,images,labels,imgidx,maskidx,cal_sections):
    rows=[]
    for sec in cal_sections:
        if sec not in imgidx or sec not in maskidx:continue
        crowd=sorted(k for k in maskidx[sec] if k.lower()!="expert")
        if not crowd:continue
        im=read_png(images,imgidx[sec],"L")
        prob=model.predict_proba(features(im).reshape(-1,14))[:,1]
        rgb=read_png(labels,maskidx[sec][crowd[0]],"RGB")
        yes=np.all(rgb==BLUE,axis=-1).ravel()
        no=np.all(rgb==ORANGE,axis=-1).ravel()
        if yes.any() and no.any():rows.append((prob,yes,no))
    if not rows:raise RuntimeError("Calibration lacks explicit positive and negative crowd pixels")
    results=[]
    for threshold in (.30,.40,.50,.60,.70):
        tp=fp=fn=0
        for p,yes,no in rows:
            pred=p>=threshold
            tp+=int((pred&yes).sum())
            fp+=int((pred&no).sum())
            fn+=int(((~pred)&yes).sum())
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        results.append({"threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"f1":f1})
    best=max(results,key=lambda r:(r["f1"],-r["fp"],-r["threshold"]))
    return best,results

@dataclass(frozen=True)
class Endpoint:
    id:int
    component:int
    point:tuple[int,int]
    tangent:tuple[float,float]

def endpoints_from_mask(mask):
    skeleton=skeletonize(np.asarray(mask,bool))
    skeleton[:7]=False;skeleton[-7:]=False
    skeleton[:,:7]=False;skeleton[:,-7:]=False
    groups,n=ndi.label(skeleton,np.ones((3,3),np.uint8))
    kernel=np.ones((3,3),np.uint8);kernel[1,1]=0
    degree=ndi.convolve(skeleton.astype(np.uint8),kernel,mode="constant")
    ends=[]
    for cid in range(1,n+1):
        coords=np.argwhere(groups==cid)
        if len(coords)<9:continue
        d=degree[coords[:,0],coords[:,1]]
        close=coords[d==1]
        if len(close)!=2 or (d>2).any():continue
        for e in close:
            begin=tuple(map(int,e))
            visited={begin};current=begin
            for _ in range(6):
                y,x=current
                choices=[(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                         if (dx or dy) and 0<=y+dy<mask.shape[0]
                         and 0<=x+dx<mask.shape[1] and groups[y+dy,x+dx]==cid
                         and (y+dy,x+dx) not in visited]
                if len(choices)!=1:break
                current=choices[0];visited.add(current)
            vec=np.asarray(begin,float)-np.asarray(current,float)
            norm=float(np.linalg.norm(vec))
            if norm<2:continue
            ends.append(Endpoint(len(ends),cid,begin,tuple((vec/norm).tolist())))
    return ends

def seismic_support(image):
    a=np.asarray(image,np.float32)
    if a.ndim==3:a=a.mean(axis=-1)
    if a.max()>1.5:a/=255.
    gx=np.abs(ndi.sobel(ndi.gaussian_filter(a,1),axis=1))
    return np.clip(gx/(np.percentile(gx,95)+1e-6),0,2)

def candidates(mask,prob,image):
    ends=endpoints_from_mask(mask)
    if not ends:return [],ends
    pts=np.asarray([e.point for e in ends],float)
    tree=cKDTree(pts)
    signal=seismic_support(image)
    proposals=[]
    for i,e in enumerate(ends):
        local=[]
        for j in tree.query_ball_point(pts[i],48.):
            if j<=i or e.component==ends[j].component:continue
            displacement=pts[j]-pts[i]
            distance=float(np.linalg.norm(displacement))
            if distance<4:continue
            direction=displacement/distance
            heading=min(float(np.dot(direction,e.tangent)),
                        float(np.dot(-direction,ends[j].tangent)))
            if heading<.50:continue
            length=max(16,int(distance*2))
            ys=np.rint(np.linspace(pts[i,0],pts[j,0],length)).astype(int)
            xs=np.rint(np.linspace(pts[i,1],pts[j,1],length)).astype(int)
            ys=np.clip(ys,0,mask.shape[0]-1)
            xs=np.clip(xs,0,mask.shape[1]-1)
            im_score=float(signal[ys,xs].mean())
            p_score=float(prob[ys,xs].mean())
            score=.54*heading+.33*min(1.,im_score)+.13*p_score-.005*distance
            local.append((score,i,j,distance))
        proposals.extend(sorted(local,reverse=True)[:8])
    return sorted(proposals,reverse=True),ends

def path_through_image(mask,prob,image,start,goal):
    h,w=mask.shape
    ya=max(0,min(start[0],goal[0])-20)
    yb=min(h,max(start[0],goal[0])+21)
    xa=max(0,min(start[1],goal[1])-20)
    xb=min(w,max(start[1],goal[1])+21)
    p=prob[ya:yb,xa:xb]
    signal=seismic_support(image)[ya:yb,xa:xb]
    cost=1+2*(1-np.clip(p,0,1))+.4*(1-np.clip(signal,0,1))
    cost[np.asarray(mask[ya:yb,xa:xb],bool)]+=15
    s=(start[0]-ya,start[1]-xa);g=(goal[0]-ya,goal[1]-xa)
    cost[s]=1;cost[g]=1
    path,_=route_through_array(cost,s,g,fully_connected=True,geometric=True)
    path=np.asarray(path,int);path[:,0]+=ya;path[:,1]+=xa
    return path

def complete(mask,prob,image,decision_threshold=.72,max_links=35):
    result=np.asarray(mask,bool).copy()
    options,ends=candidates(result,prob,image)
    parent={e.component:e.component for e in ends}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    used=set();links=[]
    for score,i,j,distance in options:
        if score<decision_threshold:break
        if i in used or j in used:continue
        c1=root(ends[i].component);c2=root(ends[j].component)
        if c1==c2:continue
        path=path_through_image(result,prob,image,ends[i].point,ends[j].point)
        if len(path)>1.7*distance+3:continue
        middle=path[2:-2]
        if len(middle) and np.any(result[middle[:,0],middle[:,1]]):continue
        result[path[:,0],path[:,1]]=True
        used.update((i,j));parent[c2]=c1
        links.append({"from":ends[i].point,"to":ends[j].point,"score":float(score),
                      "path_pixels":len(path)})
        if len(links)>=max_links:break
    return result,links,{"candidates":len(options),"endpoints":len(ends),
                         "no_link":len(links)==0}

def explicit_mask_metrics(pred,expert):
    positive=np.all(expert==BLUE,axis=-1)
    negative=np.all(expert==ORANGE,axis=-1)
    tp=int(np.sum(pred&positive))
    fp=int(np.sum(pred&negative))
    fn=int(np.sum(~pred&positive))
    return {"tp":tp,"fp":fp,"fn":fn,
            "positive_pixels":int(positive.sum()),
            "negative_pixels":int(negative.sum())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True,type=Path)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--max-test",type=int,default=40)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.images) as images,zipfile.ZipFile(a.annotations) as labels:
        ims=image_index(images);anns=annotation_index(labels)
        expert_ids=sorted(i for i in ims if "expert" in anns.get(i,{}))
        if not expert_ids:raise RuntimeError("No expert masks for testing")
        max_expert=max(expert_ids)
        train=[i for i in sorted(ims) if i>=max_expert+25 and i in anns and i%5!=0][:50]
        cal=[i for i in sorted(ims) if i>=max_expert+25 and i in anns and i%5==0][:15]
        if len(train)<12 or len(cal)<3:
            raise RuntimeError(f"Not enough spatially separated crowd data: train={len(train)} cal={len(cal)}")
        model,counts=train_crowd(images,labels,ims,anns,train)
        threshold,calibration=calibrate(model,images,labels,ims,anns,cal)
        cases=[]
        for sec in expert_ids[:a.max_test]:
            raw=read_png(images,ims[sec],"L")
            prob=model.predict_proba(features(raw).reshape(-1,14))[:,1].reshape(raw.shape)
            foreground=prob>=threshold["threshold"]
            cc,_=ndi.label(foreground,np.ones((3,3),np.uint8))
            size=np.bincount(cc.ravel())
            foreground=(cc>0)&(size[cc]>=10)
            linked,links,diagnostics=complete(foreground,prob,raw)
            # No expert annotation was accessed before all decisions above.
            expert=read_png(labels,anns[sec]["expert"],"RGB")
            cases.append({"section":sec,"baseline":explicit_mask_metrics(foreground,expert),
                          "completed":explicit_mask_metrics(linked,expert),
                          "links":len(links),**diagnostics})
            if len(cases)<=4:
                rgb=np.repeat(raw[...,None],3,axis=2)
                rgb[foreground]=[45,190,240]
                rgb[linked&~foreground]=[248,80,120]
                Image.fromarray(rgb).save(a.out/f"section_{sec:03d}.png")
        aggregates={}
        for variant in ("baseline","completed"):
            row={k:sum(item[variant][k] for item in cases) for k in
                 ("tp","fp","fn","positive_pixels","negative_pixels")}
            tp=row["tp"];fp=row["fp"];fn=row["fn"]
            row["precision"]=tp/(tp+fp) if tp+fp else 0.
            row["recall"]=tp/(tp+fn) if tp+fn else 0.
            row["f1"]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            aggregates[variant]=row
        report={"protocol":SOURCE_VERSION,"train_ids":train,"cal_ids":cal,
                "expert_ids":[x["section"] for x in cases],"buffer_sections":25,
                "training_counts":counts,"calibration":calibration,
                "pixel_threshold":threshold,"cases":cases,"aggregate":aggregates,
                "total_added_links":sum(x["links"] for x in cases),
                "uses_expert_for_inference":False,"original_ANZA_LIRA_trained":False,
                "natural_geological_link_identification_evaluated":False,
                "claim":"Real CRACKS image E2E technical BASELINE, not validated original ANZA/LIRA"}
        (a.out/"audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        with (a.out/"cases.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["section","links","candidates",
                "endpoints","baseline_tp","baseline_fp","completed_tp","completed_fp"])
            writer.writeheader()
            for row in cases:
                writer.writerow({"section":row["section"],"links":row["links"],
                  "candidates":row["candidates"],"endpoints":row["endpoints"],
                  "baseline_tp":row["baseline"]["tp"],"baseline_fp":row["baseline"]["fp"],
                  "completed_tp":row["completed"]["tp"],"completed_fp":row["completed"]["fp"]})
        print(json.dumps({"train":len(train),"cal":len(cal),
                          "expert":len(cases),"links":report["total_added_links"],
                          "aggregate":aggregates},indent=2))
if __name__=="__main__":main()
