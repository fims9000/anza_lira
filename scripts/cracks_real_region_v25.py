"""Image-derived component extremities as candidate ports (post-V24 exploration).

Candidate source is predicted segmentation only. PCA component extremity
ports allow a link when the exact skeleton has no clean two-ended trace.
No expert annotations are consulted by any inference function.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi
from scripts.cracks_real_auto_core_v23 import (
    Port,extract_traces,propose_edges,score_edges,trace_path,unit
)


def component_extremity_ports(probability,mask,*,min_pixels=12,min_aspect=2.5):
    labels,num=ndi.label(mask,structure=np.ones((3,3),np.uint8))
    ports=[]
    for component,box in enumerate(ndi.find_objects(labels),start=1):
        if box is None:continue
        pixels=np.argwhere(labels[box]==component)
        if len(pixels)<min_pixels:continue
        pixels=pixels+np.array([box[0].start,box[1].start])
        center=pixels.mean(axis=0)
        centered=pixels.astype(float)-center
        gram=centered.T@centered/max(len(pixels)-1,1)
        values,vectors=np.linalg.eigh(gram)
        if values[-1]<1.e-3 or values[-1]/max(values[0],1.e-6)<min_aspect:
            continue
        axis=unit(vectors[:,-1])
        projections=centered@axis
        if float(projections.max()-projections.min())<8.:continue
        for sign in (-1,1):
            index=int(np.argmin(projections) if sign<0 else np.argmax(projections))
            point=tuple(map(int,pixels[index]))
            tangent=tuple(map(float,sign*axis))
            ports.append(Port(len(ports),1000000+component,point,
                              tangent,float(probability[point])))
    return labels,tuple(ports)


def filter_distinct_components(edges,ports,labels):
    return tuple((i,j,d,g) for i,j,d,g in edges
      if int(labels[ports[i].point])>0 and
         int(labels[ports[j].point])>0 and
         int(labels[ports[i].point])!=int(labels[ports[j].point]))


def region_candidates(probability,threshold,*,mode):
    if mode not in ('region_only','region_hybrid'):raise ValueError(mode)
    mask,traces,term=extract_traces(probability,threshold=threshold)
    labels,regional=component_extremity_ports(probability,mask)
    if mode=='region_only':ports=tuple(Port(i,p.trace_id,p.point,p.tangent,p.confidence)
                                      for i,p in enumerate(regional))
    else:
        ports=term+tuple(Port(len(term)+i,p.trace_id,p.point,p.tangent,p.confidence)
                         for i,p in enumerate(regional))
    proposed=propose_edges(ports,min_gap=3.,max_gap=96.,
                           max_candidates=12,max_angle_deg=85.)
    proposed=filter_distinct_components(proposed,ports,labels)
    return mask,traces,term,labels,ports,proposed


def run_regions(probability,threshold,*,mode):
    mask,traces,term,labels,ports,raw=region_candidates(
        probability,threshold,mode=mode)
    ranked=score_edges(raw,ports,probability,strategy='context')
    # Prevent cycles between already joined segmentation components and
    # over-merging: no more than two new links incident on each component.
    parent={};degree={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra,rb=root(a),root(b)
        if ra!=rb:parent[rb]=ra
    used_port=set();chosen=[]
    for edge in ranked:
        if len(chosen)>=120:break
        if edge.score<-.3:continue
        i,j=edge.first,edge.second
        if i in used_port or j in used_port:continue
        ca=int(labels[ports[i].point]);cb=int(labels[ports[j].point])
        if ca==0 or cb==0 or ca==cb:continue
        if degree.get(ca,0)>=2 or degree.get(cb,0)>=2:continue
        if root(ca)==root(cb):continue
        path=trace_path(probability,ports[i].point,ports[j].point,
                        half_width=6,max_multiplier=2.0)
        if not path:continue
        union(ca,cb)
        degree[ca]=degree.get(ca,0)+1
        degree[cb]=degree.get(cb,0)+1
        used_port.update((i,j))
        chosen.append({'p0':i,'p1':j,'score':edge.score,
                       'distance':edge.distance,'support':edge.support,
                       'path':path})
    completed=mask.copy()
    for edge in chosen:
        pixel=np.asarray(edge['path'],int)
        completed[pixel[:,0],pixel[:,1]]=True
    used_terminals={j for edge in chosen for j in (edge['p0'],edge['p1'])
                    if mode=='region_hybrid' and j<len(term)}
    return {'raw_mask':mask,'completed_mask':completed,'traces':traces,
            'ports':ports,'candidates':raw,'ranked':ranked,
            'selected':tuple(chosen),'terminal_port_count':len(term),
            'none_count':len(term)-len(used_terminals)}
