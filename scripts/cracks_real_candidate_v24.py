"""V24: exploratory high-recall branch candidates from predicted seismic fragments.

All source/landing ports come from the predicted mask. Never use the expert
mask to construct proposed edges. New candidates may include a visible branch
interior (virtual landing). A virtual landing is NOT a fault instance label.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

from scripts.cracks_real_auto_core_v23 import (
    Port, extract_traces, propose_edges, score_edges, accept_edges,
    trace_path, unit,
)


def extended_terminal_candidates(ports, components=None):
    """V23 controls expanded to 68px, 85deg, 16 links/terminal (fixed)."""
    proposed=propose_edges(ports, min_gap=3.0, max_gap=68.0,
                         max_candidates=16, max_angle_deg=85.0)
    if components is None:return proposed
    return tuple((i,j,d,g) for i,j,d,g in proposed
        if int(components[ports[i].point])>0 and
           int(components[ports[j].point])>0 and
           int(components[ports[i].point])!=int(components[ports[j].point]))


def virtual_landing_candidates(traces, terminal_ports, probability, components=None):
    """Terminal-to-interior competitors plus relaxed terminal-to-terminal.

    Only deterministic interpolation of a predicted trace is used to generate
    virtual ports. For any source port, keep 8 best interior candidates by
    same geometric ranking. The output has terminal ports first.
    """
    ports=list(terminal_ports)
    raw_edges=list(extended_terminal_candidates(terminal_ports,components))
    real_count=len(terminal_ports)
    for trace in traces:
        pts=trace.points
        if len(pts)<22:continue
        ix=np.linspace(8,len(pts)-9,min(12,max(1,len(pts)//9)),dtype=int)
        for q in np.unique(ix):
            position=np.asarray(pts[q],dtype=np.float64)
            tangent=unit(np.asarray(pts[min(len(pts)-1,q+5)],float)-
                         np.asarray(pts[max(0,q-5)],float))
            ports.append(Port(len(ports),trace.id,
                       tuple(int(x) for x in position),
                       tuple(float(x) for x in tangent),trace.mean_probability))
    if len(ports)==real_count:return tuple(ports),tuple(raw_edges)
    coords=np.asarray([p.point for p in ports[real_count:]],dtype=float)
    tree=cKDTree(coords)
    by_pair={(min(i,j),max(i,j)):(i,j,d,g) for i,j,d,g in raw_edges}
    min_cos=math.cos(math.radians(85.0))
    for i,source in enumerate(terminal_ports):
        sp=np.asarray(source.point,dtype=float)
        options=[]
        for jlocal in tree.query_ball_point(sp,68.):
            j=real_count+jlocal
            target=ports[j]
            if target.trace_id==source.trace_id:continue
            if components is not None and (
                int(components[source.point])==0 or
                int(components[target.point])==0 or
                int(components[source.point])==int(components[target.point])):
                continue
            dp=np.asarray(target.point,dtype=float)-sp
            distance=float(np.linalg.norm(dp))
            if distance<3. or distance>68.:continue
            unit_chord=dp/distance
            forward=float(np.dot(np.asarray(source.tangent),unit_chord))
            downstream=abs(float(np.dot(np.asarray(target.tangent),unit_chord)))
            if min(forward,downstream)<min_cos:continue
            geo=distance/68.+.7*(2.-forward-downstream)+.15
            options.append((geo,j,distance))
        for geo,j,distance in sorted(options)[:8]:
            pair=(min(i,j),max(i,j))
            previous=by_pair.get(pair)
            if previous is None or previous[3]>geo:
                by_pair[pair]=(pair[0],pair[1],distance,float(geo))
    return tuple(ports),tuple(sorted(by_pair.values(),key=lambda x:(x[3],x[0],x[1])))


def run_expanded(probability, threshold, *,mode):
    if mode not in ('terminal_extended','virtual_landing'):
        raise ValueError(mode)
    mask,traces,terminal_ports=extract_traces(probability,threshold=threshold)
    components,_=ndi.label(mask,structure=np.ones((3,3),np.uint8))
    if mode=='terminal_extended':
        ports=terminal_ports
        raw=extended_terminal_candidates(ports,components)
    else:
        ports,raw=virtual_landing_candidates(traces,terminal_ports,probability,components)
    scored=score_edges(raw,ports,probability,strategy='context')
    selected=accept_edges(scored,ports,strategy='context',score_gate=-.3,max_links=120)
    completed=mask.copy()
    done=[]
    for e in selected:
        path=trace_path(probability,ports[e.first].point,ports[e.second].point)
        if not path:continue
        a=np.asarray(path,dtype=int)
        completed[a[:,0],a[:,1]]=True
        done.append({'p0':e.first,'p1':e.second,'score':e.score,
                     'distance':e.distance,'support':e.support,'path':path})
    used_terminals={i for e in done for i in (e['p0'],e['p1'])
                    if i<len(terminal_ports)}
    return {'raw_mask':mask,'completed_mask':completed,'traces':traces,
            'ports':ports,'terminal_port_count':len(terminal_ports),
            'candidates':raw,'ranked':scored,'selected':tuple(done),
            'none_count':len(terminal_ports)-len(used_terminals)}
