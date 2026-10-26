"""S58: compose reversible machine graph edges into *draft 2D paths*.

No real-physics GT, event identity, 3D, mass/charge or interpolation truth is
inferred. All gap bridges are model-produced, never observed H50Z pixels.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import networkx as nx
from .s35_fragment_link import _endpoint,hermite_gap
from .s56_global_path_assembly import Edge,make_edge,_forest_and_ports_valid


def _oriented_path(polyline,previous_port: int|None,following_port: int|None):
    pts=np.asarray(polyline,dtype=np.float64)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<8 or not np.isfinite(pts).all():
        raise ValueError('Invalid unverified machine fragment polyline')
    if previous_port is not None and following_port is not None and previous_port==following_port:
        raise ValueError('A fragment cannot use one endpoint twice')
    if previous_port is not None:
        return pts.copy() if previous_port==0 else pts[::-1].copy()
    if following_port is not None:
        return pts.copy() if following_port==-1 else pts[::-1].copy()
    return pts.copy()


def assemble_components(nodes:list[dict],selected_pairs:list[list[int]])->list[dict]:
    if not 0<=len(nodes)<=32:
        raise ValueError('Invalid source graph size')
    edges=[]
    for p in selected_pairs:
        if (not isinstance(p,list) or len(p)!=2 or not all(isinstance(x,int) for x in p)
            or p[0]<0 or p[1]<=p[0] or p[1]>=len(nodes)):
            raise ValueError('Unknown or duplicated native graph node identifiers')
        a=np.asarray(nodes[p[0]]['points_xy'],float)
        b=np.asarray(nodes[p[1]]['points_xy'],float)
        edges.append(make_edge(p[0],p[1],a,b,score=1.))
    if len({(e.i,e.j) for e in edges})!=len(edges) or not _forest_and_ports_valid(edges):
        raise ValueError('Expected a nonbranching machine-only open path forest')
    g=nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    for e in edges:g.add_edge(e.i,e.j,source=e)
    out=[]
    for component in nx.connected_components(g):
        if len(component)<2:continue
        sub=g.subgraph(component)
        ends=sorted(n for n in component if sub.degree(n)==1)
        if len(ends)!=2:
            raise ValueError('Open path component must have exactly two ends')
        path=[ends[0]]
        while len(path)<len(component):
            options=[z for z in sub.neighbors(path[-1]) if len(path)<2 or z!=path[-2]]
            if len(options)!=1:raise ValueError('Ambiguous non-path graph')
            path.append(options[0])
        oriented=[]
        for i,idx in enumerate(path):
            def port_of(neighbor):
                e=sub[idx][neighbor]['source']
                return e.end_i if e.i==idx else e.end_j
            prev_port=port_of(path[i-1]) if i>0 else None
            next_port=port_of(path[i+1]) if i+1<len(path) else None
            oriented.append(_oriented_path(nodes[idx]['points_xy'],prev_port,next_port))
        sequence=[]
        for i,pts in enumerate(oriented):
            sequence.append({'kind':'OBSERVED_MACHINE_CANDIDATE_NOT_GT',
                            'graph_node':path[i],
                            'points_xy_PRIVATE':np.round(pts,2).tolist()})
            if i+1<len(oriented):
                a,ta=_endpoint(pts,-1)
                b,tb=_endpoint(oriented[i+1],0)
                bridge=hermite_gap(a,b,ta,tb,n=21)
                sequence.append({'kind':'PREDICTED_GAP_NOT_OBSERVED_AND_NOT_PHYSICS_GT',
                                 'node_pair':[path[i],path[i+1]],
                                 'points_xy_PRIVATE':np.round(bridge,2).tolist()})
        out.append({'node_count':len(path),'join_count':len(path)-1,
                    'ordered_machine_graph_nodes_PRIVATE':path,
                    'segments_PRIVATE':sequence,
                    'path_status':'UNKNOWN_PHYSICAL_IDENTITY',
                    'physics_auto_joins_authorized':False})
    return out
