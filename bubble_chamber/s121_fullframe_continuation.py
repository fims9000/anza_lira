"""S121: conservative whole-frame continuation of S118 provisional tracks.

Every observed path comes from source-only S115 data. Cross-group inferred
links are hypotheses, never observed pixels. Only photograph-supported links
can form a new *provisional* group; unresolved rivals remain reviewable.
No manual labels or private classifier are read by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter,defaultdict
from hashlib import sha256
from math import ceil
import numpy as np
import cv2
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContinuationConfig:
    max_gap_px: float = 400.
    min_gap_px: float = 8.
    min_anchor_length_px: float = 55.
    dedup_port_px: float = 9.
    min_local_alignment: float = .90
    min_far_alignment: float = .82
    max_lateral_px: float = 2.5
    max_rival_fraction: float = .055
    min_photo_fraction: float = .70
    min_photo_fraction_long: float = .72
    min_score: float = .83
    min_best_margin: float = .10
    tangent_window_px: float = 22.
    far_tangent_window_px: float = 78.


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v,dtype=float)/n if n>1e-9 else None


def _end_tangent(xy, end, span):
    seq=xy if end==0 else xy[::-1]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(seq,axis=0),axis=1))]
    j=min(len(seq)-1,max(1,int(np.searchsorted(arc,span))))
    return _unit(seq[0]-seq[j])


def _new_port(key, group, xy, end, length, cfg):
    t=_end_tangent(xy,end,cfg.tangent_window_px)
    f=_end_tangent(xy,end,cfg.far_tangent_window_px)
    if t is None or f is None:return None
    return {'key':key,'end':end,'id':key+':'+str(end),'group':group,
            'pos':xy[0 if end==0 else -1],'tangent':t,'far_tangent':f,'arc_length':length}


def _extract_ports(tiles, local_groups, global_group, cfg):
    allports=[]; source_cloud=[]; cloud_owner=[]
    for tile in tiles:
        tid=tile['tile_id']; x0,y0=tile['xyxy'][:2]
        edge_in_use=set()
        for link in local_groups.get(tid,{}).get('links',[]):
            if link.get('decision') in ('AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT','EXPERT_CONFIRMED_SAME_LINK'):
                edge_in_use.update((str(k),int(e)) for k,e in zip(link['curve_ids'],link['endpoint_indices']))
        for curve in tile['curves']:
            name=str(curve.get('candidate_id') or curve.get('curve_id'))
            key=tid+':'+name
            if key not in global_group:raise ValueError('S118 group assignment missing for '+key)
            pts=np.asarray(curve['observed_pixel_path_xy'],float)
            if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<3 or not np.isfinite(pts).all():
                raise ValueError('Invalid S115 curve')
            if np.any(pts[:,0]<0) or np.any(pts[:,0]>=tile['xyxy'][2]-x0) or np.any(pts[:,1]<0) or np.any(pts[:,1]>=tile['xyxy'][3]-y0):
                raise ValueError('Curve is outside its photo tile')
            full=pts+np.asarray((x0,y0),float)
            idx=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(full,axis=0),axis=1))]
            arc=float(idx[-1]); step=max(1,int(ceil(len(full)/max(4,arc/7))))
            source_cloud.append(full[::step]);cloud_owner.extend([global_group[key]]*len(full[::step]))
            if arc<cfg.min_anchor_length_px:continue
            for end in (0,1):
                if (name,end) in edge_in_use:continue
                port=_new_port(key,global_group[key],full,end,arc,cfg)
                if port is not None:allports.append(port)
    # Collapse duplicate boundary observations of same physical source trace.
    kept=[];spatial=defaultdict(list)
    for port in sorted(allports,key=lambda a:(-a['arc_length'],a['id'])):
        x,y=port['pos'];gx=int(x//cfg.dedup_port_px);gy=int(y//cfg.dedup_port_px)
        found=False
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in spatial.get((gx+dx,gy+dy),[]):
                    a=kept[j]
                    if (a['group']==port['group'] and
                        np.linalg.norm(a['pos']-port['pos'])<cfg.dedup_port_px):
                        found=True;break
                if found:break
            if found:break
        if found:continue
        spatial[(gx,gy)].append(len(kept));kept.append(port)
    cloud=np.concatenate(source_cloud).astype(np.float32) if source_cloud else np.empty((0,2),np.float32)
    return kept,cloud,np.asarray(cloud_owner)


def _bridge(pa,pb,count):
    p,q=pa['pos'],pb['pos'];d=float(np.linalg.norm(q-p))
    points=np.stack((p,p+.29*d*pa['tangent'],q+.29*d*pb['tangent'],q))
    ts=np.linspace(0,1,count)[:,None]
    path=((1-ts)**3*points[0]+3*(1-ts)**2*ts*points[1]+3*(1-ts)*ts**2*points[2]+ts**3*points[3])
    return path,points


def _support(gray,xy):
    """Unchanged pixel intensity at the path and both normal flanks."""
    yx=xy[3:-3]
    if len(yx)<8:return 0.,0.
    grad=np.gradient(yx,axis=0)
    n=np.maximum(np.linalg.norm(grad,axis=1),1e-6)
    normal=np.column_stack((-grad[:,1]/n,grad[:,0]/n))
    h,w=gray.shape
    def sample(offset):
        loc=np.rint(yx+offset*normal).astype(np.int32)
        loc[:,0]=np.clip(loc[:,0],0,w-1);loc[:,1]=np.clip(loc[:,1],0,h-1)
        return gray[loc[:,1],loc[:,0]].astype(float)
    cen=sample(0);s4=(sample(4)+sample(-4))/2;s8=(sample(8)+sample(-8))/2
    contrast=.4*(s4-cen)+.6*(s8-cen)
    return float(np.mean(contrast>4.)),float(np.mean(contrast))


def _rival_fraction(tree,cloud_owner,xy,ga,gb):
    if tree is None:return 0.
    steps=xy[5:-5:max(1,len(xy)//110)]
    found=tree.query_ball_point(steps,r=3.2)
    bad=sum(any(cloud_owner[j] not in (ga,gb) for j in neighbors) for neighbors in found)
    return bad/max(1,len(found))


class _Union:
    def __init__(self,groups):self.parent={g:g for g in groups}
    def find(self,g):
        while self.parent[g]!=g:
            self.parent[g]=self.parent[self.parent[g]];g=self.parent[g]
        return g
    def link(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return False
        if a>b:a,b=b,a
        self.parent[b]=a
        return True


def associate_fullframe(tiles, local_groups, s118_groups, gray, cfg=ContinuationConfig()):
    """S118 global groups + S115 observed curves + original full-frame uint8.

    ``tiles``: each contains ``tile_id``, ``xyxy``, and original ``curves``;
    ``local_groups``: keyed by tile_id with S117 ``links``.
    Results retain every S118 group and every proposed edge for audit.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('Original grayscale uint8 BMP required')
    gmap={key:g['global_group_id'] for g in s118_groups for key in g['source_curve_keys']}
    if len(gmap)!=sum(len(g['source_curve_keys']) for g in s118_groups):raise ValueError('Duplicated global group membership')
    ports,cloud,owner=_extract_ports(tiles,local_groups,gmap,cfg)
    if len(ports)<2:
        return {'groups':list(s118_groups),'proposals':[], 'report':{'ports':len(ports),'accepted':0}}
    tree=cKDTree(np.array([p['pos'] for p in ports]))
    rivals=cKDTree(cloud) if len(cloud) else None
    pairs=sorted(tree.query_pairs(cfg.max_gap_px))
    proposed=[];reject=Counter()
    for a,b in pairs:
        pa,pb=ports[a],ports[b]
        if pa['group']==pb['group']:continue
        delta=pb['pos']-pa['pos'];d=float(np.linalg.norm(delta))
        if d<cfg.min_gap_px:continue
        axis=delta/d
        al=min(float(pa['tangent']@axis),float(pb['tangent']@(-axis)))
        if al<cfg.min_local_alignment:
            reject['local_angle']+=1;continue
        far=min(float(pa['far_tangent']@axis),float(pb['far_tangent']@(-axis)))
        if far<cfg.min_far_alignment:
            reject['far_angle']+=1;continue
        lateral=(abs(delta[0]*pa['tangent'][1]-delta[1]*pa['tangent'][0])+abs(delta[0]*pb['tangent'][1]-delta[1]*pb['tangent'][0]))/2
        if lateral>min(cfg.max_lateral_px,3.+.02*d):
            reject['offset']+=1;continue
        path,control=_bridge(pa,pb,max(28,min(180,int(d/2))))
        if not np.all((path[:,0]>=8)&(path[:,0]<gray.shape[1]-8)&(path[:,1]>=8)&(path[:,1]<gray.shape[0]-8)):
            reject['image_bounds']+=1;continue
        support,contrast=_support(gray,path)
        rival=_rival_fraction(rivals,owner,path,pa['group'],pb['group'])
        score=(.33*al+.16*far+.44*support+.07*(1.-d/cfg.max_gap_px)-.25*rival)
        pids=sorted([pa['id'],pb['id']])
        proposed.append({'candidate_id':'s121_'+sha256('|'.join(pids).encode()).hexdigest()[:16],
                         'group_ids':[pa['group'],pb['group']],'source_endpoint_ports':[pa['id'],pb['id']],
                         'gap_px':round(d,2),'alignment':round(al,4),'far_alignment':round(far,4),
                         'max_lateral_residual_px':round(float(lateral),3),
                         'source_photo_support':round(support,4),'photo_center_minus_flank':round(contrast,2),
                         'rival_fraction':round(rival,4),'score':round(score,5),
                         'inferred_xy':np.round(path,2).tolist(),
                         'source_observed_bridge_pixels':0,'inferred_geometry_only':True,
                         'decision':'REVIEW_ONLY_UNRESOLVED'})
    # Protect occupied ports, ambiguous alternatives, and cycles at GROUP level.
    byport=defaultdict(list)
    for e in proposed:
        for port in e['source_endpoint_ports']:byport[port].append(e)
    for rows in byport.values():rows.sort(key=lambda x:(-x['score'],x['candidate_id']))
    used=set();uf=_Union([g['global_group_id'] for g in s118_groups]);selected=[]
    for e in sorted(proposed,key=lambda x:(-x['score'],x['candidate_id'])):
        a,b=e['source_endpoint_ports']
        if a in used or b in used:
            e['decision']='REVIEW_ONLY_PORT_IN_USE';continue
        if uf.find(e['group_ids'][0])==uf.find(e['group_ids'][1]):
            e['decision']='REVIEW_ONLY_GROUP_CYCLE';continue
        ranks=[byport[a],byport[b]]
        margin=min((x[0]['score']-x[1]['score'] if len(x)>1 else 1.0) for x in ranks)
        mutual=all(x[0]['candidate_id']==e['candidate_id'] for x in ranks)
        e['score_margin']=round(float(margin),4)
        if not mutual or margin<cfg.min_best_margin:
            e['decision']='REVIEW_ONLY_COMPETING_CONTINUATION';continue
        photo_min=(cfg.min_photo_fraction_long if e['gap_px']>125 else cfg.min_photo_fraction)
        if e['source_photo_support']<photo_min or e['score']<cfg.min_score or e['rival_fraction']>cfg.max_rival_fraction:
            e['decision']='REVIEW_ONLY_INSUFFICIENT_SOURCE_EVIDENCE';continue
        # Never auto-promote inferred bridge if its path intersects an already
        # chosen inferred bridge: a crossing is a review request.
        if any(np.any(cKDTree(np.asarray(f['inferred_xy'])[5:-5:3]).query(np.asarray(e['inferred_xy'])[5:-5:3])[0]<3.) for f in selected):
            e['decision']='REVIEW_ONLY_CROSSING_BRIDGES';continue
        e['decision']='AUTO_PROVISIONAL_CONTINUATION_NOT_GT'
        selected.append(e);used.update((a,b));uf.link(*e['group_ids'])
    merged=defaultdict(list);member_original=defaultdict(list)
    for g in s118_groups:
        key=uf.find(g['global_group_id'])
        merged[key].extend(g['source_curve_keys'])
        member_original[key].append(g['global_group_id'])
    groups=[]
    for key in sorted(merged):
        ids=sorted(set(member_original[key]));curve_keys=sorted(set(merged[key]))
        groups.append({'global_group_id':'s121_group_'+sha256('|'.join(ids).encode()).hexdigest()[:16],
                       'source_curve_keys':curve_keys,'s118_member_group_ids':ids,
                       'source_observations_unchanged':True,'physical_particle_id':None,
                       'identity_status':'PHOTO_SUPPORTED_PROVISIONAL_NOT_GT'})
    return {'groups':groups,'proposals':proposed,'report':{'ports':len(ports),
           'raw_endpoint_pairs':len(pairs),'geometric_source_candidates':len(proposed),
           'accepted_provisional_links':len(selected),'review_only_links':len(proposed)-len(selected),
           'original_s118_groups':len(s118_groups),'provisional_s121_groups':len(groups),
           'no_observed_pixels_inferred':True,'manual_labels_used':False,
           'physical_identity_accuracy_known':False}}
