"""S31 machine-first drafts for real H50Z, with strict source and provenance.

A draft is neither expert-certified track nor a known physical association. It is
usable for *weak supervised exploration* and a separate correction workflow.
Real operator START is used as positional hint, never as complete track truth.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import json, math
import cv2
import numpy as np
from .s30_trajectory_identity import gate_candidates,mutual_coverage
from .s29_candidate_measurement import parametric_quadratic_fit,photo_line_profile

SCHEMA='s31_machine_first_h50z_draft_v1'


def file_sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def actual_pixel_sha(gray):
    if gray.dtype!=np.uint8 or gray.shape!=(512,512):raise ValueError('Invalid source 512x512 grayscale pixels')
    return sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()


def require_sources(tile_dir, photo_records):
    tile_dir=Path(tile_dir)
    if len(photo_records)!=39 or len({r['crop_id'] for r in photo_records})!=39:
        raise ValueError('Expected 39 unique S29 photographic source crops')
    out={}
    for r in photo_records:
        name=r['crop_id']
        if not name.startswith('H50Z_030_01_') or '/' in name or '\\' in name or '..' in name:raise ValueError('Unsafe crop name')
        src=(tile_dir/(name+'.png')).resolve()
        if not src.is_relative_to(tile_dir.resolve()) or not src.is_file():raise FileNotFoundError(src)
        im=cv2.imread(str(src),cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape!=(512,512) or im.dtype!=np.uint8:raise ValueError('Bad real photo tile')
        out[name]=(im,src)
    return out


def candidate_id(crop_sha:str,kind:str,index:int):
    return 'draft_'+sha256(f'S31|{crop_sha}|{kind}|{index}'.encode()).hexdigest()[:20]


def _s31_line_contrast(gray,pts):
    p=np.asarray(pts,float)
    if len(p)<2:return 0.
    centers=np.round(p).astype(int)
    inside=(centers[:,0]>=10)&(centers[:,0]<502)&(centers[:,1]>=10)&(centers[:,1]<502)
    p=p[inside]
    if len(p)<4:return 0.
    direction=(p[-1]-p[0]);den=float(np.linalg.norm(direction))
    if den<1:return 0.
    n=np.asarray([-direction[1],direction[0]])/den
    coords=np.round(p).astype(int)
    lx=np.clip(np.round(p[:,0]-8*n[0]).astype(int),0,511);ly=np.clip(np.round(p[:,1]-8*n[1]).astype(int),0,511)
    rx=np.clip(np.round(p[:,0]+8*n[0]).astype(int),0,511);ry=np.clip(np.round(p[:,1]+8*n[1]).astype(int),0,511)
    surround=.5*(gray[ly,lx].astype(float)+gray[ry,rx].astype(float))
    contrast=surround-gray[coords[:,1],coords[:,0]].astype(float)
    return float(np.median(contrast))


def lsd_candidates(gray,max_drafts=8):
    """Straight *pieces* of possible tracks; never assume crossing pieces same particle."""
    if gray.shape!=(512,512) or gray.dtype!=np.uint8:raise ValueError('Expected original gray tile')
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:return []
    proposals=[]
    for points in raw.reshape(-1,4):
        a=points[:2];b=points[2:];length=float(np.linalg.norm(b-a))
        if length<42:continue
        count=max(15,int(length/2)+1)
        poly=np.linspace(a,b,count,dtype='float64')
        if np.any(poly<0) or np.any(poly>511):continue
        contrast=_s31_line_contrast(gray,poly)
        if contrast<4.5:continue
        # Intrinsic ranking only, scores are NOT probabilities of a track.
        priority=math.log1p(length)*math.log1p(max(contrast,0.))
        proposals.append({'points':poly.round(2).tolist(),'length_px':length,
            'observed_contrast_8bit':contrast,'ranking_score_UNCALIBRATED':priority})
    proposals.sort(key=lambda c:-c['ranking_score_UNCALIBRATED'])
    selected=[]
    for p in proposals:
        row={'path_points_original_xy':p['points']}
        if any(mutual_coverage(row,{'path_points_original_xy':q['points']},radius=4.)>.65 for q in selected):continue
        selected.append(p)
        if len(selected)>=max_drafts:break
    return selected


def source_locked_machine_drafts(s28:dict, photos:list, tile_dir:Path, *, max_nonanchor:int=5):
    if s28.get('schema')!='s28_genuine_h50z_operator_anchor_to_unverified_2D_path_v1' or s28.get('photo_native_track_segment_masks')!=0:
        raise ValueError('Expected unverified authentic S28 anchor candidates')
    if s28.get('real_operator_start_anchors')!=9:raise ValueError('Unexpected source anchor count')
    tiles=require_sources(tile_dir,photos)
    byid={r['crop_id']:r for r in photos};anchor_map={r['crop_id']:r for r in s28['records']}
    if len(anchor_map)!=9 or not set(anchor_map)<=set(byid):raise ValueError('S28 source mismatch')
    dataset={'schema':SCHEMA,'source_material':'two original H50Z BMPs from same physical run',
        'source_photo_ids':['005','006'],'expert_track_labels_generated':0,'physical_joins_authorized':False,
        'model_drafts_are_ground_truth':False,'available_operator_start_points':9,
        'provenance':'original ANZA-assisted S28 + curvature classic S28 + image LSD on original H50Z pixels',
        'photos':[]}
    for p in photos:
        cid=p['crop_id'];gray,path=tiles[cid];crop_h=file_sha(path)
        record={'crop_id':cid,'photo_id':p['photo'],'stratum':p['review_stratum'],
           'source_original_png_sha256_from_S29':p['sha256'],
           'derived_png_sha256_from_BMP':crop_h,'source_gray_pixels_sha256':actual_pixel_sha(gray),
           'native_photo_origin_xy_px':p['native_photo_origin_xy_px'],
           'image_shape':[512,512],'drafts':[],'human_confirmed_track_count':0,
           'human_confirmed_artifact_count':0}
        anchored=cid in anchor_map
        if anchored:
            rec=anchor_map[cid]
            if rec['source_crop_sha256']!=p['sha256'] or rec['photo_id']!=p['photo']:
                raise ValueError('S28 crop source digest or parent photo mismatch')
            original=[(kind,index,c) for kind,key in [('classical','classical_candidate_paths'),('original_AZUNet_assisted','original_AZUNet_assisted_candidate_paths')]
                      for index,c in enumerate(rec[key])]
            if len(original)!=8:raise ValueError('Expected 8 original candidate paths per start')
            gate=gate_candidates([x[2] for x in original])
            for k,(kind,index,c) in enumerate(original):
                pts=c['path_points_original_xy']
                consensus=any(mutual_coverage(c,other,radius=3.)>=.75 for okind,_,other in original if okind!=kind)
                fit=parametric_quadratic_fit(pts)
                try:photometry=photo_line_profile(gray,pts,downsample=3)
                except ValueError:photometry={'fraction_with_dark_center':0,'median_apparent_width_px':None}
                row_gate=gate['rows'][k]
                recdraft={'draft_id':candidate_id(crop_h,kind,index),'source_method':kind,
                     'source_hypothesis_index':index,'points_xy':pts,'origin':'operator_start_guided_model_draft',
                     'model_score_UNCALIBRATED':float(c['model_selection_score_UNCALIBRATED']),
                     'photometry_support_fraction_UNCALIBRATED':float(c['fraction_steps_above_local_signal_0_15']),
                     'cross_method_agreement_UNCALIBRATED':bool(consensus),
                     'duplicate_of_draft_id':None,'geometry':{k:v for k,v in fit.items() if k!='fitted_points_original_xy'},
                     'photometry':photometry,'status':'MACHINE_DRAFT_UNREVIEWED',
                     'reviewed':False,'label_source':'MODEL_NOT_HUMAN','physics_permission':False}
                if row_gate['duplicate_of'] is not None:
                    j=row_gate['duplicate_of'];rec_did=candidate_id(crop_h,original[j][0],original[j][1]);recdraft['duplicate_of_draft_id']=rec_did
                record['drafts'].append(recdraft)
            record['start_xy_OPERATOR_REAL']=rec['genuine_OPERATOR_start_xy_in_crop']
            record['anchor_gate']=gate['decision']
        # Source-image line detector finds unanchored segments on ALL 39 photos,
        # not just operator START views; this supplies machine drafts to correct.
        lsd=lsd_candidates(gray,max_drafts=max_nonanchor)
        for j,c in enumerate(lsd):
            pts=c['points']
            if anchored and any(mutual_coverage({'path_points_original_xy':pts},{'path_points_original_xy':d['points_xy']},radius=4.)>.67 for d in record['drafts']):
                continue
            try:fit=parametric_quadratic_fit(pts)
            except ValueError:continue
            record['drafts'].append({'draft_id':candidate_id(crop_h,'LSD_line_piece',j),
                'source_method':'LSD_line_piece','source_hypothesis_index':j,'points_xy':pts,
                'origin':'raw_image_segment_detector','observed_contrast_8bit':round(c['observed_contrast_8bit'],3),
                'model_score_UNCALIBRATED':round(c['ranking_score_UNCALIBRATED'],4),
                'photometry_support_fraction_UNCALIBRATED':None,
                'cross_method_agreement_UNCALIBRATED':False,'duplicate_of_draft_id':None,
                'geometry':{k:v for k,v in fit.items() if k!='fitted_points_original_xy'},
                'status':'MACHINE_DRAFT_UNREVIEWED','reviewed':False,
                'label_source':'MODEL_NOT_HUMAN','physics_permission':False})
        # Candidate links are only short local geometric adjacency questions.
        # Real particles can branch at a vertex; no SAME/DIFFERENT truth is added.
        drafts=record['drafts'];relations=[]
        for i,a in enumerate(drafts):
            for b in drafts[i+1:]:
                if a['source_method']!='LSD_line_piece' and b['source_method']!='LSD_line_piece':continue
                ap=np.asarray(a['points_xy'],float);bp=np.asarray(b['points_xy'],float)
                if mutual_coverage({'path_points_original_xy':ap},{'path_points_original_xy':bp},radius=3.)>.55:continue
                ends=[(ai,bi,float(np.linalg.norm(ap[ai]-bp[bi]))) for ai in (0,-1) for bi in (0,-1)]
                ai,bi,gap=min(ends,key=lambda v:v[2])
                if not 3<gap<=42:continue
                at=(ap[1]-ap[0]) if ai==0 else (ap[-1]-ap[-2])
                bt=(bp[1]-bp[0]) if bi==0 else (bp[-1]-bp[-2])
                dot=abs(float(np.dot(at,bt)/(max(np.linalg.norm(at)*np.linalg.norm(bt),1e-10))))
                if dot<.70:continue
                rid='rel_'+sha256((crop_h+'|'+a['draft_id']+'|'+b['draft_id']).encode()).hexdigest()[:20]
                relations.append({'relation_id':rid,'candidate_a':a['draft_id'],'candidate_b':b['draft_id'],
                                  'endpoint_gap_px':round(gap,2),'axial_tangent_alignment_UNCALIBRATED':round(dot,4),
                                  'status':'MACHINE_POSSIBLE_LINK_UNKNOWN','expert_confirmed':False,'physical_join_allowed':False})
        record['relation_drafts']=sorted(relations,key=lambda r:(r['endpoint_gap_px'],-r['axial_tangent_alignment_UNCALIBRATED']))[:8]
        dataset['photos'].append(record)
    if sum('start_xy_OPERATOR_REAL' in r for r in dataset['photos'])!=9:raise AssertionError('missing genuine starts')
    dataset['counts']={'source_crops':len(dataset['photos']),
         'starts':9,'candidate_drafts':sum(len(r['drafts']) for r in dataset['photos']),
         'original_ANZA_candidates':sum(d['source_method']=='original_AZUNet_assisted' for r in dataset['photos'] for d in r['drafts']),
         'classical_candidates':sum(d['source_method']=='classical' for r in dataset['photos'] for d in r['drafts']),
         'image_LSD_drafts':sum(d['source_method']=='LSD_line_piece' for r in dataset['photos'] for d in r['drafts']),
         'possible_relation_questions':sum(len(r['relation_drafts']) for r in dataset['photos']),
         'human_accepted':0,'human_rejected':0}
    return dataset


def validate_draft_pack(obj:dict):
    if obj.get('schema')!=SCHEMA or obj.get('physical_joins_authorized') is not False:raise ValueError('Wrong or unsafe draft pack')
    if obj.get('expert_track_labels_generated')!=0 or obj.get('model_drafts_are_ground_truth') is not False:raise ValueError('Model pseudo truth masquerading as human')
    photos=obj.get('photos',[]);ids=set();draft_ids=set()
    for r in photos:
        if r['crop_id'] in ids:raise ValueError('Duplicate crop')
        ids.add(r['crop_id']);sha=r.get('source_gray_pixels_sha256','')
        if len(sha)!=64:raise ValueError('Missing authentic crop pixel provenance')
        for d in r['drafts']:
            if d['draft_id'] in draft_ids:raise ValueError('Duplicate draft ID')
            draft_ids.add(d['draft_id'])
            if d.get('status')!='MACHINE_DRAFT_UNREVIEWED' or d.get('label_source')!='MODEL_NOT_HUMAN' or d.get('physics_permission') is not False:
                raise ValueError('Unexpected expert or physical assertion')
            p=np.asarray(d['points_xy'],dtype=float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<12 or not np.isfinite(p).all() or p.min()<0 or p.max()>512:
                raise ValueError('Invalid candidate path')
        rel_ids=set();local_ids={d['draft_id'] for d in r['drafts']}
        for rel in r.get('relation_drafts',[]):
            if rel['relation_id'] in rel_ids or rel['candidate_a'] not in local_ids or rel['candidate_b'] not in local_ids or rel['candidate_a']==rel['candidate_b'] or rel['status']!='MACHINE_POSSIBLE_LINK_UNKNOWN' or rel['physical_join_allowed'] is not False:
                raise ValueError('Untrusted machine relation label')
            rel_ids.add(rel['relation_id'])
    if len(photos)!=39 or sum('start_xy_OPERATOR_REAL' in r for r in photos)!=9:raise ValueError('Incomplete photo context')
    return True
