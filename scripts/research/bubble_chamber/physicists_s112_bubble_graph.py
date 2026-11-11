#!/usr/bin/env python3
"""S112: six ORIGINAL-H50Z ROI, source bead proposal + reversible grouping.

Existing S109-S111 curves are never overwritten. Not calibrated precision or
physical event IDs. The dots are true measured image peaks, but whether those
peaks correspond to ionization bubbles is NOT established.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s112_bubble_graph import (BubbleConfig,detect_bubble_centers,
                            extend_curves_from_bubbles,fit_free_bead_chains,audit_s111_connector_bubbles,propose_s112_group_bridges,audit_bead_bridge_against_photo)
from bubble_chamber.s112_review_ui import make_review_html
from bubble_chamber.s112_expert_grouping import merge_reviewed_groups,reviewed_groups_svg,validate_review_provenance
CASES=('005_start2_1200','005_start3_1200','005_start5_1200',
       '006_start1_1200','006_start2_1200','006_start3_1200')


def render(gray,points,extensions,free):
    bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    # Tiny purple rings are candidate bubble centers, NOT claimed ground truth.
    for p in points:
        x=int(round(p['x']));y=int(round(p['y']))
        color=(190,125,220) if p['near_S88_source_mask'] else (115,160,160)
        cv2.circle(bgr,(x,y),1,color,-1,cv2.LINE_AA)
    for item in extensions:
        blue=item['status']=='AUTO_REVIEW_CANDIDATE_NOT_GT'
        # A smooth line through dots is geometry, not the photo observation.
        color=(40,180,255) if blue else (240,110,240)
        q=np.asarray(item['preview_spline_xy'],np.float32)
        if len(q)<2:continue
        q=np.rint(q).astype(np.int32)
        for k in range(0,len(q)-1,21):
            sub=q[k:k+11]
            if len(sub)>1:cv2.polylines(bgr,[sub.reshape(-1,1,2)],False,color,1,cv2.LINE_AA)
        for xy in item['bubble_xy']:
            cv2.circle(bgr,tuple(np.rint(xy).astype(int)),3,(10,90,255) if blue else (255,50,200),1)
    for item in free:
        q=np.rint(np.asarray(item['preview_spline_xy'],np.float32)).astype(np.int32)
        if len(q)<2:continue
        for k in range(0,len(q)-1,20):
            sub=q[k:k+9]
            if len(sub)>1:cv2.polylines(bgr,[sub.reshape(-1,1,2)],False,(55,255,90),1,cv2.LINE_AA)
    return bgr


def render_clean(base,extensions,free,include_uncertain=True,include_free=True):
    """Clean overlay: no thousands of independent dot proposals are painted."""
    bgr=base.copy()
    if len(bgr.shape)==2:bgr=cv2.cvtColor(bgr,cv2.COLOR_GRAY2BGR)
    pieces=list(extensions)+list(free if include_free else [])
    for item in pieces:
        status=item['status']
        valid=status=='AUTO_REVIEW_CANDIDATE_NOT_GT'
        if not include_uncertain and not valid:continue
        color=(0,140,255) if valid else (155,45,220) if status!='UNANCHORED_BEAD_CHAIN_REVIEW_ONLY' else (60,170,80)
        q=np.rint(np.asarray(item['preview_spline_xy'],float)).astype(np.int32)
        for k in range(0,len(q)-1,20):
            short=q[k:min(k+9,len(q))]
            if len(short)>=2:cv2.polylines(bgr,[short.reshape(-1,1,2)],False,color,2,cv2.LINE_AA)
        # Distinguish observed bead centers from inferred Bezier geometry.
        for xy in item['bubble_xy']:
            cv2.circle(bgr,tuple(np.rint(xy).astype(int)),2,color,-1,cv2.LINE_AA)
    return bgr


def write_svg(gray,extensions,free):
    h,w=gray.shape
    rows=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
          '<title>S112 candidate photographed bead center chains (never confirmed particle IDs)</title>']
    for kind,items in (('anchored',extensions),('free',free)):
        for it in items:
            q=np.asarray(it['preview_spline_xy'],float)
            if len(q)<2:continue
            d='M '+' '.join([f'{p[0]:.3f},{p[1]:.3f}' if i==0 else f'L {p[0]:.3f},{p[1]:.3f}' for i,p in enumerate(q[::max(1,len(q)//450)])])
            col='#ffb828' if kind=='anchored' and it['status']=='AUTO_REVIEW_CANDIDATE_NOT_GT' else '#e83fc2' if kind=='anchored' else '#36a455'
            rows.append(f'<path id="{it["path_id"]}" data-type="{kind}" d="{d}" stroke="{col}" stroke-width="1.5" stroke-dasharray="7 5" fill="none"/>')
            for x,y in it['bubble_xy']:
                rows.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="1.8" fill="{col}"/>')
    rows.append('</svg>')
    return '\n'.join(rows)


def run_one(root:Path,out:Path,name:str,cfg:BubbleConfig=BubbleConfig(),decisions=None):
    bas=root/'s108_results_all'/name
    g=cv2.imread(str(bas/'06_SOURCE_ORIGINAL_GRAY.png'),cv2.IMREAD_GRAYSCALE)
    m=cv2.imread(str(bas/'01_S88_SOURCE_RIDGE_ONLY.png'),cv2.IMREAD_GRAYSCALE)
    b=cv2.imread(str(bas/'07_BLOCKED_SOURCE.png'),cv2.IMREAD_GRAYSCALE)
    old=root/'s111_global_results'/name/'global_tracks.json'
    if any(z is None for z in (g,m,b)):raise FileNotFoundError(name)
    s111=json.loads(old.read_text(encoding='utf-8'))
    sha111=hashlib.sha256(old.read_bytes()).hexdigest()
    mask=m>0;blocked=b>0
    points,rep=detect_bubble_centers(g,mask,blocked,cfg)
    ext,extrep=extend_curves_from_bubbles(s111['curves'],points,blocked,cfg,gray=g,source_mask=mask)
    free,freerep=fit_free_bead_chains(points,s111['curves'],blocked,cfg)
    joins,joins_rep=audit_s111_connector_bubbles(s111['association']['edge_proposals'],points,cfg)
    bycur={cid:t['global_track_id'] for t in s111['association']['tracks'] for cid in t['member_curve_ids']}
    for item in ext:item['attached_to_s111_global_track_id']=bycur.get(item['curve_id'])
    candidate_group_bridges,bridge_rep=propose_s112_group_bridges(ext,points)
    for candidate in candidate_group_bridges:
        candidate['photo_audit']=audit_bead_bridge_against_photo(candidate,g,mask,blocked)
        if (candidate['status']=='BEAD_SUPPORTED_INTERGROUP_BRIDGE_REVIEW_ONLY' and
            candidate['photo_audit']['status']!='PHOTO_GAP_CONTRAST_PASS_REVIEW_ONLY'):
            candidate['status']='PHOTO_CONFLICT_REVIEW_ONLY_NO_AUTO_MERGE'
    bridge_rep['photo_contrast_satisfying_review_bridges']=sum(
        z['status']=='BEAD_SUPPORTED_INTERGROUP_BRIDGE_REVIEW_ONLY' for z in candidate_group_bridges)
    if decisions:
        for item in ext+free:
            answer=decisions.get(item['path_id'])
            if answer in ('same','different','uncertain'):
                item['manual_review']={'expert_judgement':answer,
                                      'original_machine_status':item['status'],
                                      'not_independent_physics_validation':True}
                if answer=='different':item['status']='EXPERT_REJECTED_CHAIN'
                if answer=='same':item['status']='EXPERT_APPROVED_VISUAL_BEAD_CHAIN_NOT_PHYSICAL_GT'
    target=out/name;target.mkdir(parents=True,exist_ok=True)
    payload={'schema':'S112_IMAGE_BUBBLE_CENTERS_AND_REVIEWABLE_BEAD_CHAINS_V1',
             'source_roi':name,'source_sha256':s111['source_sha256'],
             's111_global_tracks_sha256':sha111,
             'S111_group_ids_unchanged':True,'source_gray_modified':False,
             'observations_are_floating_point_dark_local_maxima_not_GT':True,
             'inferred_preview_curves_not_photo_observations':True,
             'physical_particle_id_proven':False,
             'detector':rep,'anchor_extension_audit':extrep,'free_chain_audit':freerep,
             's111_connector_bubble_audit':joins_rep,'s111_gap_bead_support_for_review':joins,
             's112_intergroup_bridge_audit':bridge_rep,'intergroup_review_only_bridge_proposals':candidate_group_bridges,
             'bubble_centers':points,'anchored_bead_chain_proposals':ext,
             'unanchored_review_only_tracklets':free}
    (target/'bubble_graph_not_gt.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (target/'bead_chains_NOT_PHOTO.svg').write_text(write_svg(g,ext,free),encoding='utf-8')
    # All dots are photometric proposals, not expert-confirmed bubbles.
    zeros=np.zeros_like(g)
    for p in points:
        x=int(round(p['x']));y=int(round(p['y']))
        if 0<=x<g.shape[1] and 0<=y<g.shape[0]:zeros[y,x]=255
    cv2.imwrite(str(target/'01_photometric_dot_centers_NOT_GT.png'),zeros)
    cv2.imwrite(str(target/'02_review_paths_DASHED.png'),render(g,points,ext,free))
    previous=cv2.imread(str(root/'s111_global_results'/name/'GLOBAL_TRACK_GROUPS_WITH_DASHED_LINKS.png'))
    if previous is None:raise FileNotFoundError(f'S111 comparison missing: {name}')
    cv2.imwrite(str(target/'03_S111_PLUS_S112_REVIEWABLE.png'),render_clean(previous,ext,free))
    cv2.imwrite(str(target/'04_S112_CLEAN_SOURCE_REVIEW.png'),render_clean(g,ext,free))
    cv2.imwrite(str(target/'05_S112_ONLY_UNAMBIGUOUS_ANCHORS.png'),
                render_clean(g,ext,free,include_uncertain=False,include_free=False))
    (target/'expert_bubble_review.html').write_text(make_review_html(name,ext,free,joins,g.shape[1],g.shape[0],bridges=candidate_group_bridges,source_sha256=s111['source_sha256'],s111_sha256=sha111),encoding='utf-8')
    review_grouping=merge_reviewed_groups(s111['association']['tracks'],candidate_group_bridges,decisions)
    (target/'expert_only_merged_groups.json').write_text(json.dumps(review_grouping,ensure_ascii=False,indent=2),encoding='utf-8')
    oldsvg=(root/'s111_global_results'/name/'grouped_vector_tracks.svg').read_text(encoding='utf-8')
    (target/'S111_PRESERVED_WITH_EXPERT_APPROVED_BRIDGES.svg').write_text(
       reviewed_groups_svg(oldsvg,candidate_group_bridges,decisions or {}),encoding='utf-8')
    report={'roi':name,**rep,**extrep,**freerep,**joins_rep,**bridge_rep,
            'groups_after_explicit_expert_review':review_grouping['groups_after_explicit_review'],
            'S111_accepted_link_count_UNCHANGED':s111['association']['association_report']['selected_link_count'],
            'physical_particle_id_verified':0,'no_trained_network_ran':True,
            'runtime_for_S112':'cpu','previous_curves_unchanged':True}
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def run(root:Path,out:Path,cases=CASES,decisions=None):
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name in cases:
        d=run_one(root,out,name,decisions=decisions.get(name,{}) if decisions else None)
        reports.append(d)
        print(name,'dot proposals',d['candidate_dark_centers'],'anchors',d['endpoints_with_supported_bead_path'],
              'unanchored',d['unanchored_review_only_tracklet_count'],flush=True)
    (out/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    return reports

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,default=None);ap.add_argument('--cases',nargs='*',choices=CASES,default=list(CASES))
    ap.add_argument('--decisions',type=Path,default=None,help='review JSON with source_roi; never change raw pixels')
    a=ap.parse_args()
    dec=None
    if a.decisions:
        review=json.loads(a.decisions.read_text(encoding='utf-8'))
        roi=review.get('source_roi')
        if roi not in CASES:raise ValueError('Source ROI missing or unexpected in expert-review JSON')
        jf=a.root/'s111_global_results'/roi/'global_tracks.json'
        s111=json.loads(jf.read_text(encoding='utf-8'))
        validate_review_provenance(review,s111['source_sha256'],hashlib.sha256(jf.read_bytes()).hexdigest())
        entries=review.get('review_decisions',{})
        if not isinstance(entries,dict) or any(z not in ('same','different','uncertain') for z in entries.values()):raise ValueError('Invalid review decision values')
        dec={roi:entries}
    run(a.root,a.out or a.root/'s112_bubble_results',a.cases,dec)