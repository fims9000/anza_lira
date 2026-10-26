#!/usr/bin/env python3
"""S45 machine-only uncertainty on two native registered H50Z photos.

PRIVATE image coordinates, fiducial affine, S39 machine drafts, and masks
remain OUTSIDE Git. PUBLIC aggregate contains only counts/statistics.
This transductive annotation MUST NOT train leave-one-photo-out models.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s44_crossframe_persistence import (
    ExactBMP8, fit_common_fiducials, warp_corresponding_window)
from bubble_chamber.s45_registered_unknown import (
    SCHEMA, MIN_COMMON_SUPPORT, MIN_DISPLACED_MARGIN,
    registered_uncertainty, downgrade_for_diagnostic_only, graph_repeat_warning,
    combine_with_known_fiducial_guard)


def _validate_source(s39:dict, s44:dict, graphs:dict, s40_graphs:dict):
    if (len(s39.get('tiles',[]))!=39 or s39.get('native_expert_gt_created')!=0
            or len(s44.get('registered_crops_PRIVATE',[]))!=39
            or len(graphs.get('graphs',[]))!=39
            or len(s40_graphs.get('graphs',[]))!=39):
        raise ValueError('Private source dataset mismatch')
    if (sum(len(t['proposals']) for t in s39['tiles'])!=567
            or sum(len(t['private_candidate_evidence']) for t in s44['registered_crops_PRIVATE'])!=567
            or sum(len(g['candidate_edges']) for g in graphs['graphs'])!=182):
        raise ValueError('Source S39/S44 machine inventory changed')
    src=[t['crop_id'] for t in s39['tiles']]
    if len(set(src))!=39 or src!=[t['crop_id'] for t in s44['registered_crops_PRIVATE']] or src!=[g['crop_id'] for g in graphs['graphs']]:
        raise ValueError('Private inventories not aligned by crop')


def _make_overlay(src:np.ndarray, aligned:np.ndarray, row:dict, decisions:list,
                  prot:np.ndarray, out:Path):
    source=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    counterpart=cv2.cvtColor(aligned,cv2.COLOR_GRAY2BGR)
    for p,d in zip(row['proposals'],decisions):
        xy=np.rint(np.asarray(p['points_xy'])).astype('int32')
        if d['registered_repeat_risk']:
            cv2.polylines(source,[xy],False,(25,150,250),2,cv2.LINE_AA)
            cv2.polylines(counterpart,[xy],False,(25,150,250),2,cv2.LINE_AA)
        else:
            cv2.polylines(source,[xy],False,(55,195,75),1,cv2.LINE_AA)
    # Colored *unknown* region is shown as a thin contour, not a deletion.
    contours,_=cv2.findContours(prot.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(source,contours,-1,(255,60,60),1)
    cv2.putText(source,'NATIVE SOURCE / ORANGE: REPEAT CUE',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,240,240),2)
    cv2.putText(counterpart,'REGISTERED OTHER PHOTO',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,240,240),2)
    im=cv2.hconcat([source,counterpart])
    cv2.imwrite(str(out),im,[cv2.IMWRITE_JPEG_QUALITY,83])


def _ci(rows:list[dict], key:str,seed:int=20261045)->list[float] | None:
    if len(rows)<5:return None
    data=np.asarray([r[key] for r in rows],dtype=float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(data),(3000,len(data)))
    return np.quantile(data[idx].mean(1),[.025,.975]).round(5).tolist()


def run(bmps:Path,s39_path:Path,s44_path:Path, s39_graph:Path,
        s40_graph:Path,s42_pseudos:Path,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    overviews=out/'PRIVATE_REVIEW_IMAGES';overviews.mkdir(exist_ok=True)
    masks=out/'PRIVATE_TRANSDUCTIVE_DIAGNOSTIC_MASKS';masks.mkdir(exist_ok=True)
    s39=json.loads(s39_path.read_text())
    s44=json.loads(s44_path.read_text())
    graphs=json.loads(s39_graph.read_text())
    guard_graphs=json.loads(s40_graph.read_text())
    _validate_source(s39,s44,graphs,guard_graphs)
    reg=fit_common_fiducials(json.loads((bmps/'labels.json').read_text()))
    affine=np.asarray(s44['forward_affine_PRIVATE'])
    if not np.allclose(affine,reg['forward_005_to_006_PRIVATE'],atol=1e-5):
        raise ValueError('S44 registration not proven on source labels')
    forwards={'005':affine,'006':cv2.invertAffineTransform(affine)}
    images={k:ExactBMP8(bmps/f'H50Z_030_01_{k}.bmp') for k in ('005','006')}
    rows=[];totals=Counter();perphoto={k:Counter() for k in images}
    private=[]
    for item,evidence,g,prior_guard in zip(s39['tiles'],s44['registered_crops_PRIVATE'],
                                           graphs['graphs'],guard_graphs['graphs']):
        if item['crop_id']!=prior_guard['crop_id']:
            raise ValueError('S40 graph not aligned with S39')
        key=item['photo_id'];other='006' if key=='005' else '005'
        src=images[key].crop(*item['native_origin_xy_PRIVATE'],512,512)
        if (sha256(src.tobytes()).hexdigest()!=item['source_gray_sha256'] or
                evidence['source_sha256']!=item['source_gray_sha256'] or
                evidence['overlap_status']!='BOTH_IMAGES_VALID'):
            raise ValueError('Native image provenance is not confirmed')
        aligned=warp_corresponding_window(images[other],item['native_origin_xy_PRIVATE'],forwards[key])
        protect,decisions=registered_uncertainty(src,aligned,item['proposals'],evidence['private_candidate_evidence'])
        original=cv2.imread(str(s42_pseudos/(item['crop_id']+'_S42_SOURCE_PURE_PSEUDO.png')),
                            cv2.IMREAD_UNCHANGED)
        if original is None or original.shape!=src.shape or original.dtype!=np.uint8:
            raise ValueError('Source-only S42 machine mask missing')
        refined,changes=downgrade_for_diagnostic_only(original,protect)
        if np.any((original==255)&(refined!=255)) or not np.isin(refined,[0,1,255]).all():
            raise AssertionError('Pseudo label promotion forbidden')
        cv2.imwrite(str(masks/(item['crop_id']+'_S45_NOT_FOR_TRAINING.png')),refined)
        cid_to_risk={d['draft_id']:d['registered_repeat_risk'] for d in decisions}
        relation=graph_repeat_warning(g,cid_to_risk)
        combined=combine_with_known_fiducial_guard(
            prior_guard['machine_graph_risk_annotation_PRIVATE'],relation)
        if relation['original_edge_count']!=len(g['candidate_edges']):
            raise AssertionError('S39 graph silently modified')
        riskcount=sum(cid_to_risk.values())
        totals.update({'crops':1,'source_proposals':len(decisions),
                       'registered_repeat_risk_proposals':riskcount,
                       'risk_pixels':int(protect.sum()),
                       'source_pseudo_positives':changes['before']['positive'],
                       'source_pseudo_assumed_negatives':changes['before']['assumed_negative'],
                       'source_pseudo_unknown':changes['before']['unknown'],
                       'pseudo_pos_downgraded':changes['positive_demoted_to_unknown'],
                       'pseudo_neg_downgraded':changes['negative_demoted_to_unknown'],
                       'original_graph_edges':relation['original_edge_count'],
                       'repeat_advisory_edges':relation['registered_repeat_edge_count'],
                       'original_graph_nodes':relation['node_count'],
                       'repeat_advisory_nodes':relation['registered_repeat_node_count'],
                       'combined_risk_union_edges':combined['union_any'],
                       'new_repeat_edges_not_already_fiducial_risk':combined['new_repeat_not_fiducial'],
                       'preliminary_unambiguous_union_risk':combined['previously_ranked_unambiguous_union_risk']})
        perphoto[key].update({'crops':1,'proposals':len(decisions),'repeat_risk_proposals':riskcount,
                        'pseudo_pos_downgraded':changes['positive_demoted_to_unknown'],
                        'pseudo_neg_downgraded':changes['negative_demoted_to_unknown'],
                        'graph_edges':relation['original_edge_count'],
                        'repeat_risk_edges':relation['registered_repeat_edge_count']})
        rows.append({'photo':key,'source_positive':changes['before']['positive'],
                     'risk_rate_per_candidate':riskcount/max(1,len(decisions)),
                     'positive_downgrade_rate':changes['positive_demoted_to_unknown']/max(1,changes['before']['positive']),
                     'edge_risk_rate':relation['registered_repeat_edge_count']/max(1,relation['original_edge_count'])})
        name=item['crop_id']+'_S45_SIDE_BY_SIDE.jpg'
        _make_overlay(src,aligned,item,decisions,protect,overviews/name)
        private.append({'crop_id':item['crop_id'],'photo_id':key,
                        'source_mask_sha256_PRIVATE':sha256(original.tobytes()).hexdigest(),
                        'diagnostic_mask_sha256_PRIVATE':sha256(refined.tobytes()).hexdigest(),
                        'candidate_machine_evidence_PRIVATE':decisions,
                        'graph_machine_evidence_PRIVATE':relation,
                        'combined_guard_machine_summary_PRIVATE':combined,
                        'pixel_changes_PRIVATE':changes})
        if len(rows)%5==0:print('S45 native processed',len(rows),'/',len(s39['tiles']),flush=True)
    report={
        'schema':SCHEMA,'source_photographs':2,'independent_physical_runs':1,
        'source_crops':len(rows),'candidate_total_preserved':totals['source_proposals'],
        'graph_proposals_total_preserved':totals['original_graph_edges'],
        'new_repeat_edges_beyond_S40_fiducial_guard':totals['new_repeat_edges_not_already_fiducial_risk'],
        'combined_S40_S45_ambiguous_graph_edges':totals['combined_risk_union_edges'],
        'other_graph_edges_still_native_unknown':totals['original_graph_edges']-totals['combined_risk_union_edges'],
        'fixed_heuristic_not_physics_calibrated':{'min_both_registered_ridge_coverage':MIN_COMMON_SUPPORT,
                                                    'min_vs_displaced_control_margin':MIN_DISPLACED_MARGIN},
        'counts':dict(totals),'by_photo':{k:dict(v) for k,v in perphoto.items()},
        'crop_bootstrap_95pct_CI_exploratory_not_independent_runs':{
            'fraction_of_original_machine_pos_downgraded':_ci(rows,'positive_downgrade_rate'),
            'fraction_candidate_repeat_risk':_ci(rows,'risk_rate_per_candidate'),
            'fraction_graph_edges_repeat_advisory':_ci(rows,'edge_risk_rate')},
        'source_only_geometry_mask':'S42 frozen photo-only Hessian pseudo, not S39 AZ teacher',
        'crossframe_risk_used_for_training':False,
        'S45_both_photos_seen_for_inference_transductive':True,
        'leave_one_photo_out_or_independent_photo_accuracy_claim':False,
        'native_GT_count':0,'native_particle_classification_performed':False,
        'native_candidates_removed':0,'physical_graph_joins_performed':0,
        'machine_repeat_cue_is_not_evidence_of_scratch_identity':True,
    }
    (out/'S45_PUBLIC_AGGREGATES.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (out/'S45_PRIVATE_ADVISORY_DETAILS.json').write_text(json.dumps({
        'schema':SCHEMA,'source_SHA256_s39':sha256(s39_path.read_bytes()).hexdigest(),
        'source_SHA256_s44':sha256(s44_path.read_bytes()).hexdigest(),
        'native_crop_machine_only':private,'physical_GT_count':0},indent=2,ensure_ascii=False)+'\n')
    # Offline visualization, not a labeling application.
    tiles='\n'.join('<article><h3>'+html.escape(item['crop_id'])+
        '</h3><img loading="lazy" src="PRIVATE_REVIEW_IMAGES/'+html.escape(item['crop_id'])+
        '_S45_SIDE_BY_SIDE.jpg"><p>Orange: repeated-structure advisory; green: other machine hypotheses. '
        'Blue outlines: uncertain overlap, not erased. No physics GT.</p></article>'
        for item in s39['tiles'])
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8">'
        '<title>ANZA S45 - machine-only registered uncertainty</title><style>body{font-family:sans-serif;'
        'background:#101923;color:#e9eff6;margin:24px}img{width:100%;max-width:1024px}'
        'article{margin:22px 0;padding:12px;background:#182434;border-radius:8px}'
        'p{color:#a7c1d6}</style></head><body><h1>ANZA-LIRA S45 - image repeat cues</h1>'
        '<p>39 crops; all machine proposals preserved; no physical truth or auto-join. '
        'Other native photo viewed for transductive uncertainty only. '
        'Do not use these masks to train a leave-one-photo-out model.</p>'+
        tiles+'</body></html>',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for a in ('bmp-directory','s39-private','s44-private','s39-graph','s40-graph','s42-pseudos','out'):
        p.add_argument('--'+a,required=True,type=Path)
    a=p.parse_args()
    run(a.bmp_directory,a.s39_private,a.s44_private,a.s39_graph,a.s40_graph,a.s42_pseudos,a.out)
if __name__=='__main__':main()
