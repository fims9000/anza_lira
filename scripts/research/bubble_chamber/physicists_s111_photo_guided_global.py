#!/usr/bin/env python3
"""Reconstruct S111 associations from two ORIGINAL H50Z films (six ROIs).

Not training, not an automated source of verified particle IDs; each connection
is reversible and marked as inferred even where black bubble dots support it.
"""
from __future__ import annotations

import argparse,hashlib,json,shutil,sys
from lxml import etree
import numpy as np
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s110_global_association import GlobalConfig
from bubble_chamber.s111_photo_guidance import associate_s111,PhotoGuideConfig
from scripts.research.bubble_chamber.physicists_s110_global_matching import (
    CASES, build_svg, render_png, review_html,
)


def _highlight_long_links_svg(svg_text:str,association:dict):
    doc=etree.fromstring(svg_text.encode('utf-8'))
    winners={e['candidate_id'] for e in association['edge_proposals']
             if e['decision'] in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME')
             and e['gap_px']>80.0}
    for node in doc.xpath('//*[@data-edge-id]'):
        if node.get('data-edge-id') in winners:
            for path in node.xpath('.//*[local-name()="path"]'):
                path.set('stroke','#f58c00')
                path.set('stroke-width','3')
                path.set('stroke-dasharray','7 5')
                path.set('opacity','.95')
    return etree.tostring(doc,encoding='unicode',xml_declaration=False)


def review_s111(envelope:dict,svg:str):
    document=review_html(envelope,svg)
    proposals=envelope['association']['edge_proposals']
    details=[]
    for e in proposals:
        details.append({'id':e['candidate_id'],'curves':e['curve_ids'],
            'gap':e['gap_px'],'score':e['score'],
            'photo':e.get('photo_support_fraction',0.0),
            'status':e['decision'],'original_s110_score':e.get('score_before_search_radius_calibration'),
            'long_gap_guard':e.get('long_gap_guard'),
            'refinement':e.get('refinement'),
            'auto_eligible':e.get('auto_eligible',True),
            'review_reason':e.get('s111_pre_matching_review_reason'),
            'photo_observation_confirmed':False})
    newjs=json.dumps(details,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')
    start=document.index('const links=')
    end=document.index(';const byid=',start)
    document=document[:start]+'const links='+newjs+document[end:]
    document=document.replace('S110', 'S111')
    document=document.replace('s110_global_matching.py','s111_photo_guided_global.py')
    document=document.replace('s110_review_decisions.json','s111_review_decisions.json')
    return document


def highlight_long_preview(image_path:Path,association:dict):
    img=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if img is None:raise IOError(image_path)
    for e in association['edge_proposals']:
        if e['decision'] not in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME') or e['gap_px']<=80:continue
        coords=np.rint(np.asarray(e['sampled_xy'],float)).astype(np.int32)
        # Visible orange DOTTED connectors, never a solid observation layer.
        for a in range(0,len(coords)-1,15):
            part=coords[a:min(a+8,len(coords))]
            if len(part)>1:cv2.polylines(img,[part.reshape(-1,1,2)],False,(0,140,255),2,cv2.LINE_AA)
    cv2.imwrite(str(image_path),img)


def run_case(root:Path,out:Path,roi:str,global_cfg:GlobalConfig,
             photo_cfg:PhotoGuideConfig=PhotoGuideConfig(),decisions=None):
    d109=root/'s109_vector_results'/roi
    data=json.loads((d109/'vector_tracks.json').read_text(encoding='utf-8'))
    base=root/'s108_results_all'/roi
    src=base/'06_SOURCE_ORIGINAL_GRAY.png'
    gray=cv2.imread(str(src),0)
    photo=cv2.imread(str(base/'01_S88_SOURCE_RIDGE_ONLY.png'),0)
    blocked=cv2.imread(str(base/'07_BLOCKED_SOURCE.png'),0)
    if gray is None or photo is None or blocked is None:raise FileNotFoundError(f'S108 source images missing: {roi}')
    result=associate_s111(data['curves'],photo>0,blocked>0,gray,
        expert_decisions=decisions,legacy_review=data.get('proposed_links_for_review',[]),
        global_cfg=global_cfg,guide_cfg=photo_cfg)
    envelope={
        'schema':'ANZA_S111_SOURCE_GUIDED_GLOBAL_ASSOCIATION_v1',
        'source_roi':roi,'source_sha256':data['source_sha256'],
        's109_sha256':hashlib.sha256((d109/'vector_tracks.json').read_bytes()).hexdigest(),
        'source_evidence':'S109 original cubic tracklets plus original H50Z S88 photometric pixels',
        'config':vars(global_cfg),'photo_guide_config':vars(photo_cfg),
        'curves':data['curves'],'association':result,
        'physical_particle_identity_proven':False,'all_automated_merges_reversible':True,
        'source_pixels_changed_by_s111':0}
    target=out/roi;target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,target/'SOURCE_GRAYSCALE.png')
    (target/'global_tracks.json').write_text(json.dumps(envelope,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (target/'grouped_vector_tracks.svg').write_text(build_svg(envelope,gray),encoding='utf-8')
    (target/'expert_global_review.html').write_text(review_html(envelope,build_svg(envelope,gray,include_unmatched=True)),encoding='utf-8')
    render_png(envelope,gray,target/'GLOBAL_TRACK_GROUPS_WITH_DASHED_LINKS.png')
    highlight_long_preview(target/'GLOBAL_TRACK_GROUPS_WITH_DASHED_LINKS.png',result)
    report={'roi':roi,'global':result['association_report'],
            'photo_guide':result['s111_photo_guide_report'],
            'candidate_generator':result['generator_audit'],'solver':result['solver_audit'],
            'all_source_observations_unchanged':True,'not_physical_ground_truth':True}
    (target/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main(args=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--cases',nargs='*',default=list(CASES),choices=CASES)
    parser.add_argument('--max-gap',type=float,default=220.0)
    parser.add_argument('--decisions',type=Path,default=None)
    a=parser.parse_args(args)
    cfg=GlobalConfig(max_gap_px=a.max_gap,min_endpoint_cos=.95,
                     min_photo_support_long=.31,min_photo_support_very_long=.65,
                     auto_candidate_score=.83)
    pcfg=PhotoGuideConfig()
    review={}
    if a.decisions:
        data=json.loads(a.decisions.read_text(encoding='utf-8'))
        review=data.get('review_decisions',data)
        if data.get('source_roi') and len(a.cases)>1:
            review={data['source_roi']:review}
    out=a.out or a.root/'s111_global_results';out.mkdir(parents=True,exist_ok=True)
    result=[]
    for roi in a.cases:
        decisions=review.get(roi,{}) if isinstance(review,dict) and any(k in CASES for k in review) else review
        rep=run_case(a.root,out,roi,cfg,pcfg,decisions)
        result.append(rep)
        print(roi,'groups',rep['global']['global_track_candidates'],'links',rep['global']['selected_link_count'],
              'refined',rep['photo_guide']['photo_guided_cubic_refinements'],flush=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

if __name__=='__main__':main()