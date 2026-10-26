#!/usr/bin/env python3
"""Render S49 standalone PRIVATE machine-only physicist-facing demonstration.

The public code ships with NO image data. A private S48 provenance manifest
is mandatory. Pixel and centerline content stays in the private output only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s44_crossframe_persistence import ExactBMP8,warp_corresponding_window
from bubble_chamber.s48_registration_jackknife import leave_one_fiducial_out,photo_transforms

TEMPLATE=Path(__file__).with_name('s49_offline_template.html')


def build_payload(s39: dict, s48_private: dict, s39_graph: dict, s48_aggregate: dict) -> dict:
    tiles=s39.get('tiles')
    private=s48_private.get('machine_crops_PRIVATE')
    graphs=s39_graph.get('graphs')
    if not all(isinstance(x,list) and len(x)==39 for x in (tiles,private,graphs)):
        raise ValueError('S39/S48 private provenance missing')
    if s39.get('native_expert_gt_created')!=0:
        raise ValueError('Unexpected native human GT')
    rows=[]
    counts={'crops':0,'machine_candidates':0,'graph_edges':0,'risky_edges':0}
    for i,(t,p,g) in enumerate(zip(tiles,private,graphs)):
        if t.get('crop_id')!=p.get('crop_id') or t.get('crop_id')!=g.get('crop_id'):
            raise ValueError('Mixing private crop identities from separate source runs')
        candidates=t.get('proposals',[])
        decisions=p.get('candidate_evidence_PRIVATE',[])
        edges=g.get('candidate_edges',[])
        annotated=p.get('graph_evidence_PRIVATE',[])
        if len(candidates)!=len(decisions) or len(edges)!=len(annotated):
            raise ValueError('Incomplete S48 uncertainty provenance')
        drawn=[]
        for orig,d in zip(candidates,decisions):
            if orig.get('draft_id')!=d.get('draft_id'):
                raise ValueError('Wrong centerline/risk identity')
            drawn.append({'draft_id':orig['draft_id'],
                          'points':orig['points_xy'],
                          'status':d['registration_status'],
                          'source_method':orig.get('source_method')})
        proposed=[]
        for original,warning in zip(edges,annotated):
            if (original['i'],original['j'])!=(warning.get('i'),warning.get('j')):
                raise ValueError('Unverifiable graph origin')
            if original.get('physics_join_authorized') is not False or warning.get('physical_auto_join_authorized') is not False:
                raise ValueError('Physical auto joining must remain OFF')
            proposed.append({'i':original['i'],'j':original['j'],
                             'bridge':original['hermite_bridge_pixels_PROPOSAL_ONLY'],
                             'risky':bool(warning['union_advisory']),
                             'geometry_rank_UNCALIBRATED':original.get('geometry_rank_UNCALIBRATED')})
        counts['crops']+=1
        counts['machine_candidates']+=len(drawn)
        counts['graph_edges']+=len(proposed)
        counts['risky_edges']+=sum(x['risky'] for x in proposed)
        rows.append({'crop_id':t['crop_id'],'photo':t['photo_id'],
                     'source_image':f"images/S49_src_{i:02d}.jpg",
                     'other_image':f"images/S49_other_{i:02d}.jpg",
                     'candidates':drawn,'graph_edges':proposed})
    expected=s48_aggregate['S48_native_machine']
    assert_counts={'machine_candidates':expected['machine_candidate_counts']['all_original_machine_candidates_preserved'],
                   'graph_edges':expected['graph_counts']['edge_count'],
                   'risky_edges':expected['graph_counts']['union_any']}
    if any(counts[k]!=v for k,v in assert_counts.items()):
        raise ValueError('Aggregate mismatch: do not create physics showcase')
    return {'schema':'s49_offline_machine_only_private_demo_v1',
            'scientific_interpretation':'UNVERIFIED_MACHINE_HYPOTHESES',
            'requires_both_images':True,'independent_physical_runs':1,
            'native_GT_count':0,'physical_auto_links':0,
            'allowed_to_use_as_heldout_photo_training_GT':False,
            'summary':counts,'tiles':rows}


def make_demo(bmpdir:Path,s39file:Path,s48file:Path,s39graph:Path,
              aggregate:Path,out:Path,public_aggregate:Path):
    private_s39=json.loads(s39file.read_text())
    private_s48=json.loads(s48file.read_text())
    graph=json.loads(s39graph.read_text())
    s48_aggregate=json.loads(aggregate.read_text())
    payload=build_payload(private_s39,private_s48,graph,s48_aggregate)
    calibration=leave_one_fiducial_out(json.loads((bmpdir/'labels.json').read_text()))
    images={k:ExactBMP8(bmpdir/f'H50Z_030_01_{k}.bmp') for k in ('005','006')}
    out.mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    files=[]
    for i,tile in enumerate(private_s39['tiles']):
        src=tile['photo_id'];other='006' if src=='005' else '005'
        x,y=tile['native_origin_xy_PRIVATE']
        native=images[src].crop(x,y,512,512)
        if sha256(native.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Native S39 BMP hash does not match')
        matrix,_=photo_transforms(calibration,src)
        counterpart=warp_corresponding_window(images[other],(x,y),matrix)
        for img,name in ((native,f'S49_src_{i:02d}.jpg'),
                         (counterpart,f'S49_other_{i:02d}.jpg')):
            if not cv2.imwrite(str(out/'images'/name),img,[cv2.IMWRITE_JPEG_QUALITY,86]):
                raise RuntimeError('Failed to save private image')
            files.append(out/'images'/name)
    template=TEMPLATE.read_text(encoding='utf-8')
    if template.count('__INLINE_PAYLOAD__') !=1:
        raise ValueError('Offline template missing data insertion point')
    data=json.dumps(payload,separators=(',',':'),ensure_ascii=False).replace('</','<\\/')
    page=template.replace('__INLINE_PAYLOAD__',data)
    if '__INLINE_PAYLOAD__' in page or 'https://' in page or 'http://' in page:
        raise ValueError('Private demo must be completely offline')
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'S49_MACHINE_HYPOTHESES_PRIVATE.json').write_text(
        json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    checksums={p.relative_to(out).as_posix():sha256(p.read_bytes()).hexdigest()
               for p in [*files,out/'index.html',out/'S49_MACHINE_HYPOTHESES_PRIVATE.json']}
    (out/'S49_INTEGRITY_PRIVATE.json').write_text(json.dumps(checksums,indent=2)+'\n')
    public={'schema':'s49_offline_physics_demonstrator_public_summary_v1',
            'machine_only_no_GT':True,'independent_physics_runs':1,
            'private_image_count':len(files),'private_image_size':512,
            'image_integrity_verified_source_sha':39,
            'offline_assets_no_external_network':True,
            'browser_uses_inline_json_no_fetch_calls':True,
            'direct_file_URL_QA_not_permitted_in_runtime':True,
            'manual_labels_requested':0,
            'native_physical_auto_join_performed':0,
            'summary':payload['summary'],
            'transductive_photo_heldout_training_forbidden':True,
            'output_PRIVATE_does_not_belong_in_public_Git':True}
    public_aggregate.parent.mkdir(parents=True,exist_ok=True)
    public_aggregate.write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(public,indent=2),flush=True)
    return public


def main():
    p=argparse.ArgumentParser()
    for x in ('bmp-directory','s39-private','s48-private','s39-graph','s48-public','out','public-aggregate'):
        p.add_argument('--'+x,required=True,type=Path)
    a=p.parse_args()
    make_demo(a.bmp_directory,a.s39_private,a.s48_private,a.s39_graph,
              a.s48_public,a.out,a.public_aggregate)


if __name__=='__main__':main()
