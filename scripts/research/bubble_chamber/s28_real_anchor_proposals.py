#!/usr/bin/env python3
"""S28 original-AZ assisted continuation proposals from genuine H50Z START points.

An operator `start` is a real anchor but NOT a full line mask. Paths are
unverified 2D suggestions only and NEVER physical identities or automatic
joins. Checkpoint had only synthetically injected track/scratch supervision.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s28_anchor_tracker import evidence_from_photo,candidate_paths
from bubble_chamber.s27_overlap_inference import overlap_predict_global_features
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint
from bubble_chamber.pilot import fit_quadratic
from scripts.research.bubble_chamber.s27_overlap_photo_audit import sha256


def seed_for_crop(record,labels):
    parent=record['source_bmp_file']
    if parent not in labels:raise ValueError('missing source authentic human start')
    marks=[p for p in labels[parent]['keypoints'] if p['label']=='start']
    if not marks:raise ValueError('no true operator START point')
    ox,oy=int(record['x']),int(record['y'])
    near=min(marks,key=lambda r:(r['x_px']-ox-256)**2+(r['y_px']-oy-256)**2)
    sx=float(near['x_px']-ox);sy=float(near['y_px']-oy)
    if (sx-256)**2+(sy-256)**2>14**2:
        raise ValueError('source crop not actually centered around operator start')
    return (sx,sy),near


def render_overlay(img,seed,candidates):
    bgr=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    colors=[(50,240,80),(80,190,250),(255,120,100),(210,70,200),(90,240,230)]
    for i,c in enumerate(candidates):
        pts=np.rint(np.asarray(c['path_points_original_xy'])).astype('int32').reshape(-1,1,2)
        cv2.polylines(bgr,[pts],False,colors[i%len(colors)],1,cv2.LINE_AA)
        cv2.putText(bgr,str(i+1),tuple(pts[-1,0]),cv2.FONT_HERSHEY_SIMPLEX,.48,colors[i%len(colors)],1,cv2.LINE_AA)
    cv2.circle(bgr,tuple(np.rint(seed).astype(int)),5,(255,255,255),1,cv2.LINE_AA)
    cv2.circle(bgr,tuple(np.rint(seed).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
    return bgr


def run(args):
    torch.set_num_threads(2)
    labels=json.loads(args.labels.read_text(encoding='utf-8-sig'))
    audit=json.loads(args.roi_manifest.read_text(encoding='utf-8'))
    if audit.get('schema')!='s26b_full_frame_provenance_bundle_v1':raise ValueError('unsupported source ROI metadata')
    data=[r for r in audit['review_crops'] if r['stratum']=='start_anchor_nearby']
    if len(data)!=9:raise ValueError('expected all nine operator START crops on two films')
    model,prep=model_for_checkpoint(args.guided_az)
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in data:
        path=args.crops/(row['crop_id']+'.png')
        if sha256(path)!=row['source_crop_sha256']:
            raise ValueError('actual photo ROI sha256 does not match metadata')
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape!=(512,512):raise ValueError('real grayscale crop missing')
        seed,point=seed_for_crop(row,labels)
        az=overlap_predict_global_features(im,model,prep,stride=32)
        classical=candidate_paths(evidence_from_photo(im),seed,length=190,width=48,top_k=4)
        azguided=candidate_paths(evidence_from_photo(im,az),seed,length=190,width=48,top_k=4)
        for group in (classical,azguided):
            for c in group:
                pts=np.asarray(c['path_points_original_xy'])
                try:c['quadratic_PCA_local_pixels_UNVERIFIED']=fit_quadratic(pts[:,::-1])
                except (ValueError,TypeError,np.linalg.LinAlgError):
                    c['quadratic_PCA_local_pixels_UNVERIFIED']=None
        side=np.hstack([cv2.cvtColor(im,cv2.COLOR_GRAY2BGR),render_overlay(im,seed,classical),render_overlay(im,seed,azguided)])
        cv2.putText(side,'ORIGINAL                         PHOTOMETRY+RIDGE                      ORIGINAL ANZA ASSISTED',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(args.out/(row['crop_id']+'_S28_anchor_review.jpg')),side,[cv2.IMWRITE_JPEG_QUALITY,90])
        record={'crop_id':row['crop_id'],'photo_id':row['source_photo'],
                'source_crop_sha256':sha256(path),'source_checkpoint_sha256':sha256(args.guided_az),
                'genuine_OPERATOR_start_xy_in_crop':[round(seed[0],2),round(seed[1],2)],
                'start_is_not_full_track_label':True,
                'classical_candidate_paths':classical,'original_AZUNet_assisted_candidate_paths':azguided,
                'automatic_confirmed_track_pixels':0,'automatic_accepted_joins':0,
                'status':'ONLY_EXPERT_REVIEW_OPTIONS_NO_PHYSICAL_IDENTIFICATION'}
        records.append(record)
        print('S28 native START',row['crop_id'],'top1 directions',round(classical[0]['direction_angle_deg']) if classical else None,round(azguided[0]['direction_angle_deg']) if azguided else None,flush=True)
    summary={'schema':'s28_genuine_h50z_operator_anchor_to_unverified_2D_path_v1',
             'real_source_photos':['005','006'],'same_run_and_view':True,
             'real_operator_start_anchors':len(records),
             'photo_native_track_segment_masks':0,'linked_physical_particle_identities':0,
             'classical_path_proposals':sum(len(x['classical_candidate_paths']) for x in records),
             'ANZA_original_architecture_assisted_path_proposals':sum(len(x['original_AZUNet_assisted_candidate_paths']) for x in records),
             'parameter_source':'fixed candidate tracing heuristic, NOT fitted on real ground-truth line masks',
             'records':records}
    (args.out/'S28_real_operator_START_path_proposals.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    for name in ('roi_manifest','labels','crops','guided_az','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
