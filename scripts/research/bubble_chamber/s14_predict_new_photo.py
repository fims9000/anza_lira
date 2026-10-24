#!/usr/bin/env python3
"""Unattended INFERENCE, not unattended physics certification, on a new BMP.

Uses classical P0 segment proposals as the current front-end. After enough
real sparse image reviews, the front-end may be replaced by original AZUNet.
Graph-LIRA portable predictor scores each proposed connection; structural
constraints prohibit duplicate endpoint use, multiple-branch cycles, etc.
A calibration receipt is mandatory before any proposed merges are emitted.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.pilot import PilotConfig,analyze_frame
from bubble_chamber.review import SCHEMA,result_digest
from bubble_chamber.s14_fewshot_graph import pair_rows_from_review,portable_predict
from bubble_chamber.graph_safety import select_global_links


def predict_image(image:np.ndarray,model_parameters:dict,*,film_id:str,run_id:str,frame_id:str,
                  calibration:dict|None=None,config:PilotConfig=PilotConfig(),
                  original_az_checkpoint:Path|None=None):
    if not film_id or not run_id or not frame_id:
        raise ValueError('explicit image provenance is required')
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError('expected grayscale uint8 photographic image')
    if original_az_checkpoint is not None:
        from bubble_chamber.s14_anza_frontend import load_original_az_checkpoint,az_mask_to_segments
        original_az,meta=load_original_az_checkpoint(original_az_checkpoint)
        original,mask,overlay=az_mask_to_segments(image,original_az)
    else:
        original,mask,overlay=analyze_frame(image,config)
    # No *human* annotations are generated: merely schema-compatible empty
    # candidate review, so the same source-bound orientation mapping is reused.
    empty_review={'schema':SCHEMA,'source_sha256_grayscale_pixels':original['source_sha256_grayscale_pixels'],
                  'pipeline':original['pipeline'],'result_sha256':result_digest(original),'labels':[]}
    pairs=pair_rows_from_review(original,empty_review,film_id=film_id,run_id=run_id,frame_id=frame_id)
    if pairs:
        probabilities=portable_predict(pairs,model_parameters)
    else:
        probabilities=np.zeros(0)
    threshold=None;valid=False
    if calibration:
        c=calibration.get('inference',calibration)
        if c.get('schema')!='s14_provisional_graph_lira_inference_v1':
            raise ValueError('unsupported calibration receipt schema')
        threshold=c.get('threshold_fitted_on_validation')
        valid=bool(c.get('different_run_calibration')) and int(c.get('calibration_frames',0))>=5 and threshold is not None
        if threshold is not None and not 0<=float(threshold)<=1.000001:
            raise ValueError('calibration threshold invalid')
    if valid and pairs:
        graph=[{'scene_id':r['scene_id'], 'pair':[int(r['left_segment']),'B',int(r['right_segment']),'A']}
               for r in pairs]
        accepted,reasons=select_global_links(graph,probabilities,threshold=float(threshold),margin=.08)
    else:
        accepted=[False]*len(pairs)
        reasons=['NO_INDEPENDENT_REAL_CALIBRATION']*len(pairs)
    track_candidates=[]
    for row,p,selected,reason in zip(pairs,probabilities,accepted,reasons):
        track_candidates.append({'left_segment':int(row['left_segment']),
                      'right_segment':int(row['right_segment']),
                      'score':round(float(p),6),'selected_provisionally':bool(selected),
                      'reason':reason,
                      'physical_join_certified':False})
    record={'schema':'s14_auto_track_candidates_v1',
            'source_sha256_grayscale_pixels':sha256(image.tobytes()).hexdigest(),
            'film_id':film_id,'run_id':run_id,'frame_id':frame_id,
            'fragment_count':len(original['segments']),
            'candidate_links':track_candidates,'calibration_receipt_usable':bool(valid),
            'real_auto_merged_fragments':0,
            'physical_3d_momentum_available':False,
            'mode':'PROVISIONAL_AUTOMATED_ANALYSIS_NO_PER_FRAME_HUMAN_LABELS',
            'limitation':'Cannot certify track identities or charge/momentum without independent real masks and optics/field calibration'}
    return original,record,overlay


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True)
    p.add_argument('--model-json',type=Path,required=True)
    p.add_argument('--calibration-json',type=Path)
    p.add_argument('--film-id',required=True);p.add_argument('--run-id',required=True);p.add_argument('--frame-id',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--method',choices=['contrast','frangi'],default='frangi')
    p.add_argument('--az-checkpoint',type=Path,help='Verified S14 original AZUNet best.pt (otherwise P0 baseline)')
    args=p.parse_args()
    image=cv2.imread(str(args.image),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError('failed to decode original image')
    model=json.loads(args.model_json.read_text(encoding='utf-8'))
    calibration=json.loads(args.calibration_json.read_text(encoding='utf-8')) if args.calibration_json else None
    p0,record,overlay=predict_image(image,model,film_id=args.film_id,run_id=args.run_id,frame_id=args.frame_id,
                      calibration=calibration,config=PilotConfig(segmentation_method=args.method),
                      original_az_checkpoint=args.az_checkpoint)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'p0_segments.json').write_text(json.dumps(p0,indent=2,ensure_ascii=False)+'\n')
    (args.out_dir/'graph_track_candidates.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    cv2.imwrite(str(args.out_dir/'segment_overlay.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    print(json.dumps({'fragments':record['fragment_count'],'links':len(record['candidate_links']),
                      'physical_certified':False,'run_id':args.run_id},ensure_ascii=False))
if __name__=='__main__':main()
