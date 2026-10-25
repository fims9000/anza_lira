#!/usr/bin/env python3
"""S27 inference ablation: full-ROI contrast/ridge features vs per-64px features.

Uses EXACT same trained original U-Net / AZUNet / guided-AZUNet weights and
frozen synthetic-photo005 calibration thresholds. No real track ground truth.
"""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import cv2,torch,numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s27_overlap_inference import overlap_predict_global_features,tile_boundary_jump
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint
from scripts.research.bubble_chamber.s27_overlap_photo_audit import model_thresholds,sha256


def run(args):
    torch.set_num_threads(2)
    m=json.loads(args.source.read_text())
    if m.get('schema')!='s27_h50z_no_label_overlap_robustness_v1':
        raise ValueError('need completed audited real source S27')
    provenance=json.loads(args.roi_manifest.read_text())
    if provenance.get('schema')!='s26b_full_frame_provenance_bundle_v1':
        raise ValueError('expected trusted H50Z source manifest')
    expected={r['crop_id']:r for r in provenance['review_crops']}
    items=m['per_crop'];by_id={r['source']['crop_id']:r for r in items}
    if set(by_id)!=set(expected):raise ValueError('missing/duplicate real ROIs')
    base=json.loads(args.baseline_report.read_text());guided=json.loads(args.guided_report.read_text());th=model_thresholds(base,guided)
    models={k:model_for_checkpoint(v) for k,v in {'unet':args.unet,'azunet':args.azunet,'guided_az':args.guided_az}.items()}
    out=args.out;out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    results=[]
    for record in items:
        id=record['source']['crop_id'];row=expected[id];path=args.crop_dir/(id+'.png')
        if sha256(path)!=row['source_crop_sha256'] or sha256(path)!=record['source']['source_crop_sha256']:
            raise ValueError('source crop digest drift')
        target=cache/(id+'.json')
        if target.exists() and args.use_cache:
            result=json.loads(target.read_text());results.append(result);continue
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        result={'source_crop_id':id,'photo_id':row['source_photo'],'review_stratum':row['stratum'],
                'source_sha256':sha256(path),'native_expert_truth_available':False,'models':{}}
        for name,(model,preprocess) in models.items():
            pred=overlap_predict_global_features(im,model,preprocess,stride=32)
            met=tile_boundary_jump(pred)
            prev=record['original_score_to_overlap'][name]
            result['models'][name]={
                'original_stride64_seam_jump':prev['no_overlap']['mean_boundary_jump'],
                'overlap_per_patch_preprocessing_seam_jump':prev['with_overlap']['mean_boundary_jump'],
                'overlap_full_roi_preprocessing_seam_jump':met['mean_boundary_jump'],
                'full_roi_vs_original_reduction_pct':100*(1-met['mean_boundary_jump']/max(prev['no_overlap']['mean_boundary_jump'],1e-8)),
                'full_roi_vs_per_patch_ratio':met['mean_boundary_jump']/max(prev['with_overlap']['mean_boundary_jump'],1e-8),
                'full_roi_candidate_pixel_fraction_UNVERIFIED':float((pred>=th[name]).mean())}
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        results.append(result)
        print('GLOBAL',id,'jump ratio global/per-patch',[(k,round(v['full_roi_vs_per_patch_ratio'],3)) for k,v in result['models'].items()],flush=True)
    stats={'schema':'s27_full_roi_feature_stage_ablation_v1','source_crops':len(results),
           'original_architecture_sources_unchanged':True,'source_photo_ids':sorted(set(x['photo_id'] for x in results)),
           'pixel_track_ground_truth':0,'physics_track_merges_authorized':0,'statistics':{},'per_crop':results}
    for photo in ('005','006'):
        rows=[r for r in results if r['photo_id']==photo]
        stats['statistics'][photo]={'crops':len(rows)}
        for name in models:
            s=np.array([[r['models'][name][k] for k in ('original_stride64_seam_jump','overlap_per_patch_preprocessing_seam_jump','overlap_full_roi_preprocessing_seam_jump')] for r in rows])
            stats['statistics'][photo][name]={'legacy_mean':float(s[:,0].mean()),'overlap_patch_feature_mean':float(s[:,1].mean()),
               'overlap_global_feature_mean':float(s[:,2].mean()),'global_improved_over_patch_crops':int(np.sum(s[:,2]<s[:,1])),
               'global_vs_legacy_reduction_pct':float(100*(1-s[:,2].sum()/s[:,0].sum()))}
    (out/'S27_global_input_feature_comparison.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats['statistics'],indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','roi_manifest','crop_dir','baseline_report','guided_report','unet','azunet','guided_az','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--use-cache',action='store_true')
    a=p.parse_args();run(a)

if __name__=='__main__':main()
