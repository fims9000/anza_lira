#!/usr/bin/env python3
"""S27 photo006 synthetic LONG-track crossing seam stress, no native pixel truth.

The exact same artificially deposited line/scratch pixels are scored by the
original 64px patch inference vs overlap-averaged inference. 005-trained model
weights and PHOTO005-frozen synthetic-mark thresholds are never changed.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,cv2,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint,tiled_predict
from scripts.research.bubble_chamber.s27_overlap_photo_audit import model_thresholds,prepare_models,sha256
from bubble_chamber.s27_overlap_inference import overlap_predict,tile_boundary_jump


def injected_long_track(bg:np.ndarray,seed:int)->tuple[np.ndarray,np.ndarray,np.ndarray,list[list[int]]]:
    if bg.dtype!=np.uint8 or bg.shape!=(512,512):raise ValueError('real source image tile must be original 512x512')
    rng=np.random.default_rng(seed)
    xs=np.arange(18,498,dtype=np.int32)
    base=int(rng.integers(160,350));slope=float(rng.uniform(-.2,.2));curve=float(rng.uniform(-.0009,.0009))
    ys=np.rint(base+slope*(xs-260)+curve*(xs-260)**2).astype(np.int32)
    ys=np.clip(ys,14,498)
    coordinates=np.stack([xs,ys],axis=1).reshape(-1,1,2)
    fg=np.zeros_like(bg,dtype='uint8');cv2.polylines(fg,[coordinates],False,1,int(rng.integers(1,3)),cv2.LINE_8)
    artifact=np.zeros_like(bg,dtype='uint8')
    for i in range(15):
        x0=int(rng.integers(12,480));y0=int(rng.integers(12,480))
        x1=int(np.clip(x0+rng.integers(-20,21),4,508));y1=int(np.clip(y0+rng.integers(-40,41),4,508))
        cv2.line(artifact,(x0,y0),(x1,y1),1,int(rng.integers(1,3)),cv2.LINE_8)
    artifact[fg>0]=0
    raw=bg.astype('float32')
    raw-=float(rng.uniform(43,85))*fg
    raw-=float(rng.uniform(35,75))*artifact
    raw=np.clip(raw,0,255).astype('uint8')
    if (fg>0).sum()<200 or (artifact>0).sum()<100:
        raise ValueError('too few synthetic positive/distractor pixels')
    return raw,fg.astype(bool),artifact.astype(bool),coordinates[:,0,:].tolist()


def seam_band(shape:tuple[int,int],half_width:int=4,tile:int=64)->np.ndarray:
    yy,xx=np.indices(shape)
    return (np.minimum(xx%tile, tile-(xx%tile))<=half_width)|(
            np.minimum(yy%tile, tile-(yy%tile))<=half_width)


def longest_run(array:np.ndarray)->int:
    arr=np.asarray(array,bool).ravel()
    if not arr.any():return 0
    edges=np.diff(np.r_[False,arr,False].astype('int8'))
    return int((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).max())


def measure(scores:np.ndarray,truth:np.ndarray,distractor:np.ndarray,threshold:float,points)->dict:
    y=np.r_[np.ones(truth.sum(),int),np.zeros(distractor.sum(),int)]
    p=np.r_[scores[truth],scores[distractor]]
    accepted=scores>=threshold
    seam=seam_band(scores.shape)
    coords=np.asarray(points)
    positive_along=accepted[coords[:,1],coords[:,0]]
    answer={'injected_positive_pixels':int(truth.sum()),'injected_negative_scratch_pixels':int(distractor.sum()),
            'synthetic_roc_auc':float(roc_auc_score(y,p)),
            'overall_synthetic_tpr':float(accepted[truth].mean()),
            'overall_synthetic_scratch_fpr':float(accepted[distractor].mean()),
            'longest_accepted_synthetic_centerline_fraction':float(longest_run(positive_along)/len(positive_along)),
            'physical_native_track_truth':'UNKNOWN'}
    for mode,area in [('near_old_tile_seams',seam),('away_from_old_tile_seams',~seam)]:
        f=truth&area;n=distractor&area
        answer[mode]={'positives':int(f.sum()),'scratches':int(n.sum()),
           'synthetic_tpr':float(accepted[f].mean()) if f.any() else None,
           'synthetic_scratch_fpr':float(accepted[n].mean()) if n.any() else None}
    return answer


def run(args):
    torch.set_num_threads(2)
    base=json.loads(args.baseline_report.read_text())
    guided=json.loads(args.guided_report.read_text())
    thresholds=model_thresholds(base,guided)
    checkpoint={'unet':args.unet,'azunet':args.azunet,'guided_az':args.guided_az}
    models=prepare_models(checkpoint)
    reviewed=json.loads(args.roi_manifest.read_text())['review_crops']
    target=sorted([r for r in reviewed if r['source_photo']=='006'],key=lambda r:(r.get('stratum',''),r['crop_id']))
    if len(target)<8:raise ValueError('not enough 006 private real photographs')
    # Fixed source-only and stratified selection, no candidate truth consulted.
    order=[];by_stratum={}
    for r in target:by_stratum.setdefault(r['stratum'],[]).append(r)
    while len(order)<min(args.n_per_seed,len(target)):
        for key in sorted(by_stratum):
            if by_stratum[key]:order.append(by_stratum[key].pop(0))
            if len(order)>=args.n_per_seed:break
    if len(order)!=args.n_per_seed:raise ValueError('insufficient selected images')
    out=[];cache=args.out/'per_trial';cache.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        for i,row in enumerate(order):
            photo=args.roi_dir/(row['crop_id']+'.png')
            if sha256(photo)!=row['source_crop_sha256']:raise ValueError('real source SHA-256 mismatch')
            image=cv2.imread(str(photo),0)
            augmented,fg,scratch,trace=injected_long_track(image,seed*1000+i)
            entry={'seed':seed,'crop_id':row['crop_id'],'photo_id':'006','real_photo_source_sha256':sha256(photo),
                   'human_expert_native_pixel_masks':0,'synthetic_overlay_labels_only':True,'methods':{}}
            for name,(model,preproc) in models.items():
                one=tiled_predict(augmented,model,preproc)
                mixed=overlap_predict(augmented,model,preproc,stride=32,batch_size=32)
                t=thresholds[name]
                entry['methods'][name]={'stride64':measure(one,fg,scratch,t,trace),
                                        'stride32':measure(mixed,fg,scratch,t,trace),
                                        'stride64_boundary_jump':tile_boundary_jump(one)['mean_boundary_jump'],
                                        'stride32_boundary_jump':tile_boundary_jump(mixed)['mean_boundary_jump']}
            (cache/f'{row["crop_id"]}_seed{seed}.json').write_text(json.dumps(entry,indent=2)+'\n')
            out.append(entry)
            print('SIMULATED 006',seed,row['crop_id'],flush=True)
    result={'schema':'s27_seam_crossing_synthetic_superimposed_on_genuine_H50Z_v1',
            'seeds':args.seeds,'independent_physical_runs':0,'source_photo_id':'006',
            'train_photo_id':'005','006_never_used_for_supervised_checkpoint_or_threshold':True,
            'test_photo_already_seen_for_previous_exploratory_method_design':True,
            'source_roi_count':len(order),'trials':out,
            'frozen_photo005_synthetic_mark_thresholds':thresholds,
            'meaning':'True labels ONLY for newly injected software marks, not native physical tracks',
            'safe_for_real_particle_join_decisions':False}
    (args.out/'S27_synthetic_crossing_006.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ('roi_dir','roi_manifest','baseline_report','guided_report','unet','azunet','guided_az','out'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261029,20261030]);p.add_argument('--n-per-seed',type=int,default=8)
    run(p.parse_args())

if __name__=='__main__':main()
