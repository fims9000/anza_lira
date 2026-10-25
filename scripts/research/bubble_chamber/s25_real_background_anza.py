#!/usr/bin/env python3
"""S25: original AZUNet vs original U-Net on injected traces over genuine H50Z backgrounds.

Ground truth describes ONLY newly injected mathematical stroke pixels. Real
bubble traces are NEVER called positives or negatives by this experiment.
Existing real pixels are outside the pixel-loss/evaluation contract.
"""
from __future__ import annotations
import argparse,json,hashlib,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig,render_measured_track


def examples(measurements:Path, tiles:Path,seed:int=20261025,per_photo:int=24,size:int=64):
    manifest=audited_large_manifest(measurements)
    grouped=[(f['frame_id'],tr) for f in manifest['frames'] for tr in f['tracks'] if tr['n_points']>=7]
    if len(grouped)<per_photo*2:raise ValueError('insufficient measured track shapes')
    files=sorted(tiles.glob('H50Z_030_01_*.png'))
    if len(files)<4:raise ValueError('need real H50Z image crop tiles')
    photos={}
    for path in files:
        photo=path.name.split('_start')[0]
        photos.setdefault(photo,[]).append(path)
    if sorted(photos)!=['H50Z_030_01_005','H50Z_030_01_006']:raise ValueError('expect exactly two independent original source photographs')
    out={};shas={}
    for photo,inputs in sorted(photos.items()):
        rows=[];h=hashlib.sha256()
        for j in range(per_photo):
            path=inputs[j%len(inputs)]; tile=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if tile is None or tile.shape!=(512,512):raise ValueError('expected real 512px crop')
            h.update(path.read_bytes())
            rng=np.random.default_rng(seed+j*919+int(photo[-3:]))
            x0=int(rng.integers(0,512-size));y0=int(rng.integers(0,512-size))
            real=tile[y0:y0+size,x0:x0+size].copy()
            # Unique expert measured track for each synthetic overlay. Track clicks
            # and authentic source pixels are never physically co-registered.
            _,track=grouped[(j*131+int(photo[-3:]))%len(grouped)]
            _,mask,_=render_measured_track(track,(j%3)+1,j,SyntheticFilmConfig(size=size,sample_seed=seed))
            mask=np.asarray(mask,dtype=np.uint8)
            if mask.sum()<12:continue
            contrast=float(rng.uniform(26,85))
            result=np.clip(real.astype('float32')-contrast*mask,0,255).astype('uint8')
            # Only known synthetic stroke (1) and synthetic artifacts (0) can
            # be supervised; original background, including real tracks, UNKNOWN.
            annotated=np.full((size,size),255,dtype=np.uint8)
            pos=np.argwhere(mask>0)
            selected=pos[rng.choice(len(pos),min(42,len(pos)),replace=False)]
            annotated[selected[:,0],selected[:,1]]=1
            artifact=np.zeros((size,size),np.uint8)
            for _ in range(5):
                p=tuple(int(x) for x in rng.integers(3,size-3,2))
                q=tuple(int(np.clip(z+delta,1,size-2)) for z,delta in zip(p,rng.integers(-13,14,2)))
                cv2.line(artifact,p,q,1,int(rng.integers(1,3)))
            artifact[mask>0]=0
            neg=np.argwhere(artifact>0)
            if len(neg)<12:continue
            selected_neg=neg[rng.choice(len(neg),min(42,len(neg)),replace=False)]
            result[artifact>0]=np.clip(result[artifact>0].astype('float32')-rng.uniform(22,65),0,255).astype('uint8')
            annotated[selected_neg[:,0],selected_neg[:,1]]=0
            assert set(np.unique(annotated))<= {0,1,255}
            rows.append((result,annotated,mask,real,{'photo':photo,'tile':path.name,'source_track':track['track_id'],'source_frame':track['frame_id']}))
        out[photo]=rows;shas[photo]=h.hexdigest()
    return out,{'schema':'s25_real_background_simulated_trace_v1',
       'real_photo_filenames':sorted(photos),'source_measurements_sha256':manifest['provenance_sha256'],
       'real_photo_tile_sha256_aggregate':shas,'real_human_track_pixels':0,
       'synthetic_positive_and_negative_pixels':True,'no_registered_expert_line_ground_truth':True,
       'independent_physical_run_test':False,
       'train_photo':'H50Z_030_01_005','validation_photo':'H50Z_030_01_006',
       'warning':'Synthetic stroke superimposed on real H50Z pixels; no native real track was identified or labeled'}


def train(name,train_data,val_data,seed,epochs,outdir,init=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
    model=build_model(name,widths=(4,6,8,12),rules=2)
    opt=torch.optim.AdamW(model.parameters(),lr=.0006)
    history=[];best=1e6;started=time.monotonic()
    for epoch in range(epochs):
        model.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(train_data)):
            image,labels,*_=train_data[j]
            x=torch.from_numpy((255-image).astype('float32')[None,None]/255)
            y=torch.from_numpy(labels.astype('int64'))[None,None]
            known=y!=255
            pred=model(x)
            loss=F.binary_cross_entropy_with_logits(pred[known],y[known].float())
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval();vals=[]
        with torch.no_grad():
            for image,labels,*_ in val_data:
                x=torch.from_numpy((255-image).astype('float32')[None,None]/255)
                y=torch.from_numpy(labels.astype('int64'))[None,None];known=y!=255
                vals.append(float(F.binary_cross_entropy_with_logits(model(x)[known],y[known].float())))
        metric=float(np.mean(vals));history.append({'epoch':epoch+1,'sparse_train_bce':float(np.mean(losses)),'heldout_photo_synthetic_marks_bce':metric})
        if metric<best:
            best=metric
            outdir.mkdir(parents=True,exist_ok=True)
            torch.save({'architecture':name,'state':model.state_dict(),'seed':seed,'epoch':epoch+1,
                 'experiment':'S25 injected synthetic tracks on real background, NOT authentic track masks',
                 'trained_real_expert_labels':0},outdir/'best.pt')
    ck=torch.load(outdir/'best.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state']);model.eval()
    evaluation={'sparse_bce':best,'synthetic_injected_tp':0,'synthetic_injected_fp_on_known_artifact_marks':0,'synthetic_injected_fn':0,'pred_count':0}
    mosaics=[]
    with torch.no_grad():
        for image,labels,mask,base,meta in val_data:
            x=torch.from_numpy((255-image).astype('float32')[None,None]/255)
            p=torch.sigmoid(model(x))[0,0].numpy()
            pred=p>=.5;known_pos=labels==1;known_neg=labels==0
            evaluation['synthetic_injected_tp']+=int((pred&known_pos).sum())
            evaluation['synthetic_injected_fn']+=int((~pred&known_pos).sum())
            evaluation['synthetic_injected_fp_on_known_artifact_marks']+=int((pred&known_neg).sum())
            evaluation['pred_count']+=int(pred.sum())
            if len(mosaics)<6:
                overlay=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR);overlay[pred]=[65,205,70]
                mosaics.append(np.hstack([cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),overlay]))
    if mosaics:cv2.imwrite(str(outdir/'six_real_background_injected_prediction_examples.png'),np.vstack(mosaics))
    return {'architecture':name,'seed':seed,'training_samples':len(train_data),'heldout_synthetic_samples':len(val_data),
       'epochs':epochs,'best_epoch':ck['epoch'],'history':history,
       'heldout_photo_injected_synthetic_label_metrics':evaluation,'elapsed_seconds':round(time.monotonic()-started,2),
       'real_bmp_track_segmentation_dice':'NOT_MEASURED','real_particle_identity_accuracy':'NOT_MEASURED'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--real-tiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--per-photo',type=int,default=24)
    p.add_argument('--seed',type=int,default=20261025);a=p.parse_args()
    groups,receipt=examples(a.measurements,a.real_tiles,seed=a.seed,per_photo=a.per_photo)
    a.output.mkdir(parents=True,exist_ok=True)
    result={'provenance':receipt,'models':{}}
    for modelname in ['unet','azunet']:
        result['models'][modelname]=train(modelname,groups['H50Z_030_01_005'],groups['H50Z_030_01_006'],
                                     seed=a.seed,epochs=a.epochs,outdir=a.output/modelname)
        (a.output/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('FINISHED',modelname,result['models'][modelname]['heldout_photo_injected_synthetic_label_metrics'],flush=True)
if __name__=='__main__':main()
