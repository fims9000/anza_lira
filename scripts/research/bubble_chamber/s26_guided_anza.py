#!/usr/bin/env python3
"""Exploratory S26: source-unchanged ANZA vs UNet, two real-image feature channels.

Channels are local-contrast and directionally matched ridge response. The
H50Z training supervision remains SYNTHETIC overlaid line/scratch pixels.
Validation is 005-only. Test 006 is never used for selecting an epoch/threshold.
"""
from __future__ import annotations
import argparse,json,sys,time,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_two_photo_baselines import sample_two_photos,pixel_input,classical_score,reviewed_scores,score_measures,select_threshold
from scripts.research.bubble_chamber.s14_train_sparse_anza import verify_model_source


def two_channel(im):
    a=pixel_input(im,'local')
    s=classical_score(im,'steerable_ridge')
    b=np.clip(s/.065,0,1).astype('float32')
    return np.stack([a,b])


def build(name):
    verify_model_source()
    from models.segmentation import AZUNet,BaselineUNet
    from models.azconv import AZConvConfig
    if name=='azunet':return AZUNet(in_channels=2,out_channels=1,num_rules=2,widths=(4,6,8,12),
                              cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True))
    if name=='unet':return BaselineUNet(in_channels=2,out_channels=1,widths=(4,6,8,12))
    raise ValueError('invalid architecture')


def fit(data,name,seed,epochs,out):
    torch.set_num_threads(2);torch.manual_seed(seed)
    model=build(name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4)
    xx=torch.tensor(np.stack([two_channel(x.gray) for x in data['train']]),dtype=torch.float32)
    yy=torch.tensor(np.stack([x.sparse for x in data['train']])[:,None],dtype=torch.long)
    vx=torch.tensor(np.stack([two_channel(x.gray) for x in data['val']]),dtype=torch.float32)
    vy=torch.tensor(np.stack([x.sparse for x in data['val']])[:,None],dtype=torch.long)
    best=1e8;history=[];start=time.monotonic()
    out.mkdir(parents=True,exist_ok=True)
    for ep in range(epochs):
        model.train();losses=[]
        rng=np.random.default_rng(seed+ep)
        indices=rng.permutation(len(xx))
        for ids in np.array_split(indices,math.ceil(len(indices)/8)):
            x=xx[ids].clone();y=yy[ids]
            gain=torch.tensor(rng.uniform(.8,1.2,size=(len(ids),1,1,1)),dtype=x.dtype)
            shift=torch.tensor(rng.uniform(-.1,.1,size=(len(ids),1,1,1)),dtype=x.dtype)
            x=(x*gain+shift).clamp(0,1)
            z=model(x)
            p=z[y==1];n=z[y==0]
            loss=.5*(F.binary_cross_entropy_with_logits(p,torch.ones_like(p))+F.binary_cross_entropy_with_logits(n,torch.zeros_like(n)))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval();vl=[]
        with torch.no_grad():
            for ix in np.array_split(np.arange(len(vx)),math.ceil(len(vx)/8)):
                z=model(vx[ix]);y=vy[ix];p=z[y==1];n=z[y==0]
                vl.append(float(.5*(F.binary_cross_entropy_with_logits(p,torch.ones_like(p))+F.binary_cross_entropy_with_logits(n,torch.zeros_like(n)))))
        metric=float(np.mean(vl));history.append({'epoch':ep+1,'training_bce':np.mean(losses),'photo005_val_bce':metric})
        if metric<best:
            best=metric;torch.save({'architecture':name,'input_channels':2,'state':model.state_dict(), 'seed':seed,'epoch':ep+1,
                                  'train_photo':'005','val_photo':'005','test_photo':'006',
                                  'true_native_photo_track_labels':0},out/'best.pt')
    ck=torch.load(out/'best.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state']);model.eval()
    def predict(img):
        with torch.no_grad():return model(torch.from_numpy(two_channel(img))[None]).sigmoid()[0,0].cpu().numpy()
    val,truth,_,_=reviewed_scores(data['val'],predict);cal=select_threshold(val,truth)
    test,y,group,_=reviewed_scores(data['test'],predict)
    result={'architecture':name,'input_channels':['local_darkness','steerable_ridge'],
            'source_unchanged_AZConv':True,'training':{'best_epoch':ck['epoch'],'history':history,'seconds':time.monotonic()-start},
            'validation_005':cal,'heldout_006':score_measures(y,test,cal['threshold'],group),
            'actual_real_photo_track_pixel_truth':'ABSENT','physically_verified_tracks':0}
    (out/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bmps',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--methods',nargs='+',default=['azunet','unet'])
    args=p.parse_args();data,meta=sample_two_photos(args.bmps,args.measurements,args.seed,train_n=144,val_n=48,test_n=96)
    for name in args.methods:
        result=fit(data,name,args.seed,args.epochs,args.out/name)
        print('GUIDED',args.seed,name,'AUC',result['heldout_006']['roc_auc_synthetic_marks'],'TPR',result['heldout_006']['tpr_at_frozen_threshold'],flush=True)
