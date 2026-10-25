#!/usr/bin/env python3
"""S31 original AZUNet versus U-Net on MACHINE-PROVISIONAL sparse marks.

No real photo pixel accuracy is reported. Test photo 006 is used only for
machine-draft agreement, never for fitting/threshold selection or model choice.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_weak_data(manifest:Path,tiles:Path):
    manifest=Path(manifest);obj=json.loads(manifest.read_text());root=manifest.parent
    if obj.get('schema')!='s31_weak_training_masks_source_locked_v1' or obj.get('human_certified_track_pixels')!=0:
        raise ValueError('Training data must be machine weak supervision; expert truth may NOT be replaced')
    if len(obj.get('entries',[]))!=39:raise ValueError('Incomplete source crops')
    groups={'005':[],'006':[]}
    for r in obj['entries']:
        cid=r['crop_id'];photo=r['photo_id']
        if photo not in groups or not cid.startswith(f'H50Z_030_01_{photo}_') or '/' in cid or '..' in cid:
            raise ValueError('Bad source image identity')
        im=Path(tiles)/(cid+'.png');m=root/r['mask'];w=root/r['weight']
        if not im.is_file() or _sha(im)!=r['image_sha256'] or _sha(m)!=r['mask_sha256'] or _sha(w)!=r['weight_sha256']:
            raise ValueError('Weak source photo/mask/weight SHA mismatch')
        x=cv2.imread(str(im),0);target=cv2.imread(str(m),cv2.IMREAD_UNCHANGED);weights=np.load(w,allow_pickle=False).astype(np.float32)
        if x.shape!=(512,512) or target.shape!=x.shape or weights.shape!=x.shape or target.dtype!=np.uint8:
            raise ValueError('Malformed source arrays')
        if set(np.unique(target))-{0,1,255} or np.any(weights[target==255]>0) or np.any(~np.isfinite(weights)):
            raise ValueError('Illegal train pixels')
        groups[photo].append((x,target,weights,r))
    return groups,obj


def make_patches(rows, *, size=64,per_positive_crop=10,seed=20261025):
    if size!=64:raise ValueError('S31 patch size fixed to 64 for controlled comparison')
    rng=np.random.default_rng(seed);patches=[]
    for x,mask,weight,rec in rows:
        points=np.argwhere(mask==1)
        if not len(points):continue
        for k in range(per_positive_crop):
            cy,cx=points[int(rng.integers(0,len(points)))];y=int(np.clip(cy-32,0,448));xx=int(np.clip(cx-32,0,448))
            a=x[y:y+64,xx:xx+64];b=mask[y:y+64,xx:xx+64];w=weight[y:y+64,xx:xx+64]
            if not (b==1).any():continue
            # Include small number of pseudo negative pixels, if available.
            # Do not hallucinate missing class on original photo.
            patches.append((a.copy(),b.copy(),w.copy(),rec['crop_id']))
    return patches


def split_bce(logits,target,weights):
    known=(target==0)|(target==1)
    pos=target==1;neg=target==0
    if not bool(known.any()):return None
    yp=torch.clamp(target.float(),0,1)
    # Label smoothing acknowledges uncertain machine pseudo-ground-truth.
    yp=torch.where(pos,torch.tensor(.92,device=logits.device),torch.tensor(.08,device=logits.device))
    bce=F.binary_cross_entropy_with_logits(logits,yp,reduction='none')
    terms=[]
    if bool(pos.any()):terms.append(bce[pos].mean())
    if bool(neg.any()):terms.append(bce[neg].mean())
    # Empirical agreement proxy, not native-real segmentation loss.
    return sum(terms)/len(terms)


def evaluate(model,patches):
    model.eval();counts={'positive_pseudo_total':0,'positive_pseudo_detected':0,'negative_assumption_total':0,'negative_assumption_detected':0,'pseudo_known_bce_sum':0,'sample_count':0}
    with torch.no_grad():
        for image,mask,weights,_ in patches:
            x=torch.from_numpy((255-image).astype('float32')[None,None]/255.)
            pred=model(x);prob=torch.sigmoid(pred)[0,0].numpy();mask0=mask==0;mask1=mask==1
            loss=split_bce(pred,torch.from_numpy(mask.astype('int64'))[None,None],torch.from_numpy(weights)[None,None])
            if loss is None:continue
            counts['pseudo_known_bce_sum']+=float(loss);counts['sample_count']+=1
            counts['positive_pseudo_total']+=int(mask1.sum());counts['positive_pseudo_detected']+=int(((prob>=.5)&mask1).sum())
            counts['negative_assumption_total']+=int(mask0.sum());counts['negative_assumption_detected']+=int(((prob>=.5)&mask0).sum())
    if counts['sample_count']==0:raise ValueError('No heldout pseudo sparse observations')
    return {'pseudo_label_proxy_BCE':counts['pseudo_known_bce_sum']/counts['sample_count'],
        'machine_positive_fraction_found':counts['positive_pseudo_detected']/max(counts['positive_pseudo_total'],1),
        'machine_assumed_negative_fraction_misidentified':counts['negative_assumption_detected']/max(counts['negative_assumption_total'],1),
        'evaluated_pseudo_pixels_positive':counts['positive_pseudo_total'],
        'evaluated_pseudo_pixels_negative_assumed':counts['negative_assumption_total'],
        'real_native_photo_track_Dice':'NOT_AVAILABLE','real_track_false_joins':'NOT_AVAILABLE',
        'test_source_photo':'006_same_physical_run_as_train_005'}


def train_models(manifest:Path,tiles:Path,out:Path,*,epochs=5,seed=20261025,train_per_crop=14,val_per_crop=8,models=('unet','azunet')):
    if not 1<=epochs<=30:raise ValueError('epochs out of bounds')
    verify_model_source();torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups,provenance=load_weak_data(manifest,tiles)
    training=make_patches(groups['005'],per_positive_crop=train_per_crop,seed=seed)
    heldout=make_patches(groups['006'],per_positive_crop=val_per_crop,seed=seed+998)
    if len(training)<10 or len(heldout)<3:raise ValueError('Insufficient weak pseudo annotations')
    result={'schema':'s31_original_anza_provisional_bootstrap_ablation_v1',
        'source_manifest_sha256':_sha(manifest),'seed':seed,'train_photo':'005','heldout_photo':'006',
        'both_photos_from_same_physical_run':True,'human_verified_labels':0,
        'ground_truth_from_machine_drafts':True,'real_photo_track_accuracy':'NOT_EVALUABLE',
        'models':{}}
    for method in models:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=build_model(method,widths=(4,6,8,12),rules=2)
        optim=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        hist=[];best=float('inf');t0=time.time()
        for epoch in range(epochs):
            model.train();epoch_losses=[]
            for idx in np.random.default_rng(seed+epoch).permutation(len(training)):
                image,mask,weight,_=training[int(idx)]
                rot=int(np.random.default_rng(seed+epoch*900+int(idx)).integers(0,4))
                im=np.rot90(image,rot).copy();mt=np.rot90(mask,rot).copy();wt=np.rot90(weight,rot).copy()
                x=torch.from_numpy((255-im).astype('float32')[None,None]/255.)
                tar=torch.from_numpy(mt.astype('int64'))[None,None]
                pred=model(x);loss=split_bce(pred,tar,torch.from_numpy(wt)[None,None])
                if loss is None or not torch.isfinite(loss):raise RuntimeError('Training loss missing or nonfinite')
                optim.zero_grad(set_to_none=True);loss.backward();optim.step();epoch_losses.append(float(loss.detach()))
            val=evaluate(model,heldout)
            hist.append({'epoch':epoch+1,'train_machine_BCE':float(np.mean(epoch_losses)),
                         'heldout_machine_BCE':val['pseudo_label_proxy_BCE']})
            if val['pseudo_label_proxy_BCE']<best:
                best=val['pseudo_label_proxy_BCE'];dest=out/method;dest.mkdir(exist_ok=True)
                torch.save({'model_state':model.state_dict(),'kind':method,'source_manifest_sha256':_sha(manifest),
                 'label_type':'MACHINE_DRAFT_NOT_HUMAN_EXPERT','validation_is_pseudo_label_agreement':True,
                 'photo_group':'005_train_006_val_same_run','epoch':epoch+1},dest/'best.pt')
        bestck=torch.load(out/method/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(bestck['model_state'])
        result['models'][method]={'history':hist,'best_epoch':bestck['epoch'],
           'pseudo_holdout':evaluate(model,heldout),'train_patches':len(training),
           'heldout_patches':len(heldout),'time_seconds':round(time.time()-t0,2),
           'trained_original_ANZA_architecture':method=='azunet'}
        (out/'S31_training_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('S31 TRAINED',method,'best',result['models'][method]['best_epoch'],result['models'][method]['pseudo_holdout'],flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--tiles',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--train-per-crop',type=int,default=14);p.add_argument('--val-per-crop',type=int,default=8)
    p.add_argument('--models',nargs='+',choices=['unet','azunet'],default=['unet','azunet'])
    a=p.parse_args();train_models(a.manifest,a.tiles,a.out,epochs=a.epochs,train_per_crop=a.train_per_crop,val_per_crop=a.val_per_crop,models=tuple(a.models))
