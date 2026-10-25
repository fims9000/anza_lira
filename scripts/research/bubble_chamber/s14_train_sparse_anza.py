#!/usr/bin/env python3
"""Train original AZUNet/UNet from FEW human-reviewed sparse image masks.

Manifest entries: image,mask,film_id,run_id,frame_id,split in
train/val/unlabeled. 0 = certified artifact, 1 = confirmed track,
255 = UNKNOWN; unreviewed image split has no mask. Unknown is NOT background.

Teacher pseudo-labels train only; they NEVER alter original reviewed masks,
validation, or real expert labels. Saves a reusable checkpoint but makes no
physical-camera deployment claim without held-out independent film validation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256,sha1
import argparse,json,copy
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import cv2

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s14_sparse_teacher import sparse_teacher_loss,SparseTeacherProtocol
from bubble_chamber.anza_segmentation_adapter import mask_loss

MODEL_SOURCE_BLOBS={'models/azconv.py':'f4217ada00024927506fa8ebe520aa1f68ae5854',
                    'models/segmentation.py':'0b971c117f3d56cc588ebade6991c65faceb8a2a'}


def verify_model_source(root=ROOT):
    for relative,expected in MODEL_SOURCE_BLOBS.items():
        p=Path(root)/relative
        if not p.is_file():raise FileNotFoundError('Original model missing: '+str(p))
        data=p.read_bytes()
        actual=sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=expected:raise ValueError('Original model source changed: '+relative)
    return MODEL_SOURCE_BLOBS


def read_reviewed_manifest(path:Path, *, image_root:Path|None=None):
    path=Path(path).resolve()
    root=(Path(image_root) if image_root is not None else path.parent).resolve()
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!='s14_sparse_image_reviews_v1':raise ValueError('unsupported sparse reviewed manifest schema')
    entries=data.get('entries')
    if not isinstance(entries,list) or not entries:raise ValueError('empty sparse reviewed manifest')
    seen=set();group_splits=defaultdict(set);groups=defaultdict(list)
    for item in entries:
        if not all(isinstance(item.get(x),str) and item[x].strip() for x in ('image','film_id','run_id','frame_id','split')):
            raise ValueError('source filename, film, run, frame and split required')
        split=item['split']
        if split not in {'train','val','unlabeled'}:raise ValueError('unsupported split')
        group=(item['run_id'],item['frame_id'])
        group_splits[group].add(split)
        path_image=(root/item['image']).resolve()
        if not path_image.is_relative_to(root) or not path_image.is_file() or path_image.suffix.lower() not in {'.png','.bmp','.tif','.tiff'}:
            raise ValueError('invalid source image reference')
        image=cv2.imread(str(path_image),cv2.IMREAD_GRAYSCALE)
        if image is None or image.dtype!=np.uint8:raise ValueError('source image must be grayscale uint8')
        expected_sha=item.get('image_sha256')
        if not expected_sha or expected_sha!=sha256(path_image.read_bytes()).hexdigest():
            raise ValueError('source image SHA mismatch')
        ident=(split,item['film_id'],item['run_id'],item['frame_id'],item['image'])
        if ident in seen:raise ValueError('duplicate image in manifest')
        seen.add(ident)
        if split=='unlabeled':
            if item.get('mask') is not None:raise ValueError('unlabeled entry cannot include reviewed mask')
            mask=np.full(image.shape,255,dtype=np.uint8)
        else:
            raw=item.get('mask')
            if not isinstance(raw,str) or not raw.strip():raise ValueError('reviewed image needs sparse mask')
            mp=(root/raw).resolve()
            if not mp.is_relative_to(root) or not mp.is_file() or mp.suffix.lower()!='.png':
                raise ValueError('invalid sparse mask path')
            if item.get('mask_sha256')!=sha256(mp.read_bytes()).hexdigest():
                raise ValueError('sparse mask SHA mismatch')
            mask=cv2.imread(str(mp),cv2.IMREAD_UNCHANGED)
            if mask is None or mask.dtype!=np.uint8 or mask.ndim!=2 or mask.shape!=image.shape or not set(np.unique(mask)).issubset({0,1,255}):
                raise ValueError('invalid/misaligned sparse review mask')
            if split=='train' and not np.any((mask==1)|(mask==0)):
                raise ValueError('reviewed train crop has no expert marks; put it in unlabeled split')
        groups[split].append((image,mask,item))
    if any(len(s)>1 for s in group_splits.values()):raise ValueError('frame leakage across splits')
    if not groups['train'] or not groups['val']:raise ValueError('train and independent-frame validation required')
    if not any((m==1).any() for _,m,_ in groups['train']) or not any((m==0).any() for _,m,_ in groups['train']):
        raise ValueError('training needs both expert foreground and certified artifacts')
    if not any((m==1).any() for _,m,_ in groups['val']) or not any((m==0).any() for _,m,_ in groups['val']):
        raise ValueError('validation needs both expert classes')
    return groups,{'schema':data['schema'],'source_manifest_sha256':sha256(path.read_bytes()).hexdigest(),
                   'image_source_roots_not_persisted':True,
                   'split_examples':{k:len(v) for k,v in groups.items()},
                   'split_independent_frames':{k:len({(r['run_id'],r['frame_id']) for _,_,r in group}) for k,group in groups.items()},
                   'source_run_count':len({r['run_id'] for group in groups.values() for _,_,r in group}),
                   'source_film_count':len({r['film_id'] for group in groups.values() for _,_,r in group}),
                   'real_dense_pixel_ground_truth':False}


def build_model(name='azunet',widths=(4,6,8,12),rules=2):
    verify_model_source()
    from models.segmentation import AZUNet,BaselineUNet
    from models.azconv import AZConvConfig
    if name=='azunet':
        return AZUNet(in_channels=1,out_channels=1,num_rules=rules,widths=tuple(widths),
                      cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True))
    if name=='unet':return BaselineUNet(in_channels=1,out_channels=1,widths=tuple(widths))
    raise ValueError('unknown architecture')


def _normalize(image:np.ndarray):
    return torch.from_numpy(((255-image).astype('float32')/255)[None,None])


def _known_bce(model,examples):
    scores=[]
    model.eval()
    with torch.no_grad():
        for image,mask,_ in examples:
            logits=model(_normalize(image))
            expert=torch.from_numpy(mask.astype('int64'))[None,None]
            known=(expert==0)|(expert==1)
            if not bool(known.any()):continue
            y=expert[known].float()
            vals=torch.nn.functional.binary_cross_entropy_with_logits(logits[known],y)
            scores.append(float(vals))
    return float(np.mean(scores)) if scores else float('inf')


def fit_sparse_reviews(groups:dict,manifest_summary:dict,*,out_dir:Path,architecture='azunet',
                      epochs:int=2,seed:int=20261024,teacher_cfg:SparseTeacherProtocol=SparseTeacherProtocol(),
                      widths=(4,6,8,12),rules=2,learning_rate=3e-4,
                      init_unlabeled_encoder:Path|None=None):
    if epochs<1 or learning_rate<=0:raise ValueError('epochs/rate invalid')
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    model=build_model(architecture,widths,rules)
    encoder_transfer=None
    if init_unlabeled_encoder is not None:
        from bubble_chamber.s20_transfer import transfer_unlabeled_encoder
        encoder_transfer=transfer_unlabeled_encoder(model,init_unlabeled_encoder,
            architecture=architecture,widths=tuple(widths),rules=rules)
    teacher=copy.deepcopy(model)
    for p in teacher.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    train=groups['train'];unlabeled=groups.get('unlabeled',[])
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf')
    for epoch in range(epochs):
        order=np.random.default_rng(seed+epoch).permutation(len(train))
        epoch_losses=[];pseudocount=0
        for step,index in enumerate(order):
            im,mask,_=train[int(index)]
            images=[im];human_masks=[mask]
            # Some genuine sparse-review crops may have ONLY foreground or
            # ONLY certified artifacts. Never invent the absent class from
            # unmarked pixels. Find a same-shape *reviewed training* companion
            # supplying the missing class so sparse_teacher_loss sees both.
            if not ((mask==1).any() and (mask==0).any()):
                missing=0 if not (mask==0).any() else 1
                companions=[(j,other,m) for j,(other,m,_) in enumerate(train)
                            if j!=int(index) and other.shape==im.shape and (m==missing).any()]
                if not companions:
                    raise ValueError('cannot form both-class batch from reviewed training crops of matching shapes')
                _,friend,fmask=companions[(epoch+step)%len(companions)]
                images.append(friend);human_masks.append(fmask)
            if unlabeled:
                other,_,_=unlabeled[(int(index)+epoch)%len(unlabeled)]
                if other.shape!=im.shape:raise ValueError('mixed shapes in batch: crop tiles to same size')
                images.append(other);human_masks.append(np.full(other.shape,255,dtype=np.uint8))
            x=torch.cat([_normalize(z) for z in images],dim=0)
            labels=torch.from_numpy(np.stack(human_masks).astype('int64'))[:,None]
            # No augmentation changes pixel coordinates. Both teacher maps
            # refer to the same input grid; photometric perturbation only.
            with torch.no_grad():
                teacher.eval()
                p1=teacher(x).sigmoid()
                gen=torch.Generator().manual_seed(seed*31+epoch*97+step)
                noise=torch.randn(x.shape,generator=gen,dtype=x.dtype)*.025
                p2=teacher((x*.92+noise).clamp(0,1)).sigmoid()
            model.train()
            logits=model(x)
            loss,counts=sparse_teacher_loss(logits,labels,p1,p2,teacher_cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            pseudocount+=counts['used_positive_count']+counts['used_negative_count']
            epoch_losses.append(float(loss.detach()))
            # Stable EMA teacher, not student probabilities used as labels.
            with torch.no_grad():
                for dest,src in zip(teacher.parameters(),model.parameters()):dest.lerp_(src,.02)
                for dest,src in zip(teacher.buffers(),model.buffers()):dest.copy_(src)
        validation_loss=_known_bce(model,groups['val'])
        info={'epoch':epoch+1,'mean_sparse_train_loss':float(np.mean(epoch_losses)),
              'known_validation_bce':validation_loss,
              'teacher_pseudo_pixels_used':pseudocount,
              'annotation_type':'expert 0/1/255 only, teacher never certified'}
        history.append(info)
        snap={'epoch':epoch+1,'architecture':architecture,'model_weights':model.state_dict(),
              'teacher_weights':teacher.state_dict(),'optimizer':optimizer.state_dict(),
              'history':history,'source_git_blobs':MODEL_SOURCE_BLOBS,
              'dataset_fingerprint':manifest_summary['source_manifest_sha256'],
              'architecture_widths':tuple(widths),'rules':rules,'source_data_private':True,
              'unlabeled_encoder_pretraining':encoder_transfer}
        torch.save(snap,out_dir/'last.pt')
        if validation_loss<best:
            best=validation_loss;torch.save(snap,out_dir/'best.pt')
    report={'schema':'s14_sparse_original_anza_training_v1','architecture':architecture,
            'source_models_unchanged':True,'data':manifest_summary,
            'train_episodes':epochs,'history':history,
            'unlabeled_encoder_pretraining':encoder_transfer,
            'validation_metric':'BCE on CERTIFIED pixels only, NOT full-image Dice',
            'real_unattended_accuracy_established':False,
            'limitations':['No real physical track-instance labels used for graph joining in this step',
                           'Without independently reviewed target films, no out-of-domain accuracy or deployment certification.',
                           'Pseudo target masks never saved as human truth; high-confidence threshold is not calibrated.']}
    (out_dir/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--image-root',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--model',choices=['unet','azunet'],default='azunet')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--widths',type=int,nargs=4,default=[4,6,8,12])
    p.add_argument('--rules',type=int,default=2)
    p.add_argument('--init-unlabeled-encoder',type=Path,help='Optional trusted S20 unlabeled photo encoder; expert 0/1/255 labels still mandatory')
    args=p.parse_args()
    groups,summary=read_reviewed_manifest(args.manifest,image_root=args.image_root)
    result=fit_sparse_reviews(groups,summary,out_dir=args.out_dir,architecture=args.model,
                              epochs=args.epochs,widths=args.widths,rules=args.rules,
                              init_unlabeled_encoder=args.init_unlabeled_encoder)
    print(json.dumps({'architecture':result['architecture'], 'data':result['data'],
                      'training':result['history'],'unattended_validated':False},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
