"""S108 candidate model: multi-scale image patches, explicitly sparse labels only.

NO automatic training on unlabeled H50Z. No pseudo-GT from S88/S105. Output is
uncalibrated track-likeness ranking, not a calibrated probability of a particle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .s107_review import EXPORT_SCHEMA, validate_annotation, verify_pack, evaluate


class ConvBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(inp,out,3,padding=1,bias=False),
            nn.GroupNorm(4,out),nn.SiLU(),
            nn.Conv2d(out,out,3,padding=1,bias=False),
            nn.GroupNorm(4,out),nn.SiLU(),
        )

    def forward(self,x):
        return self.layer(x)


class S108UNet(nn.Module):
    """Two-channel local+wide appearance model; no pretrained weights."""
    def __init__(self, base=16):
        super().__init__()
        if base not in (8,12,16,24):
            raise ValueError('Unsupported model capacity')
        self.base=base
        self.enc1=ConvBlock(2,base)
        self.enc2=ConvBlock(base,2*base)
        self.enc3=ConvBlock(2*base,4*base)
        self.bottleneck=ConvBlock(4*base,6*base)
        self.dec3=ConvBlock(10*base,4*base)
        self.dec2=ConvBlock(6*base,2*base)
        self.dec1=ConvBlock(3*base,base)
        self.head=nn.Conv2d(base,1,1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=2 or min(x.shape[-2:])<32:
            raise ValueError('S108 expects N,2,H,W with spatial dims >=32')
        a=self.enc1(x)
        b=self.enc2(F.avg_pool2d(a,2))
        c=self.enc3(F.avg_pool2d(b,2))
        d=self.bottleneck(F.avg_pool2d(c,2))
        d=self.dec3(torch.cat((F.interpolate(d,size=c.shape[2:],mode='bilinear',align_corners=False),c),dim=1))
        d=self.dec2(torch.cat((F.interpolate(d,size=b.shape[2:],mode='bilinear',align_corners=False),b),dim=1))
        d=self.dec1(torch.cat((F.interpolate(d,size=a.shape[2:],mode='bilinear',align_corners=False),a),dim=1))
        return self.head(d)


def image_channels(gray:np.ndarray)->np.ndarray:
    if gray.ndim!=2 or gray.dtype!=np.uint8:
        raise ValueError('Requires original uint8 grayscale image')
    x=gray.astype(np.float32)
    # Preserve dark ridges before local-contrast enhancement: original photo is
    # still a separate raw input channel. No S105 or annotation leakage.
    wide=cv2.GaussianBlur(x,(0,0),sigmaX=7.0)
    return np.stack(((255-x)/255.0, np.clip((wide-x)/45.0,-1,1)),axis=0)


def sparse_masked_bce(logits:torch.Tensor,label:torch.Tensor)->torch.Tensor:
    """Only 1=clear track and 2=clear reviewed artifact get supervision."""
    if logits.ndim!=4 or logits.shape[1]!=1 or logits[:,0].shape!=label.shape:
        raise ValueError('Logits/labels mismatch')
    if label.dtype not in (torch.long,torch.int64,torch.uint8):
        raise ValueError('Label dtype should be integral')
    select=(label==1)|(label==2)
    if not bool(select.any()):
        raise ValueError('No explicitly annotated, certain source-pixel labels')
    expected=(label[select]==1).float()
    values=logits[:,0][select]
    # Weighted classes to avoid long artifact strokes drowning short traces.
    pos=int(expected.sum().item());negative=len(expected)-pos
    if pos==0 or negative==0:
        raise ValueError('Both explicit track and artifact labels required')
    w=torch.where(expected>0,torch.tensor(min(6.,max(.4,negative/pos)),device=values.device),
                  torch.tensor(1.,device=values.device))
    return (F.binary_cross_entropy_with_logits(values,expected,reduction='none')*w).sum()/w.sum()


@dataclass
class SparseTile:
    image: np.ndarray
    label: np.ndarray
    tile_id: str


def load_sparse_training_inputs(pack:Path,review_json:Path,mask_dir:Path) -> tuple[list[SparseTile],dict]:
    manifest=verify_pack(pack)
    review=json.loads(Path(review_json).read_text(encoding='utf-8'))
    summary=validate_annotation(review,manifest)
    if not summary['reviewer_confirmed']:
        raise ValueError('Human annotation has not been confirmed by an expert')
    labelled=[]
    for tile in manifest['tiles']:
        gray=cv2.imread(str(Path(pack)/tile['original_png']),cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(str(Path(mask_dir)/f"{tile['tile_id']}_partial_stroke_targets.png"),cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape!=mask.shape:
            raise ValueError('Missing source image or S107 explicit stroke targets')
        if np.any(~np.isin(mask,[0,1,2,3])):
            raise ValueError('Illegal sparse label encoding')
        if np.any(mask==1) or np.any(mask==2):
            labelled.append(SparseTile(gray,mask,tile['tile_id']))
    return labelled,summary


def make_patch(tile:SparseTile, y:int,x:int,patch:int=192):
    if patch<32 or patch%16:
        raise ValueError('Patch size must be >=32 and multiple of 16')
    h,w=tile.image.shape
    y0=int(np.clip(y-patch//2,0,max(0,h-patch)))
    x0=int(np.clip(x-patch//2,0,max(0,w-patch)))
    src=tile.image[y0:y0+patch,x0:x0+patch]
    lab=tile.label[y0:y0+patch,x0:x0+patch]
    if src.shape!=(patch,patch):
        dh=patch-src.shape[0];dw=patch-src.shape[1]
        src=cv2.copyMakeBorder(src,0,dh,0,dw,cv2.BORDER_REFLECT_101)
        lab=cv2.copyMakeBorder(lab,0,dh,0,dw,cv2.BORDER_CONSTANT,value=0)
    return image_channels(src),lab.copy()


def sampled_batches(tiles:list[SparseTile],rng:np.random.Generator,*,batch_size:int,patch:int):
    tracks=[];artifacts=[]
    for i,tile in enumerate(tiles):
        for cls,collection in ((1,tracks),(2,artifacts)):
            ys,xs=np.where(tile.label==cls)
            for y,x in zip(ys[::max(1,len(ys)//5000)],xs[::max(1,len(xs)//5000)]):
                collection.append((i,int(y),int(x)))
    if not tracks or not artifacts:
        raise ValueError('Need confirmed track AND artifact strokes; missing class remains UNKNOWN')
    while True:
        features=[];targets=[]
        for j in range(batch_size):
            group=tracks if (j%2==0) else artifacts
            i,y,x=group[int(rng.integers(len(group)))]
            img,mask=make_patch(tiles[i],y,x,patch)
            if bool(rng.integers(2)):
                img=np.flip(img,axis=2).copy();mask=np.fliplr(mask).copy()
            if bool(rng.integers(2)):
                img=np.flip(img,axis=1).copy();mask=np.flipud(mask).copy()
            features.append(img)
            targets.append(mask)
        yield torch.from_numpy(np.stack(features)),torch.from_numpy(np.stack(targets).astype(np.int64))


def research_pilot_train(pack:Path,review_json:Path,out:Path,*,steps:int=50,
                         patch:int=192,batch_size:int=4,seed:int=20261110,
                         one_run_pilot_only:bool=False,base:int=16):
    """Requires real expert labels. Cannot claim independent test with H50Z pair."""
    if not one_run_pilot_only:
        raise ValueError('H50Z 005/006 are one run; explicit --one-run-pilot-only needed')
    if not 1<=steps<=20000 or not 2<=batch_size<=32:
        raise ValueError('Unsafe training run budget')
    torch.manual_seed(seed)
    np.random.seed(seed)
    out=Path(out)
    if out.exists():raise FileExistsError('Refusing to overwrite trained-model output')
    pack,review_json=Path(pack),Path(review_json)
    # The S107 evaluator guards all source SHA hashes and exports only manually
    # reviewed partial scribbles; no S105 source candidate becomes a target.
    eval_tmp=out.parent/(out.name+'_partial_targets')
    if eval_tmp.exists():raise FileExistsError(eval_tmp)
    review=json.loads(review_json.read_text(encoding='utf-8'))
    manifest=verify_pack(pack)
    validate_annotation(review,manifest)
    if not review.get('reviewer_confirmed'):
        raise ValueError('Reviewer must explicitly confirm annotations first')
    r=evaluate(review,pack,eval_tmp)
    sparse,summary=load_sparse_training_inputs(pack,review_json,eval_tmp/'sparse_labels')
    if not sparse:raise ValueError('No annotated image regions')
    npos=sum(int((tile.label==1).sum()) for tile in sparse)
    nneg=sum(int((tile.label==2).sum()) for tile in sparse)
    if npos<50 or nneg<50:
        raise ValueError('Need >=50 clear labeled pixels of each class for pilot')
    rng=np.random.default_rng(seed)
    batches=sampled_batches(sparse,rng,batch_size=batch_size,patch=patch)
    model=S108UNet(base=base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    logs=[]
    for i in range(steps):
        x,y=next(batches)
        optimizer.zero_grad(set_to_none=True)
        score=model(x)
        loss=sparse_masked_bce(score,y)
        if not torch.isfinite(loss):raise ValueError('NaN loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
        logs.append(round(float(loss.item()),5))
    out.mkdir(parents=True,exist_ok=False)
    checkpoint={'state_dict':model.state_dict(),'base':base,'in_channels':2,
                'schema':'S108_UNCALIBRATED_SPARSE_LINE_RANKER_PILOT_V1'}
    torch.save(checkpoint,out/'weights.pt')
    h=hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest()
    metadata={'schema':checkpoint['schema'],
              'source_manifest_sha256':manifest['manifest_sha256'],
              'review_export_sha256':hashlib.sha256(review_json.read_bytes()).hexdigest(),
              'weights_sha256':h,'training_steps':steps,'model_base_width':base,
              'training_patch_px':patch,'labelled_track_pixels':npos,
              'labelled_artifact_pixels':nneg,'labelled_tile_count':len(sparse),
              'training_loss_only':logs,'independent_test_accuracy':None,
              'independent_physical_runs_tested':0,
              'same_run_only_pilot':True,
              'uncalibrated_track_likeness_not_true_probability':True,
              'unlabelled_pixels_never_used_as_background':True,
              'not_particle_identity_segmentation':True}
    (out/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def predict_score_preview(model:S108UNet,gray:np.ndarray,*,patch:int=256,overlap:int=64)->np.ndarray:
    """Tile-safe image-score preview; remains uncalibrated and non-physical."""
    if gray.ndim!=2 or gray.dtype!=np.uint8 or patch%16 or overlap<0 or overlap>=patch//2:
        raise ValueError('Invalid native image/tiling')
    h,w=gray.shape
    if min(h,w)<patch:raise ValueError('ROI smaller than input patch')
    def starts(n):
        seq=list(range(0,n-patch+1,patch-overlap))
        if seq[-1]!=n-patch:seq.append(n-patch)
        return seq
    weights=np.zeros(gray.shape,np.float32)
    accum=np.zeros(gray.shape,np.float32)
    ramp=(.2+.8*np.outer(np.hanning(patch),np.hanning(patch))).astype(np.float32)
    model.eval()
    with torch.no_grad():
        for y in starts(h):
            for x in starts(w):
                channels=image_channels(gray[y:y+patch,x:x+patch])
                z=model(torch.from_numpy(channels[None]))[0,0].numpy()
                accum[y:y+patch,x:x+patch]+=z*ramp
                weights[y:y+patch,x:x+patch]+=ramp
    return accum/np.maximum(weights,1e-6)