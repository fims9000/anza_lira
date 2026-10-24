#!/usr/bin/env python3
"""S17: physics-conditioned *artificial* training set from Belyaev's hadroida.

Produces 0/1/255 sparse masks for existing S14 original-AZUNet trainer and
completely held-out simulated truths for debugging. Never creates expert labels
from real images, never asserts camera calibration or real particle identity.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bubble_chamber.s17_belyaev_hadroide import simulate_stopping_track


def _random_trajectory(size:int, rng:np.random.Generator, medium:str):
    mass=float(rng.uniform(120,940))  # parameter exploration, NOT particle ID labels
    alpha=float(rng.uniform(-.6,.6))
    hz=-float(rng.uniform(13,20))
    charge=int(rng.choice([-1,1]))
    fraction=float(rng.uniform(.08,.72))
    sim=simulate_stopping_track(mass_mev=mass,alpha_rad=alpha,
          medium=medium,mean_hz_kgauss=hz,charge_sign=charge,
          max_s_over_a=fraction,points=600)
    xy=sim.xy_cm
    span=max(np.ptp(xy[:,0]),np.ptp(xy[:,1]))
    if span <=0:raise ValueError('degenerate simulated curve')
    angle=float(rng.uniform(-math.pi,math.pi))
    rot=np.asarray([[math.cos(angle),-math.sin(angle)],
                    [math.sin(angle),math.cos(angle)]])
    pix=(xy-xy.mean(axis=0))@rot.T*(size*.73/span)
    shift=rng.uniform(-size*.05,size*.05,size=2)
    pix+=size/2+shift
    pts=np.rint(pix).astype(np.int32)
    if not np.all((pts[:,0]>=0)&(pts[:,0]<size)&(pts[:,1]>=0)&(pts[:,1]<size)):
        # Keep same physical shape but reduce slightly to fit; never clip unknown pixels.
        center=pts.mean(axis=0)
        pts=np.rint((pts-center)*.80+size/2).astype(np.int32)
    return pts,{'medium':medium,'mean_hz_kgauss':round(hz,4),'mass_sample_mev':round(mass,3),
                'angle_alpha_rad':round(alpha,5),'charge_sign':charge,
                's_over_a_max':round(fraction,5),'a_cm':round(sim.a_cm,5),
                'method':'Belyaev hadroida eq9 limited near actual stopping point',
                'pixel_projection':'SYNTHETIC_SIMILARITY_TRANSFORM_NOT_STEREO_OPTICAL_CALIBRATION'}


def make_example(size:int,rng:np.random.Generator,medium:str,*,cut:bool=True):
    if size<64 or size%16:raise ValueError('size must be divisible by 16 and >=64')
    pts,physics=_random_trajectory(size,rng,medium)
    truth=np.zeros((size,size),np.uint8)
    hidden_path=np.zeros_like(truth)
    cv2.polylines(hidden_path,[pts.reshape(-1,1,2)],False,1,2,cv2.LINE_8)
    if cut:
        # Remove a few image-observed segments, but RETAIN underlying path only
        # in the synthetic metadata. These are NOT true occlusions of real film.
        excluded=[]
        for _ in range(int(rng.integers(1,4))):
            center=int(rng.integers(80,len(pts)-80));width=int(rng.integers(6,23))
            excluded.append((center-width,center+width))
        visible=np.ones(len(pts)-1,dtype=bool)
        for lo,hi in excluded:visible[max(0,lo):min(len(visible),hi)]=False
        for idx in np.flatnonzero(visible):
            a=pts[idx];b=pts[idx+1]
            cv2.line(truth,tuple(a),tuple(b),1,int(rng.integers(1,3)),cv2.LINE_8)
    else:
        cv2.polylines(truth,[pts.reshape(-1,1,2)],False,1,2,cv2.LINE_8)
        excluded=[]
    # False optical scratches are intentionally NOT particle tracks.
    artifacts=np.zeros_like(truth)
    count=5 if medium=='liquid_propane' else 2
    for _ in range(count):
        a=tuple(int(x) for x in rng.integers(4,size-4,size=2))
        b=tuple(int(x) for x in np.clip(np.asarray(a)+rng.integers(-35,36,size=2),2,size-3))
        cv2.line(artifacts,a,b,1,1,cv2.LINE_8)
    artifacts[truth>0]=0
    yy,xx=np.mgrid[:size,:size]
    sigma=11 if medium=='liquid_propane' else 6
    img=(rng.uniform(165,227) +rng.normal(0,sigma,(size,size)) +
         (xx/size-.5)*rng.uniform(-22,22)+(yy/size-.5)*rng.uniform(-22,22))
    img-=truth*rng.uniform(50,98)
    img-=artifacts*rng.uniform(13,50)
    speckle=rng.random((size,size))<(0.004 if medium=='liquid_propane' else 0.001)
    img[speckle]-=rng.uniform(10,70)
    img=cv2.GaussianBlur(np.uint8(np.clip(img,0,255)),(3,3),rng.uniform(.5,1.1))
    # Sparse artificial scribbles: 1 positive, 0 confirmed artificial scratch,
    # 255 unknown. No real image pixels are automatically certified.
    sparse=np.full((size,size),255,dtype=np.uint8)
    pos=np.argwhere(truth>0)
    neg=np.argwhere(artifacts>0)
    if len(neg)==0 or len(pos)==0:raise ValueError('no foreground or synthetic artifact for sparse supervision')
    positive=pos[rng.choice(len(pos),min(25,len(pos)),replace=False)]
    negative=neg[rng.choice(len(neg),min(20,len(neg)),replace=False)]
    sparse[positive[:,0],positive[:,1]]=1
    sparse[negative[:,0],negative[:,1]]=0
    return img,truth,sparse,hidden_path,{**physics,'artificial_removed_index_ranges':excluded,
                                          'synthetic_visible_positive_count':int((truth>0).sum()),
                                          'synthetic_gt_only':True,'never_expert_certified':True}


def generate(out_dir:Path,*,seed:int=20261025,size:int=128,train:int=24,val:int=8,
             unlabeled:int=8,test:int=8) -> dict:
    if min(train,val,unlabeled,test)<1:raise ValueError('nonempty splits required')
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    entries=[];tests=[];counts={}
    for split,n in (('train',train),('val',val),('unlabeled',unlabeled),('test',test)):
        for i in range(n):
            # Deterministic independent seeds per image and split.
            digest=sha256(f'{seed}:{split}:{i}:book_v1'.encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            medium=['liquid_hydrogen','liquid_propane'][i%2]
            im,dense,sparse,full,info=make_example(size,rng,medium)
            stem=f'{split}_{i:04d}'
            photo=out_dir/f'{stem}.png'
            if not cv2.imwrite(str(photo),im):raise RuntimeError('failed to encode synthetic sample')
            common={'film_id':f'BOOK_SYNTHETIC_{split}', 'run_id':f'BOOK_SYNTHETIC_{split}',
                    'frame_id':stem,'split':split,'image':photo.name,
                    'image_sha256':sha256(photo.read_bytes()).hexdigest(),
                    'source_kind':'GENERATED_FROM_BELYAEV_P1_2023_22_EQUATION_9_NOT_REAL_BMP'}
            # read_reviewed_manifest only supports train/val/unlabeled; test is segregated.
            if split in ('train','val'):
                mp=out_dir/f'{stem}_sparse_mask.png'
                if not cv2.imwrite(str(mp),sparse):raise RuntimeError('failed to encode sparse mask')
                common.update(mask=mp.name,mask_sha256=sha256(mp.read_bytes()).hexdigest())
                entries.append(common)
            elif split=='unlabeled':
                entries.append(common)
            else:
                d=out_dir/f'{stem}_synthetic_truth.png'
                f=out_dir/f'{stem}_invisible_full_path.png'
                cv2.imwrite(str(d),dense*255)
                cv2.imwrite(str(f),full*255)
                tests.append({'image':photo.name,'visible_gt':d.name,'full_path_unobserved':f.name,
                              'medium':medium,'model_params':info})
            counts[split]=counts.get(split,0)+1
    manifest=out_dir/'s17_book_synthetic_sparse_manifest.json'
    manifest.write_text(json.dumps({'schema':'s14_sparse_image_reviews_v1','entries':entries,
                  'dataset_origin':'SIMULATED_FROM_BOOK_EQUATIONS_NO_HUMAN_EXPERT_LABELS',
                  'physical_applicability':'STOPPING_CHARGED_PARTICLE_LOCAL_SEGMENT_ONLY'},
                 indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out_dir/'s17_book_test_truth_only.json').write_text(json.dumps({'status':'UNTOUCHED_SYNTHETIC_TEST_ONLY',
          'examples':tests},indent=2,ensure_ascii=False)+'\n')
    rec={'source':'Belyaev P1-2023-22 eq.9 + table 1; visualMetro guide three-view/fiducial conventions',
         'schema':'s17_book_prior_synthetic_data_v1','seed':seed,'counts':counts,
         'real_photo_input_count':0,'expert_labeled_pixels':0,
         'synthetic_sparse_trainable_masks':train+val,
         'synthetic_full_test_masks':test,
         'physics_mass_or_identity_validated_on_real_data':False,
         'experiment_status':'MODEL_PRETRAINING_SIMULATED_DATA_ONLY',
         'must_not': ['interpret book curve as real photo annotation','infer a stopping point from image tip',
                      'infer 3D calibration/B map from a 2D shape', 'call test masks real expert labels']}
    (out_dir/'s17_receipt.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
    return rec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--train',type=int,default=24)
    p.add_argument('--val',type=int,default=8)
    p.add_argument('--unlabeled',type=int,default=8)
    p.add_argument('--test',type=int,default=8)
    args=p.parse_args()
    print(json.dumps(generate(args.out_dir,seed=args.seed,size=args.size,train=args.train,
                      val=args.val,unlabeled=args.unlabeled,test=args.test),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
