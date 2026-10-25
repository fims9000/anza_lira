#!/usr/bin/env python3
"""S19: REAL expert review -> sparse original AZUNet/U-Net -> provisional Graph-LIRA.

This does not synthesize expert labels, estimate physics from START anchors, or
allow autonomous physical track joins. Only train if two source photos have
both confirmed foreground/artifact classes collectively.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s19_real_review import load_package,compile_review
from scripts.research.bubble_chamber.s14_train_sparse_anza import (
    read_reviewed_manifest,fit_sparse_reviews,verify_model_source,
)
from scripts.research.bubble_chamber.s19_fit_reviewed_graph import fit_from_real_pair_file
from bubble_chamber.s14_sparse_teacher import SparseTeacherProtocol
from bubble_chamber.s19_sparse_patches import make_sparse_patches


def run_pipeline(*,proposals:Path, images:Path, reviews:Path, out_dir:Path, epochs:int=4,
                 seed:int=20261025,widths:tuple[int,int,int,int]=(4,6,8,12),rules:int=2,
                 patch_size:int=128, init_azunet:Path|None=None, init_unet:Path|None=None):
    if epochs<1 or any(w<2 for w in widths) or rules<1 or (patch_size and (patch_size<32 or patch_size%16)):
        raise ValueError('invalid training budget')
    package=load_package(proposals,images)
    reviewed=json.loads(Path(reviews).read_text(encoding='utf-8'))
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    masks_dir=out/'reviewed_source'
    compilation=compile_review(package,reviewed,out_dir=masks_dir,image_root=images)
    model_source_blobs=verify_model_source()
    ready=bool(compilation['trainable_real_image_baseline'])
    receipt={'schema':'s19_real_small_label_pipeline_v1',
             'status':'REAL_REVIEW_TRAINING_COMPLETED_PROVISIONAL_ONLY' if ready else 'NEEDS_MORE_CONFIRMED_EXPERT_MARKS',
             'source_package_sha256':package['package_sha256'],
             'expert_review_file_sha256':sha256(Path(reviews).read_bytes()).hexdigest(),
             'reviewed':compilation,'exact_original_model_source_blobs':model_source_blobs,
             'real_dense_ground_truth_available':False,'same_physical_run_only':compilation['real_photo_runs']==1,
             'independent_test_physical_run_available':False,'real_automatic_track_merges_allowed':False,
             'validation_metric':'BCE at manually confirmed pixels only, NOT photograph-wide Dice',
             'pseudo_teacher_used_for_training':False,'trained_models':{},'graph':{}}
    graph=fit_from_real_pair_file(masks_dir/'s19_pair_labels.json',out/'graph')
    receipt['graph']=graph
    if ready:
        train_manifest=masks_dir/'s14_sparse_image_reviews.json'
        if patch_size:
            train_manifest,patch_receipt=make_sparse_patches(train_manifest,out/'reviewed_patches',size=patch_size)
            receipt['patch_sampling']=patch_receipt
        groups,summary=read_reviewed_manifest(train_manifest)
        # Until real-photo domain-shift calibration, avoid teacher self-labeling.
        disabled_teacher=SparseTeacherProtocol(consistency_weight=0.0,
            min_reliable_pseudo_per_class=1000000000)
        for architecture in ('azunet','unet'):
            train=fit_sparse_reviews(groups,summary,out_dir=out/architecture,
                                     architecture=architecture,epochs=epochs,seed=seed,
                                     teacher_cfg=disabled_teacher,widths=widths,rules=rules,
                                     init_unlabeled_encoder=init_azunet if architecture=='azunet' else init_unet)
            receipt['trained_models'][architecture]={
                'best_val_known_pixel_bce':min(float(x['known_validation_bce']) for x in train['history']),
                'epochs':len(train['history']), 'checkpoint':'best.pt',
                'source':'original source-locked architecture',
                'unlabeled_pretraining':train['unlabeled_encoder_pretraining'],
                'real_full_image_pixel_dice':'NOT_MEASURED',
            }
    (out/'s19_pipeline_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--expert-review-json',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--widths',type=int,nargs=4,default=[4,6,8,12])
    p.add_argument('--rules',type=int,default=2)
    p.add_argument('--patch-size',type=int,default=128,help='Review-centered patches; 0 uses full photo tiles')
    p.add_argument('--init-azunet',type=Path,help='Optional S20 checkpoint for original AZUNet encoder only')
    p.add_argument('--init-unet',type=Path,help='Optional S20 checkpoint for ordinary U-Net encoder only')
    a=p.parse_args()
    r=run_pipeline(proposals=a.proposals,images=a.images,reviews=a.expert_review_json,
                   out_dir=a.out_dir,epochs=a.epochs,seed=a.seed,widths=tuple(a.widths),rules=a.rules,
                   patch_size=a.patch_size,init_azunet=a.init_azunet,init_unet=a.init_unet)
    print(json.dumps({'status':r['status'],'trained_models':r['trained_models'],
                      'graph_status':r['graph']['status'],'physical_track_merges':0},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
