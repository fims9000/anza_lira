#!/usr/bin/env python3
"""Unsupervised gray-level style randomization for book-generated tracks.

Only image INTENSITIES are matched to real H50Z unlabeled crops. Geometry,
synthetic labels and track IDs stay artificial. This is NOT pseudo labelling
or proof that simulated textures match real bubbles. Shared H50Z style for
train/test means NOT an independently held-out camera transfer study.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np

Q=np.asarray([0,.005,.025,.1,.25,.5,.75,.90,.975,.995,1.0])


def grayscale_quantile_match(src:np.ndarray, target:np.ndarray) -> np.ndarray:
    if src.ndim!=2 or target.ndim!=2 or src.dtype!=np.uint8 or target.dtype!=np.uint8:
        raise ValueError('8-bit grayscale arrays required')
    s=np.quantile(src,Q);t=np.quantile(target,Q)
    s_unique,inds=np.unique(s,return_index=True)
    if len(s_unique)<2:raise ValueError('degenerate source quantiles')
    t_unique=t[inds]
    mapped=np.interp(src.astype(np.float64),s_unique,t_unique)
    return np.uint8(np.rint(np.clip(mapped,0,255)))


def run(source_dir:Path, real_crops:Path, out_dir:Path, *, seed:int=20261025) -> dict:
    source_dir=Path(source_dir).resolve();out_dir=Path(out_dir).resolve()
    if source_dir==out_dir:raise ValueError('must not overwrite original source images')
    refs=sorted(p for p in Path(real_crops).glob('*.png') if p.is_file())
    if not refs:raise ValueError('no real photographs for unlabeled style reference')
    ref_data=[]
    for p in refs:
        im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if im is None or im.dtype!=np.uint8:raise ValueError('invalid reference crop')
        ref_data.append((p,im,sha256(p.read_bytes()).hexdigest()))
    source_manifest=json.loads((source_dir/'s17_book_synthetic_sparse_manifest.json').read_text())
    if not source_manifest.get('dataset_origin','').startswith('SIMULATED_FROM_BOOK'):
        raise ValueError('input must be book-only generated dataset')
    test_info=json.loads((source_dir/'s17_book_test_truth_only.json').read_text())
    out_dir.mkdir(parents=True,exist_ok=True)
    entries=[];test=[];assign=[]
    all_items=[(row,'review') for row in source_manifest['entries']]+[(row,'test') for row in test_info['examples']]
    for j,(item,role) in enumerate(all_items):
        source=source_dir/item['image']
        orig=cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
        if orig is None:raise ValueError('missing book sample')
        choice=int.from_bytes(sha256(f'{seed}:{role}:{item["image"]}'.encode()).digest()[:8],'little')%len(ref_data)
        p,real,real_hash=ref_data[choice]
        styled=grayscale_quantile_match(orig,real)
        renamed=item['image'].replace('.png','_style.png')
        file=out_dir/renamed
        if not cv2.imwrite(str(file),styled):raise RuntimeError('PNG write failure')
        if role=='review':
            row={**item,'image':renamed,'image_sha256':sha256(file.read_bytes()).hexdigest(),
                 'source_kind':'BOOK_PHYSICS_SIMULATED + UNLABELED_REAL_PHOTO_HISTOGRAM_ONLY',
                 'synthetic_truth_unchanged':True}
            if item.get('mask'):
                old=source_dir/item['mask'];new=out_dir/item['mask'];new.write_bytes(old.read_bytes())
                assert row['mask_sha256']==sha256(new.read_bytes()).hexdigest()
            entries.append(row)
        else:
            testrow={**item,'image':renamed,'is_real_photo':False,'style_reference_source_sha256':real_hash}
            for maskfield in ('visible_gt','full_path_unobserved'):
                source_mask=source_dir/item[maskfield];(out_dir/item[maskfield]).write_bytes(source_mask.read_bytes())
            test.append(testrow)
        assign.append({'sample':item['image'],'ref_sha256':real_hash,'ref_filename':p.name,
                       'source_kind':'unlabeled real H50Z appearance only'})
    (out_dir/'s17_book_synthetic_sparse_manifest.json').write_text(json.dumps({
        'schema':'s14_sparse_image_reviews_v1','entries':entries,
        'dataset_origin':'SIMULATED_FROM_BOOK_EQUATIONS_WITH_UNLABELED_H50Z_HISTOGRAM_RANDOMIZATION',
        'human_verified_labels':0},ensure_ascii=False,indent=2)+'\n')
    (out_dir/'s17_book_test_truth_only.json').write_text(json.dumps({'status':'UNTOUCHED_SYNTHETIC_TEST_ONLY',
        'examples':test},ensure_ascii=False,indent=2)+'\n')
    report={'schema':'s17_unlabeled_photo_appearance_v1',
            'photo_reference_count':len(ref_data),'book_samples_restyled':len(assign),
            'new_real_track_annotations':0,'synthetic_labels_unchanged':True,
            'reference_sha256s':[x[2] for x in ref_data],
            'reference_assignments':assign,
            'not_a_real_domain_transfer_test':True,
            'limitations':['Only global grayscale quantiles; no optical point-spread model',
                           'Real crops may contain unknown tracks; they are NOT used as labeled background',
                           'Styles from same H50Z source influence train/val/test; no independent run holdout']}
    (out_dir/'s17_style_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--real-crops',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps({k:v for k,v in run(a.source_dir,a.real_crops,a.out_dir).items() if k!='reference_assignments'},indent=2))

if __name__=='__main__': main()
