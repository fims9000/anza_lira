#!/usr/bin/env python3
"""Mix genuine visualMetro expert track SHAPES with Belyaev-book curve shapes.

Every image pixel and 0/1/255 sparse mask remains SYNTHETIC. The measured
source provides positions, not corresponding photographs. No real tracks are
called human-reviewed pixels or all-other-pixels-as-negative.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import json
import shutil
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.large_relation_study import frame_partition
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig,render_measured_track


def sparse_for_measured_gray(image:np.ndarray,original_synthetic_truth:np.ndarray,rng:np.random.Generator):
    """Inject known synthetic scratches to get 0s without mislabelling unknown pixels."""
    size=image.shape[0]
    if image.shape!=(size,size) or original_synthetic_truth.shape!=image.shape:
        raise ValueError('expected aligned square synthetic inputs')
    truth=original_synthetic_truth>0
    altered=image.astype(np.float32).copy()
    artifact=np.zeros_like(original_synthetic_truth,dtype=np.uint8)
    for _ in range(4):
        p=tuple(int(x) for x in rng.integers(4,size-4,size=2))
        q=tuple(int(x) for x in np.clip(np.asarray(p)+rng.integers(-25,26,size=2),1,size-2))
        cv2.line(artifact,p,q,1,1)
    artifact[truth]=0
    altered[artifact>0]=np.maximum(0,altered[artifact>0]-rng.uniform(35,70))
    return_im=np.uint8(np.clip(altered,0,255))
    mask=np.full(image.shape,255,np.uint8)
    pts=np.argwhere(truth)
    neg=np.argwhere(artifact>0)
    if len(pts)<5 or len(neg)<5:raise ValueError('too few known artificial pixels')
    ps=pts[rng.choice(len(pts),min(32,len(pts)),replace=False)]
    ns=neg[rng.choice(len(neg),min(24,len(neg)),replace=False)]
    mask[ps[:,0],ps[:,1]]=1
    mask[ns[:,0],ns[:,1]]=0
    return return_im,mask


def _choose_measured(manifest,frame_ids,n,size,seed,split):
    grouped=[];source_frames=set(frame_ids)
    for f in manifest['frames']:
        if f['frame_id'] not in source_frames:continue
        for tr in f['tracks']:
            if tr['n_points']>=7:
                grouped.append((f['frame_id'],tr))
    if not grouped:raise ValueError('no eligible measured source shapes')
    rng=np.random.default_rng(int.from_bytes(sha256(f'{seed}:{split}:ordering'.encode()).digest()[:8],'little'))
    perm=rng.permutation(len(grouped))
    # Avoid reusing identical frame+track+view as another *independent* sample.
    if n>len(grouped):raise ValueError('request exceeds distinct measured tracks')
    for i in range(n):
        frame_id,track=grouped[int(perm[i])]
        view=int((i%3)+1)
        gray,truth,meta=render_measured_track(track,view,i,SyntheticFilmConfig(size=size,sample_seed=seed))
        art_seed=int.from_bytes(sha256(f'{seed}:{split}:{frame_id}:{track["track_id"]}:art'.encode()).digest()[:8],'little')
        image,mask=sparse_for_measured_gray(gray,truth,np.random.default_rng(art_seed))
        yield image,mask,truth,{'frame_id':frame_id,'track_id':track['track_id'],'view':view,
                                'real_geometry_only':True,'real_pixel_labels':False}


def build(book_dir:Path,measurements:Path,out_dir:Path,*,seed:int=20261025,
          measured_train:int=96,measured_val:int=24,measured_test:int=24,size:int=128):
    book_dir=Path(book_dir).resolve();out_dir=Path(out_dir).resolve()
    source=json.loads((book_dir/'s17_book_synthetic_sparse_manifest.json').read_text())
    if not source.get('dataset_origin','').startswith('SIMULATED_FROM_BOOK'):
        raise ValueError('only book synthetic input accepted')
    book_records=source['entries']
    book_photo=cv2.imread(str(book_dir/book_records[0]['image']),cv2.IMREAD_GRAYSCALE)
    if book_photo.shape!=(size,size):raise ValueError('measured and book dimensions must match')
    full=audited_large_manifest(Path(measurements))
    splits=frame_partition([f['frame_id'] for f in full['frames']],seed)
    out_dir.mkdir(exist_ok=True,parents=True)
    entries=[];test_rows=[];used={'train':set(),'val':set(),'test':set()}
    for r in book_records:
        image_from=book_dir/r['image'];image_to=out_dir/('book_'+image_from.name)
        shutil.copyfile(image_from,image_to)
        row={**r,'image':image_to.name,'image_sha256':sha256(image_to.read_bytes()).hexdigest(),
             'source_kind':'BELYAEV_BOOK_SYNTHETIC_NOT_REAL_FILM'}
        if r.get('mask'):
            src=book_dir/r['mask'];dest=out_dir/('book_'+src.name)
            shutil.copyfile(src,dest)
            row['mask']=dest.name;row['mask_sha256']=sha256(dest.read_bytes()).hexdigest()
        entries.append(row)
    for split,n in [('train',measured_train),('val',measured_val),('test',measured_test)]:
        for idx,(im,mask,dense,info) in enumerate(_choose_measured(full,splits[split],n,size,seed,split)):
            name=f'visualmetro_{split}_{idx:04d}'
            fp=out_dir/(name+'.png');cv2.imwrite(str(fp),im)
            used[split].add(info['frame_id'])
            row={'film_id':'MEASURED_SHAPES_FILM_070','run_id':'SYNTHETIC_RENDER_FROM_070',
                 'frame_id':info['frame_id'],'split':split,'image':fp.name,
                 'image_sha256':sha256(fp.read_bytes()).hexdigest(),
                 'source_kind':'VISUALMETRO_REAL_CLICKS_SYNTHETIC_PIXELS_NOT_MATCHED_BMP',
                 'synthetic_track_id':info['track_id'],'view':info['view']}
            if split in ('train','val'):
                mp=out_dir/(name+'_sparse_mask.png');cv2.imwrite(str(mp),mask)
                row['mask']=mp.name;row['mask_sha256']=sha256(mp.read_bytes()).hexdigest()
                entries.append(row)
            else:
                truth=out_dir/(name+'_synthetic_truth.png');cv2.imwrite(str(truth),dense*255)
                test_rows.append({'image':fp.name,'visible_gt':truth.name,'source_frame':info['frame_id'],
                                  'track_id':info['track_id'],'source_kind':'REAL_GEOMETRY_ONLY_SYNTHETIC_IMAGE'})
    for a in used:
        for b in used:
            if a!=b and used[a]&used[b]:raise RuntimeError('visualMetro frame leaked across splits')
    test_book=json.loads((book_dir/'s17_book_test_truth_only.json').read_text())
    for row in test_book['examples']:
        for field in ('image','visible_gt','full_path_unobserved'):
            src=book_dir/row[field];dst=out_dir/('book_'+src.name)
            shutil.copyfile(src,dst)
        test_rows.append({**row,'image':'book_'+row['image'],'visible_gt':'book_'+row['visible_gt'],
                          'full_path_unobserved':'book_'+row['full_path_unobserved']})
    (out_dir/'s17_book_mixed_sparse_manifest.json').write_text(json.dumps({
        'schema':'s14_sparse_image_reviews_v1','entries':entries,
        'dataset_origin':'MIXED_BELYAEV_BOOK_FORMULA_AND_VISUALMETRO_REAL_CLICKS_WITH_ALL_SIMULATED_IMAGE_PIXELS',
        'certified_real_pixels':0},ensure_ascii=False,indent=2)+'\n')
    (out_dir/'s17_mixed_heldout_synthetic_only.json').write_text(json.dumps({'schema':'s17_mixed_synthetic_truth_v1',
        'examples':test_rows,'is_real_photographic_gt':False},ensure_ascii=False,indent=2)+'\n')
    book_train=sum(x['split']=='train' and x['source_kind'].startswith('BELYAEV') for x in entries)
    rec={'schema':'s17_book_measured_mixture_v1',
         'measured_geometry_source_sha256':full['provenance_sha256'],
         'frame_split_counts':{k:len(v) for k,v in splits.items()},
         'used_distinct_frame_counts':{k:len(v) for k,v in used.items()},
         'book_train_count':book_train,'measured_train_count':measured_train,
         'book_train_fraction':round(book_train/(book_train+measured_train),4),
         'train_total':sum(x['split']=='train' for x in entries),
         'validation_total':sum(x['split']=='val' for x in entries),
         'unlabeled_total':sum(x['split']=='unlabeled' for x in entries),
         'synthetic_test_total':len(test_rows),
         'all_pixel_masks_from_simulation':True,'real_photographic_validation':False,
         'independent_film_holdout':False,'source_film_count_for_real_clicks':1,
         'no_human_expert_dense_track_masks':True,
         'conclusion':'Book-derived stopping curves are supplemental; most shapes originate from real expert points but images are simulated'}
    (out_dir/'s17_mixed_receipt.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
    return rec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--book-dir',type=Path,required=True)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--measured-train',type=int,default=96)
    p.add_argument('--measured-val',type=int,default=24)
    p.add_argument('--measured-test',type=int,default=24)
    a=p.parse_args()
    print(json.dumps(build(a.book_dir,a.measurements,a.out_dir,seed=a.seed,size=a.size,
         measured_train=a.measured_train,measured_val=a.measured_val,measured_test=a.measured_test),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
