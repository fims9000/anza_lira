"""Deterministic patch sampling around REAL expert 0/1 annotations.

No class interpolation, pseudo labels or classical ridge masks are used.
Masks are cropped with nearest-neighbor pixel identity, preserving UNKNOWN=255.
Source photo grouping remains exact across train/validation splits.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np


def make_sparse_patches(manifest_path:Path,out_dir:Path,*,size:int=128,
                        maximum_per_image:int=4)->tuple[Path,dict]:
    if size<32 or size%16 or not 1<=maximum_per_image<=16:
        raise ValueError('patch size must divide 16, and max patches should be between 1 and 16')
    path=Path(manifest_path).resolve()
    source_root=path.parent
    raw=json.loads(path.read_text(encoding='utf-8'))
    if raw.get('schema')!='s14_sparse_image_reviews_v1' or not isinstance(raw.get('entries'),list):
        raise ValueError('expected source-locked S14 reviewed manifest')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    entries=[];coverage=[];used_photo_groups={}
    for idx,entry in enumerate(raw['entries']):
        split=entry['split']
        if split not in {'train','val','unlabeled'}:raise ValueError('unrecognized split')
        group=(entry['run_id'],entry['frame_id'])
        if group in used_photo_groups and used_photo_groups[group]!=split:
            raise ValueError('source frame present in multiple split stages')
        used_photo_groups[group]=split
        src_image=(source_root/entry['image']).resolve()
        if not src_image.is_relative_to(source_root) or not src_image.is_file():
            raise ValueError('invalid source image path')
        if sha256(src_image.read_bytes()).hexdigest()!=entry['image_sha256']:
            raise ValueError('source image checksum drift')
        im=cv2.imread(str(src_image),cv2.IMREAD_GRAYSCALE)
        if im is None or im.dtype!=np.uint8:raise ValueError('expected source gray PNG')
        h,w=im.shape
        if min(h,w)<size:raise ValueError('selected patch bigger than original image')
        if split=='unlabeled':
            # No unreviewed crop is silently treated as training data with labels.
            coverage.append({'source':entry['image'],'split':split,'patch_count':0,'reviewed_pixels':0})
            continue
        src_mask=(source_root/entry['mask']).resolve()
        if not src_mask.is_relative_to(source_root) or not src_mask.is_file():
            raise ValueError('invalid reviewed mask path')
        if sha256(src_mask.read_bytes()).hexdigest()!=entry.get('mask_sha256'):
            raise ValueError('expert mask checksum drift')
        mask=cv2.imread(str(src_mask),cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape!=im.shape or mask.dtype!=np.uint8 or not set(np.unique(mask)).issubset({0,1,255}):
            raise ValueError('invalid reviewed mask alignment or values')
        # Deterministically sample evenly spaced annotated positions from both
        # classes, at most half of budget per class when both are available.
        groups=[]
        n_pos=int((mask==1).sum());n_neg=int((mask==0).sum())
        for cls in (1,0):
            pix=np.argwhere(mask==cls)
            if len(pix)==0:continue
            take=max(1,maximum_per_image//2) if n_pos and n_neg else maximum_per_image
            ind=np.linspace(0,len(pix)-1,min(take,len(pix)),dtype=np.int64)
            groups.extend([(int(pix[k,0]),int(pix[k,1]),cls) for k in ind])
        patches=[];seen=set()
        for y,x,cls in groups:
            top=max(0,min(h-size,y-size//2));left=max(0,min(w-size,x-size//2))
            if (top,left) in seen:continue
            seen.add((top,left));patches.append((top,left,cls))
            if len(patches)>=maximum_per_image:break
        for j,(top,left,cls) in enumerate(patches):
            im_patch=np.ascontiguousarray(im[top:top+size,left:left+size])
            mask_patch=np.ascontiguousarray(mask[top:top+size,left:left+size])
            if not ((mask_patch==0)|(mask_patch==1)).any():
                raise AssertionError('review-centered patch unexpectedly lost all expert labels')
            stem=f's19_{idx:03d}_{j:02d}'
            i_name=stem+'_original_photo_patch.png'
            m_name=stem+'_expert_sparse_0_1_255.png'
            cv2.imwrite(str(out/i_name),im_patch)
            cv2.imwrite(str(out/m_name),mask_patch)
            entries.append({'image':i_name,'mask':m_name,'split':split,
                            'frame_id':entry['frame_id'],'run_id':entry['run_id'],'film_id':entry['film_id'],
                            'image_sha256':sha256((out/i_name).read_bytes()).hexdigest(),
                            'mask_sha256':sha256((out/m_name).read_bytes()).hexdigest(),
                            's19_patch_origin':{'original_crop':entry['image'],
                                'window_crop_xyxy':[left,top,left+size,top+size],
                                'sampling_class':int(cls),
                                'sampling_only_from_certified_pixels':True}})
        coverage.append({'source':entry['image'],'split':split,'patch_count':len(patches),
                         'reviewed_pixels':n_pos+n_neg})
    payload={'schema':'s14_sparse_image_reviews_v1','entries':entries}
    final=out/'s19_reviewed_patch_manifest.json'
    final.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'schema':'s19_sparse_real_photo_crop_manifest_v1','patch_size':size,
             'source_manifest_sha256':sha256(path.read_bytes()).hexdigest(),
             'original_source_groups_by_split':{k:len({(e['run_id'],e['frame_id']) for e in raw['entries'] if e['split']==k}) for k in ('train','val','unlabeled')},
             'patch_examples_by_split':{k:sum(e['split']==k for e in entries) for k in ('train','val')},
             'coverage':coverage,'auto_human_labels_created':0,
             'unknown_pixels_preserved':True,'independent_physical_run_test':False}
    (out/'s19_patch_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return final,summary
