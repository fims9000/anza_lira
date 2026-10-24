#!/usr/bin/env python3
"""Generate a tiny *entirely synthetic* 0/1/255 review dataset for CI/demo.

NOT any actual bubble film or expert labels. Demonstrates file formats only.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json
import numpy as np
import cv2


def generate(out_dir:Path, *, seed=41, size=48):
    if size<32 or size%16:raise ValueError('tile size must be >=32 and divisible by 16')
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    entries=[];rng=np.random.default_rng(seed)
    for split,count in [('train',6),('val',3),('unlabeled',6)]:
        for index in range(count):
            image=np.full((size,size),rng.integers(180,230),dtype=np.float32)
            image+=rng.normal(0,8,(size,size))
            truth=np.zeros((size,size),np.uint8)
            line_x=int(rng.integers(size//5,size//2))
            points=np.asarray([[int(np.clip(line_x+.008*(y-size//2)**2,0,size-1)),y]
                              for y in range(3,size-3)],np.int32)
            cv2.polylines(truth,[points],False,1,2)
            artifact=np.zeros_like(truth)
            xx=int(rng.integers(size*3//5,size*4//5))
            cv2.line(artifact,(xx,4),(xx,size-4),1,1)
            image-=truth*90;image-=artifact*25
            image=np.clip(image,0,255).astype(np.uint8)
            fn=f'{split}_{index:02}.png'
            p=out_dir/fn;cv2.imwrite(str(p),image)
            row={'film_id':'SYNTHETIC_'+split,'run_id':split,'frame_id':f'{split}_{index:02}',
                 'split':split,'image':fn,'image_sha256':sha256(p.read_bytes()).hexdigest()}
            if split!='unlabeled':
                mask=np.full((size,size),255,np.uint8)
                pos=np.argwhere(truth>0);neg=np.argwhere(artifact>0)
                for yy,xx0 in pos[::max(1,len(pos)//16)][:16]:mask[yy,xx0]=1
                for yy,xx0 in neg[::max(1,len(neg)//12)][:12]:mask[yy,xx0]=0
                mp=out_dir/f'{split}_{index:02}_mask.png';cv2.imwrite(str(mp),mask)
                row.update(mask=mp.name,mask_sha256=sha256(mp.read_bytes()).hexdigest())
            entries.append(row)
    path=out_dir/'s14_sparse_demo_manifest.json'
    path.write_text(json.dumps({'schema':'s14_sparse_image_reviews_v1','entries':entries},ensure_ascii=False,indent=2)+'\n')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();print(generate(args.out_dir))
