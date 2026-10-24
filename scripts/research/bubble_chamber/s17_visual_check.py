#!/usr/bin/env python3
"""Generate honest contact sheets: synthetic book geometry vs unlabeled real H50Z."""
from pathlib import Path
import argparse
import cv2
import numpy as np


def label_image(x,label):
    img=cv2.cvtColor(x,cv2.COLOR_GRAY2BGR) if x.ndim==2 else x.copy()
    img=cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)
    bar=np.zeros((40,img.shape[1],3),np.uint8)+35
    cv2.putText(bar,label,(7,25),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
    return np.concatenate([bar,img],axis=0)


def montage(book:Path,styled:Path,real:Path,out:Path):
    panels=[]
    for idx in range(4):
        stem=f'train_{idx:04d}'
        orig=cv2.imread(str(book/f'{stem}.png'),cv2.IMREAD_GRAYSCALE)
        syn=cv2.imread(str(styled/f'{stem}_style.png'),cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(str(book/f'{stem}_sparse_mask.png'),cv2.IMREAD_GRAYSCALE)
        if orig is None or syn is None or mask is None:raise ValueError('missing train image')
        disp=np.zeros((mask.shape[0],mask.shape[1],3),np.uint8)+70
        disp[mask==1]=(55,215,60)
        disp[mask==0]=(40,65,230)
        panels.append(cv2.hconcat([label_image(orig,'Book / synthetic'),label_image(syn,'Real-photo style only'),
                label_image(disp,'Green/Red only; gray UNKNOWN')]))
    fig=cv2.vconcat(panels)
    out.parent.mkdir(exist_ok=True,parents=True)
    cv2.imwrite(str(out),fig)
    # Separate genuine-photo sample so scientific user cannot confuse sources.
    photos=sorted(real.glob('*_512.png'))[:4]
    rows=[]
    for p in photos:
        im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        thumb=cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)
        rows.append(label_image(thumb,'REAL H50Z (UNLABELED)'))
    if rows:cv2.imwrite(str(out.with_name('s17_real_H50Z_UNLABELED.png')),cv2.hconcat(rows))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--book',type=Path,required=True)
    p.add_argument('--styled',type=Path,required=True)
    p.add_argument('--real',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(montage(a.book,a.styled,a.real,a.out))
