#!/usr/bin/env python3
"""Compare simple real human 'start' vs fiducial baselines, NOT track segmentation.

Train photo 005 (6 start / 7 fid) vs heldout photo 006 (3/8). No model
selection on test; any 11-point estimate has large statistical uncertainty.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from skimage.feature import hog
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s25_real_anchor_anza import dataset


def features(gray,kind):
    gray=np.asarray(gray,np.uint8)
    if gray.shape!=(64,64):raise ValueError('64x64 crop required')
    norm=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(gray)
    if kind=='simple':
        regions=[gray[24:40,24:40],gray[16:48,16:48],gray]
        feat=[]
        for x in regions:
            f=x.astype(np.float32)
            feat.extend([f.mean()/255.,f.std()/255.,np.quantile(f,.1)/255.,
                         np.quantile(f,.9)/255.,float(np.mean(f<np.quantile(gray,.2)))])
        gx=cv2.Sobel(norm,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(norm,cv2.CV_32F,0,1,ksize=3)
        a=np.arctan2(gy,gx);m=np.hypot(gx,gy)
        for center in [(slice(20,44),slice(20,44)),(slice(0,64),slice(0,64))]:
            t=a[center];w=m[center];z=np.sum(w)+1e-9
            feat.extend([float(np.sum(np.cos(2*t)*w)/z),float(np.sum(np.sin(2*t)*w)/z)])
        return np.asarray(feat,float)
    if kind=='hog':
        return hog(norm.astype(np.float32)/255,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),
                   block_norm='L2-Hys').astype(float)
    raise ValueError('unknown feature type')


def run(bmps:Path,labels:Path,out:Path):
    tr,tmeta=dataset(bmps,labels,'005');te,emeta=dataset(bmps,labels,'006')
    y=np.asarray([a for _,a,_ in tr],int);yt=np.asarray([a for _,a,_ in te],int)
    records={}
    for kind in ['simple','hog']:
        X=np.stack([features(im,kind) for im,_,_ in tr]);Z=np.stack([features(im,kind) for im,_,_ in te])
        for algo in ['logistic','rbf_svm']:
            def model():
                if algo=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000))
                return make_pipeline(StandardScaler(),SVC(C=.1,kernel='rbf',class_weight='balanced'))
            clf=model()
            # Model/hyperparams preregistered, no 006 tuning. Train only 005.
            clf.fit(X,y)
            score=clf.decision_function(Z)
            loo=[]
            for tr_idx,val_idx in LeaveOneOut().split(X):
                sub=model();sub.fit(X[tr_idx],y[tr_idx]);loo.append((int(val_idx[0]),float(sub.decision_function(X[val_idx])[0])))
            cvscore=np.asarray([v for i,v in sorted(loo)])
            key=f'{kind}_{algo}'
            records[key]={'train005_leave_one_out_auc':float(roc_auc_score(y,cvscore)),
                         'heldout006_11_point_auc':float(roc_auc_score(yt,score)),
                         'heldout006_11_point_ap':float(average_precision_score(yt,score)),
                         'test_labels':yt.tolist(),'test_scores':score.tolist(),
                         'test_images_n':len(yt),'test_start_n':int(yt.sum()),
                         'not_real_track_segmentation':True}
    report={'schema':'s26_real_anchor_baselines_v1','train':tmeta,'heldout':emeta,
            'target':'human-annotated START vs fiducial location, NOT pixelwise trace mask',
            'cross_run_generalization':False,'methods':records,'hyperparameters_fixed_before_test':True}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmps',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run(a.bmps,a.labels,a.out)
    for k,v in r['methods'].items():print(k,v['train005_leave_one_out_auc'],v['heldout006_11_point_auc'])
