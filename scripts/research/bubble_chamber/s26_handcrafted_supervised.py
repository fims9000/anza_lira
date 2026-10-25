#!/usr/bin/env python3
"""Strong same-budget hand-crafted pixel baselines vs original ANZA on H50Z.

Training uses ONLY 005 synthetic trace/distractor marks. Unreviewed native
photo pixels are UNKNOWN and never converted to class 0. Test 006 untouched
until final evaluation. Not ground truth on native particle track identity.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_two_photo_baselines import sample_two_photos,pixel_input,classical_score,select_threshold,score_measures


def local_features(im):
    x=np.asarray(im,np.float32)/255.
    local=pixel_input(im,'local')
    bh=classical_score(im,'blackhat')
    ridge=classical_score(im,'steerable_ridge')
    gx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)
    grad=np.hypot(gx,gy)
    noise=np.maximum(cv2.GaussianBlur(x*x,(0,0),2)-cv2.GaussianBlur(x,(0,0),2)**2,0)**.5
    contrast=cv2.GaussianBlur(x,(0,0),4)-x
    channels=[1.-x,local,bh,ridge,grad,noise,contrast]
    return np.stack(channels,axis=-1)


def known_features(examples):
    x=[];y=[];group=[]
    for i,e in enumerate(examples):
        features=local_features(e.gray)
        known=e.sparse!=255
        x.append(features[known]);y.append(e.sparse[known].astype(int));group.extend([i]*int(known.sum()))
    return np.concatenate(x,axis=0),np.concatenate(y),np.asarray(group)


def run(args):
    ds,source=sample_two_photos(args.bmps,args.measurements,args.seed,train_n=144,val_n=48,test_n=96)
    X,y,_=known_features(ds['train']);V,vy,_=known_features(ds['val']);T,ty,tg=known_features(ds['test'])
    specs={
     'local_only_logreg':(make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=1500,random_state=args.seed)),[1]),
     'multi_feature_logreg':(make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=1500,random_state=args.seed)),list(range(7))),
     'multi_feature_HGB':(HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=13,min_samples_leaf=18,l2_regularization=2,learning_rate=.065,random_state=args.seed),list(range(7))),
    }
    report={'schema':'s26_photo005_synthetic_supervised_handcrafted_v1','seed':args.seed,
            'n_known_training_labels':len(y),'n_known_validation_labels':len(vy),'n_known_test_labels':len(ty),
            'labels_real_native_tracks':0,'heldout_photo':'006','checkpoint_tuning_on_test':False,
            'results':{}}
    for name,(model,feat) in specs.items():
        st=time.monotonic();model.fit(X[:,feat],y)
        # The validation threshold uses 005 labels; no evaluation of 006 used for selection.
        vs=model.predict_proba(V[:,feat])[:,1]
        cal=select_threshold(vs,vy)
        te=model.predict_proba(T[:,feat])[:,1]
        report['results'][name]={'validation':cal,'heldout_006':score_measures(ty,te,cal['threshold'],tg),
                'training_seconds':time.monotonic()-st,'input_feature_indices':feat}
        print('HANDCRAFTED',args.seed,name,round(report['results'][name]['heldout_006']['roc_auc_synthetic_marks'],4),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmps',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();run(args)
