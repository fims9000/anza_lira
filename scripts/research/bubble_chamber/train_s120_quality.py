"""Fit a review-prioritization classifier from private, manually marked 007.

Cross-validation groups by complete human stroke ID. Same film is not an
independent run-disjoint evaluation; positive score is not a calibrated probability.
Do not publish private labels or raw/derived user imagery in the GitHub PR.
"""
from __future__ import annotations
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


def model():
    return RandomForestClassifier(n_estimators=200,max_depth=11,min_samples_leaf=12,
        n_jobs=4,random_state=12,class_weight='balanced_subsample')


def train(features_file:Path,model_file:Path,report_file:Path,threshold=.5):
    z=np.load(features_file);labels=z['labels'];mask=labels!='unknown'
    X=z['features'][mask];y=(labels[mask]=='track').astype(int)
    groups=z['label_ids'][mask];id_count=Counter(groups.tolist())
    if len(set(groups))<10 or len(set(y))!=2:
        raise ValueError('Insufficient independent annotated stroke IDs')
    weights=np.asarray([1./np.sqrt(id_count[int(k)]) for k in groups]);weights/=weights.mean()
    oof=np.full(len(y),np.nan)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=27)
    for tr,te in cv.split(X,y,groups):
        fit=model();fit.fit(X[tr],y[tr],sample_weight=weights[tr]);oof[te]=fit.predict_proba(X[te])[:,1]
    if not np.isfinite(oof).all():raise RuntimeError('OOF predictions missing')
    cm=confusion_matrix(y,oof>=threshold,labels=[0,1])
    final=model();final.fit(X,y,sample_weight=weights)
    report={'feature_names':z['names'].tolist(),'label_source_sha256':str(z['source_sha256']),
            'labeled_curve_instances':len(y),'distinct_annotated_strokes':len(set(groups.tolist())),
            'manual_track_instances':int(y.sum()),'manual_trash_instances':int((1-y).sum()),
            'cross_val':'5-fold stratified by human stroke ID, SAME FRAME; NOT independent external test',
            'oof_auc':float(roc_auc_score(y,oof)),'display_threshold':float(threshold),
            'oof_track_instance_retention':float(cm[1,1]/max(1,cm[1].sum())),
            'oof_trash_instance_retention':float(cm[0,1]/max(1,cm[0].sum())),
            'unlabeled_instances_not_negative_class':True,
            'score_not_calibrated_probability':True,'accuracy_on_006_unknown':True}
    model_file.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(final,model_file)
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--features',type=Path,required=True)
    a.add_argument('--model',type=Path,required=True)
    a.add_argument('--report',type=Path,required=True)
    q=a.parse_args();print(json.dumps(train(q.features,q.model,q.report),indent=2))
