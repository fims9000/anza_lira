"""S21: auditable low-label graph continuation and selective-risk experiment.

The REAL inputs are manually measured visualMetro coordinates on *one film*.
Occlusions are SYNTHETICALLY introduced by withholding expert clicks. No AZConv
pixel observations, no real photo instance masks, no physical PID or calibrated
momentum claims are possible from this study. This module never authorizes
unattended physical joins.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, asdict
from hashlib import sha256
import math
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .real_sparse_lira import _accept


@dataclass(frozen=True)
class S21Protocol:
    source_seed: int = 20261025
    label_budgets: tuple[int, ...] = (160,320,640)
    warm_start: int = 60
    review_batch: int = 40
    max_from_one_frame_per_batch: int = 4
    cal_max_false_joins: int = 0
    margin: float = 0.05
    support_train_quantile: float = 0.995
    bootstrap_repeats: int = 750
    def validate(self):
        if (not self.label_budgets or tuple(sorted(set(self.label_budgets)))!=self.label_budgets
            or self.warm_start<20 or self.warm_start>self.label_budgets[0]
            or self.review_batch<=0 or self.max_from_one_frame_per_batch<1
            or self.cal_max_false_joins<0 or not 0<=self.margin<1
            or not .9<=self.support_train_quantile<1
            or self.bootstrap_repeats<100):
            raise ValueError('invalid S21 protocol')


def grouped_graph_features(rows: list[dict]) -> tuple[np.ndarray,np.ndarray]:
    """Purely observed graph-context features; never use labels, track identity or film IDs as numeric features.

    Each context is a bipartite candidate scene: left/right nodes are observed
    fragments. Second-best gaps reveal ambiguous competing continuations.
    """
    if not rows:raise ValueError('empty candidate set')
    left=defaultdict(list);right=defaultdict(list)
    for i,r in enumerate(rows):
        s=r['scene_id']
        if not s or not isinstance(r['geometry_axial_fuzzy'],list):raise ValueError('bad candidate rows')
        if not np.isfinite(r['selection_score']):raise ValueError('nonfinite rank score')
        left[(s,r['left_id'])].append(i)
        right[(s,r['right_id'])].append(i)
    base=np.asarray([r['geometry_axial_fuzzy'] for r in rows],dtype=np.float64)
    if base.ndim!=2 or base.shape[1]!=14 or not np.isfinite(base).all():
        raise ValueError('expected finite 14D geometry features')
    conf=np.exp(-np.clip(np.asarray([r['selection_score'] for r in rows],float),0,100))
    extra=np.zeros((len(rows),10),float)
    for i,r in enumerate(rows):
        li=left[(r['scene_id'],r['left_id'])];ri=right[(r['scene_id'],r['right_id'])]
        # Ranking is label-blind and uses ONLY observed geometric selection scores.
        ls=sorted([conf[j] for j in li],reverse=True)
        rs=sorted([conf[j] for j in ri],reverse=True)
        mine=conf[i]
        lbetter=sum(conf[j]>mine+1e-14 for j in li)
        rbetter=sum(conf[j]>mine+1e-14 for j in ri)
        lrunner=max((conf[j] for j in li if j!=i),default=0.)
        rrunner=max((conf[j] for j in ri if j!=i),default=0.)
        extra[i]=[
            mine, math.log1p(len(li)),math.log1p(len(ri)),
            mine-lrunner,mine-rrunner,
            ls[0]-mine,rs[0]-mine,
            lbetter,rbetter,
            abs((mine-lrunner)-(mine-rrunner)),
        ]
    return base,np.concatenate([base,extra],axis=1)


def stable_candidate_id(row:dict)->str:
    """No label leakage into deterministic acquisition order."""
    parts=(row['scene_id'],row['left_id'],row['right_id'])
    return sha256('|'.join(map(str,parts)).encode('utf-8')).hexdigest()


def acquire_human_pairs(rows:list[dict],full_features:np.ndarray,baseline_features:np.ndarray,
                        budget:int,*,seed:int,warm_start:int=60,batch_size:int=40,max_per_frame:int=4):
    """Simulate human reviews using withheld train track IDs ONLY after selection.

    Unreviewed rows never become negative. Validation/test never passed here.
    Returns indices and an audit log (human-label oracle is simulated track ID).
    """
    n=len(rows)
    if n<budget or len(full_features)!=n or len(baseline_features)!=n:
        raise ValueError('bad label acquisition budget')
    if warm_start<10 or max_per_frame<1:raise ValueError('bad acquisition policy')
    rng=np.random.default_rng(seed)
    seen=set();selected=[];batches=[]
    keys=[stable_candidate_id(row) for row in rows]
    order=sorted(range(n),key=lambda i:keys[i])
    # First batch diversifies by original frame, not by candidate label.
    def add_batch(ranked,count):
        taken=[];per_frame=defaultdict(int)
        for i in ranked:
            if i in seen:continue
            f=rows[i]['frame_id']
            if per_frame[f]>=max_per_frame:continue
            taken.append(i);per_frame[f]+=1
            if len(taken)>=count:break
        if len(taken)<count:
            for i in ranked:
                if i not in seen and i not in taken:
                    taken.append(i)
                    if len(taken)==count:break
        for i in taken:seen.add(i)
        selected.extend(taken)
        return taken
    first=add_batch(order[:],min(warm_start,budget))
    batches.append({'batch':0,'labels_consumed':len(first),'acquisition':'hash-diverse-unlabeled','class_labels_consulted_for_selection':False})
    while len(selected)<budget:
        # Use ONLY the current selected training labels to fit an acquisition model.
        y=np.asarray([int(rows[i]['truth_same_track']) for i in selected],int)
        if len(set(y))==2:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.4,class_weight='balanced',max_iter=1000,random_state=seed))
            model.fit(full_features[selected],y)
            prob=model.predict_proba(full_features)[:,1]
            uncertainty=np.abs(prob-.5)
        else:
            # No oracle class peeking: choose new diverse observed candidates.
            uncertainty=np.asarray([int(k[:12],16)/2**48 for k in keys],float)
        # Alternate uncertain and diverse scores (no random test supervision).
        scored=sorted(range(n),key=lambda i:(uncertainty[i],keys[i]))
        count=min(batch_size,budget-len(selected))
        taken=add_batch(scored,count)
        batches.append({'batch':len(batches),'labels_consumed':len(taken),
                        'acquisition':'training-only uncertainty / frame coverage',
                        'class_labels_consulted_for_selection':False})
    return selected,batches


def fit_selective_models(rows,base,context,reviewed,seed:int):
    if min(len(rows),len(base),len(context))<=0 or len(reviewed)<20:
        raise ValueError('insufficient supervised examples')
    y=np.asarray([int(rows[i]['truth_same_track']) for i in reviewed],int)
    if len(set(y))<2:
        raise ValueError('need both same/different expert labels')
    # Same human-reviewed pairs for both arms, no extra labels for graph-context model.
    variants={
        'geometry_logreg':(base,make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=seed))),
        'context_logreg':(context,make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=seed))),
        'context_hgb':(context,HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=9,min_samples_leaf=7,learning_rate=.07,l2_regularization=5,random_state=seed)),
    }
    result={}
    for name,(xx,est) in variants.items():
        est.fit(xx[reviewed],y)
        result[name]=(est,'context' if xx.shape[1]==24 else 'base')
    return result


def _support_cut(train_geometry:np.ndarray, quantile:float)->float:
    """Training-observation support: OOD gap distances abstain. No test labels."""
    if train_geometry.ndim!=2 or train_geometry.shape[1]<1:
        raise ValueError('invalid geometry')
    x=train_geometry[:,0]
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('bad distance geometry')
    return float(np.quantile(x,quantile))


def accept_with_support(rows,scores,threshold,margin,dist_cut):
    """Existing mutual-best matcher plus source-only gap/scale support gate."""
    if len(rows)!=len(scores):raise ValueError('score count mismatch')
    scores=np.asarray(scores,float)
    if not np.isfinite(scores).all():raise ValueError('nonfinite model probabilities')
    score_accepted=_accept(rows,scores,threshold,margin)
    supported=np.asarray([r['geometry_axial_fuzzy'][0]<=dist_cut for r in rows],bool)
    return score_accepted & supported


def calibrate_zero_false_joins(val_rows:list[dict],scores:np.ndarray,*,margin:float,
                               dist_cut:float,max_false:int=0,
                               reviewed_indices:list[int]|None=None):
    """Tune an abstention threshold on labeled *reviewed validation pairs* only.

    Geometry ranking still uses ALL proposed candidates in the scene; removing
    unreviewed candidates from matching would cause optimistic leakage. Labels
    of unreviewed validation candidates are NEVER accessed. A zero observed
    reviewed false-link count is not a population FPR guarantee.
    """
    if len(val_rows)!=len(scores) or not val_rows:raise ValueError('bad val candidate set')
    if reviewed_indices is None: reviewed_indices=list(range(len(val_rows)))
    if len(set(reviewed_indices))!=len(reviewed_indices) or any(not 0<=int(i)<len(val_rows) for i in reviewed_indices):
        raise ValueError('invalid reviewed validation indices')
    known=np.zeros(len(val_rows),bool);known[reviewed_indices]=True
    y=np.zeros(len(val_rows),bool)
    for i in reviewed_indices: y[i]=bool(val_rows[i]['truth_same_track'])
    if not y[known].any() or y[known].all():raise ValueError('reviewed validation needs both classes')
    values=np.unique(np.asarray(scores,float))
    if not np.isfinite(values).all():raise ValueError('nonfinite validation probabilities')
    candidate_valid=accept_with_support(val_rows,scores,-1.0,margin,dist_cut)
    # Calculate threshold sweep on precomputed geometric matching. Without it,
    # every threshold would redundantly re-sort all scene candidate graphs.
    eligible=np.where(candidate_valid)[0]
    # Full-abstain sentinel is always feasible; no test labels are read here.
    choices=np.r_[np.nextafter(float(values[-1]),np.inf),np.unique(scores[eligible])]
    best=None
    for threshold in choices:
        acc=candidate_valid&(scores>=threshold)
        tp=int((acc&known&y).sum()); fp=int((acc&known&~y).sum())
        if fp>max_false:continue
        key=(tp,float(threshold))
        if best is None or key>best[0]:best=(key,float(threshold),tp,fp,int((acc&~known).sum()))
    if best is None:raise AssertionError('all-abstain solution must be feasible')
    return {'threshold':best[1],'val_reviewed_tp':best[2],
            'val_reviewed_fp':best[3],
            'validation_reviewed_pairs':int(known.sum()),
            'validation_reviewed_positive_pairs':int(y.sum()),
            'selected_unreviewed_val_candidates':best[4],
            'zero_observed_false_joins_on_reviewed_pairs':best[3]==0,
            'reviewed_only_calibration':True,
            'physical_release_authorized':False}


def blind_validation_reviews(rows:list[dict], budget:int,seed:int=20261024)->list[int]:
    """Deterministic top-rank + diverse source-frame reviews, ignoring labels.

    Validation review cost is budgeted IN ADDITION TO training reviews.
    """
    if budget<20 or budget>len(rows):raise ValueError('invalid calibration budget')
    # Alternate easy confident and hard ambiguous candidates across frames,
    # no access to the withheld validation truth.
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['candidate_rank'],rows[i]['selection_score'],stable_candidate_id(rows[i])))
    by_frame=defaultdict(list)
    for i in order:by_frame[rows[i]['frame_id']].append(i)
    frame_order=sorted(by_frame,key=lambda f:sha256(f'{seed}|{f}'.encode()).digest())
    chosen=[];seen=set();
    while len(chosen)<budget:
        made=False
        for f in frame_order:
            if by_frame[f]:
                i=by_frame[f].pop(0)
                if i in seen:continue
                chosen.append(i);seen.add(i);made=True
                if len(chosen)==budget:break
        if not made:break
    if len(chosen)!=budget:raise ValueError('not enough distinct validation candidates')
    return chosen

def group_metrics(rows:list[dict],accepted:np.ndarray)->dict[str,Any]:
    y=np.asarray([bool(r['truth_same_track']) for r in rows])
    if len(y)!=len(accepted):raise ValueError('mismatch')
    a=np.asarray(accepted,bool)
    tp=int((a&y).sum());fp=int((a&~y).sum())
    false_by_frame=defaultdict(int);by_frame=defaultdict(lambda:[0,0,0])
    for row,yes,truth in zip(rows,a,y):
        if yes and not truth:false_by_frame[row['frame_id']]+=1
        by_frame[row['frame_id']][0]+=int(yes and truth)
        by_frame[row['frame_id']][1]+=int(yes and not truth)
        by_frame[row['frame_id']][2]+=int(truth)
    return {'tp':tp,'fp':fp,'positive_candidates':int(y.sum()),
            'negative_candidates':int((~y).sum()),
            'recall_on_present_candidates':round(tp/max(y.sum(),1),6),
            'accepted_links':int(a.sum()),'rejected_or_uncertain_candidates':int((~a).sum()),
            'false_join_rate_per_accepted':round(fp/max(a.sum(),1),6),
            'false_joins_per_frame':round(fp/max(len(by_frame),1),6),
            'frames_with_any_false_join':len(false_by_frame),
            'frames':len(by_frame),
            'release_to_physics':False}


def paired_frame_ci(rows,a,b,seed:int=20261024,repeats:int=750):
    if len(a)!=len(rows) or len(b)!=len(rows):raise ValueError('candidate mismatch')
    by=defaultdict(lambda:np.zeros(3,float))
    for row,ap,bp in zip(rows,a,b):
        truth=bool(row['truth_same_track']);bucket=by[row['frame_id']]
        bucket[0]+=int(truth);bucket[1]+=int(ap and truth);bucket[2]+=int(bp and truth)
    if len(by)<5:raise ValueError('need multiple held-out frames')
    counts=np.stack(list(by.values()));rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(counts),(repeats,len(counts)))
    d=counts[ids].sum(1)
    dif=(d[:,2]-d[:,1])/np.maximum(d[:,0],1)
    return {'delta_recall_b_minus_a':float((counts[:,2].sum()-counts[:,1].sum())/max(counts[:,0].sum(),1)),
            'frame_bootstrap_ci95':[float(q) for q in np.quantile(dif,[.025,.975])],
            'same_film_cluster_only':True}
