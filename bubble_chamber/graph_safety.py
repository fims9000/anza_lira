"""Synthetic-only Graph-LIRA-inspired PAIR/NONE safety policy.

Never use this synthetic model's decisions to auto-merge real film fragments.
The current pilot has no calibrated labels, no real junction rules, no 3D
optics and no validated threshold for physical events.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .relation_study import Protocol, _features, create_split, threshold_at_fpr


@dataclass(frozen=True)
class GraphSafetyProtocol:
    seed: int = 20266024  # disjoint from earlier S0 seed range
    train_scenes: int = 160
    val_scenes: int = 70
    test_scenes: int = 70
    max_val_candidate_fpr: float = 0.02
    margin_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.25)


def select_global_links(rows: list[dict], probabilities: np.ndarray, *, threshold: float,
                        margin: float = 0.10) -> tuple[np.ndarray, list[str]]:
    """Mutual endpoint best + score margin + one-to-one + cycle abstain.

    No hidden ground truth enters this function. All correspondence decisions
    are based on local model scores and endpoint indices within one scene.
    """
    prob = np.asarray(probabilities, dtype=float)
    if len(rows) != len(prob) or not np.all(np.isfinite(prob)):
        raise ValueError("invalid rows/probabilities")
    if not 0 <= margin <= 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid selection thresholds")
    by_endpoint: dict[tuple, list[int]] = defaultdict(list)
    node_pairs = []
    for k, r in enumerate(rows):
        if not isinstance(r.get('pair'), list) or len(r['pair']) != 4:
            raise ValueError("invalid endpoint pair")
        a,side_a,b,side_b=r['pair']
        if a==b or side_a not in ('A','B') or side_b not in ('A','B'):
            raise ValueError("invalid endpoint identities")
        edge=(r['scene_id'],int(a),side_a),(r['scene_id'],int(b),side_b)
        node_pairs.append(edge)
        for ep in edge:
            by_endpoint[ep].append(k)
    ranking={ep:sorted(ids,key=lambda i:(-prob[i],i)) for ep,ids in by_endpoint.items()}
    accepted=np.zeros(len(rows),bool)
    reasons=['threshold']*len(rows)
    parent={}

    def root(node):
        parent.setdefault(node,node)
        while parent[node] != node:
            node=parent[node]
        return node

    used=set()
    for k in sorted(range(len(rows)),key=lambda i:(-prob[i],i)):
        if prob[k] < threshold:
            continue
        endpoints=node_pairs[k]
        if any(ep in used for ep in endpoints):
            reasons[k]='endpoint_already_used';continue
        if any(ranking[ep][0]!=k for ep in endpoints):
            reasons[k]='not_mutual_best';continue
        if any(len(ranking[ep])>1 and prob[k]-prob[ranking[ep][1]] < margin for ep in endpoints):
            reasons[k]='ambiguous_margin';continue
        # Also prohibit cycles across a chain in the same 2D frame.
        a=(endpoints[0][0],endpoints[0][1]);b=(endpoints[1][0],endpoints[1][1])
        if root(a)==root(b):
            reasons[k]='cycle';continue
        parent[root(a)]=root(b)
        used.update(endpoints)
        accepted[k]=True;reasons[k]='accepted_synthetic_only'
    return accepted,reasons


def count_links(rows: list[dict], accepted: np.ndarray) -> dict[str, Any]:
    y=np.asarray([bool(r['truth_same_track']) for r in rows],bool)
    acc=np.asarray(accepted,bool)
    if acc.shape!=y.shape:
        raise ValueError('label length mismatch')
    tp=int((acc&y).sum());fp=int((acc&~y).sum());fn=int((~acc&y).sum());tn=int((~acc&~y).sum())
    scene_to_false=defaultdict(int)
    for row,match,truth in zip(rows,acc,y):
        if match and not truth:
            scene_to_false[row['scene_id']]+=1
    all_scenes=set(r['scene_id'] for r in rows)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'recall_candidates':round(tp/max(tp+fn,1),6),
            'candidate_fpr':round(fp/max(fp+tn,1),6),
            'false_bridges_per_scene':round(fp/max(len(all_scenes),1),6),
            'scenes_with_false_bridge':len(scene_to_false),
            'fraction_scenes_with_false_bridge':round(len(scene_to_false)/max(len(all_scenes),1),6),
            'accepted_edges':int(acc.sum()),
            'scenario_is_oracle_tracelets':True,
            'real_automatic_merges':0}


def run_graph_safety_study(protocol: GraphSafetyProtocol = GraphSafetyProtocol()) -> dict:
    if not protocol.margin_grid:
        raise ValueError('empty margin grid')
    base=Protocol(seed=protocol.seed,train_scenes=protocol.train_scenes,
                  val_scenes=protocol.val_scenes,test_scenes=protocol.test_scenes,
                  max_val_false_positive_rate=protocol.max_val_candidate_fpr)
    sets={group:create_split(base,group=group) for group in ('train','validation','test')}
    rows={group:sets[group]['rows'] for group in sets}
    xtrain,ytrain=_features(rows['train'],True)
    xval,yval=_features(rows['validation'],True)
    xtest,ytest=_features(rows['test'],True)
    model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=protocol.seed))])
    model.fit(xtrain,ytrain)
    validation_probs=model.predict_proba(xval)[:,1]
    selected_thr=threshold_at_fpr(yval,validation_probs,protocol.max_val_candidate_fpr)
    validation_grid=[]
    for margin in protocol.margin_grid:
        selected,_=select_global_links(rows['validation'],validation_probs,threshold=selected_thr,margin=margin)
        validation_grid.append({'margin':margin,'metrics':count_links(rows['validation'],selected)})
    feasible=[entry for entry in validation_grid if entry['metrics']['candidate_fpr'] <= protocol.max_val_candidate_fpr]
    if feasible:
        chosen=max(feasible,key=lambda x:(x['metrics']['recall_candidates'],-x['metrics']['candidate_fpr'],x['margin']))
        frozen_margin=chosen['margin']
        safety_status='VAL_GATE_REACHED_TEST_NOT_YET_EVALUATED'
    else:
        # No margin met the safety gate; safest default is to abstain from ALL
        # automatic structural joins. Test is still scored diagnostically.
        frozen_margin=None
        safety_status='VAL_GATE_FAILED_ABSTAIN_ALL'
    test_prob=model.predict_proba(xtest)[:,1]
    local_selected=test_prob>=selected_thr
    if frozen_margin is None:
        global_selected=np.zeros(len(rows['test']),bool)
        reasons=['failed_validation_gate']*len(rows['test'])
    else:
        global_selected,reasons=select_global_links(rows['test'],test_prob,threshold=selected_thr,margin=frozen_margin)
    result={'status':'SYNTHETIC_PAIR_NONE_GRAPH_SAFETY_ONLY',
            'protocol':asdict(protocol), 'split_candidate_counts':{g:len(rows[g]) for g in rows},
            'validation_selected_candidate_threshold':float(selected_thr),
            'margin_validation_grid':validation_grid,
            'frozen_margin':frozen_margin,'selection_status':safety_status,
            'test':{'local_probability_only':count_links(rows['test'],local_selected),
                    'graph_constrained':count_links(rows['test'],global_selected),
                    'graph_rejection_reasons':{reason:reasons.count(reason) for reason in sorted(set(reasons))}},
            'allow_real_auto_linking':False,
            'boundary':'Oracle synthetic source fragments only; no real film masks, no H2/propane domain shift, no stereo 3D, no measured physical identities.'}
    return result
