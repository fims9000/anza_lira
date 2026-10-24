"""S14 few-expert-decisions Graph-LIRA proxy on REAL visualMetro clicked geometry.

The hidden interior measurements are ARTIFICIALLY omitted; labels are track
identities already recorded by operators. This experiment cannot establish
accuracy on real photographic segmentation or independent chamber runs.

Never inspect withheld test labels during frame acquisition, threshold fitting,
training, or feature selection. All unreviewed pairs remain UNKNOWN to training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .large_relation_study import frame_partition
from .real_sparse_lira import SparseProtocol, _accept, _threshold_at_fpr, make_examples


@dataclass(frozen=True)
class LabelBudgetProtocol:
    seed: int = 20261024
    hidden_clicks: int = 2
    cuts: tuple[float, ...] = (0.5,)
    k_candidates: int = 3
    initial_reviews: int = 20
    budgets: tuple[int, ...] = (20, 40, 80, 160, 320, 640)
    review_batch_size: int = 20
    validation_max_candidate_fpr: float = 0.01
    margin: float = 0.05
    max_new_candidates_per_frame: int = 3

    def validate(self):
        if (self.initial_reviews < 10 or len(self.budgets) < 2 or
            self.budgets[0] != self.initial_reviews or
            sorted(set(self.budgets)) != list(self.budgets) or
            self.review_batch_size < 1 or self.max_new_candidates_per_frame < 1 or
            self.k_candidates < 2 or self.hidden_clicks < 1 or
            not (0 < self.validation_max_candidate_fpr < 0.2) or
            not (0 <= self.margin < 1)):
            raise ValueError('invalid expert-review budget study')


def _row_id(row: dict) -> str:
    """Stable human-review query identity; independent of whether truth is known."""
    return f"{row['frame_id']}:{row['view']}:{row['cut_index']}:{row['left_id']}:{row['right_id']}"


def _frames_split(rows: list[dict], frames: list[str], split: dict[str, list[str]]):
    ids = {part: set(names) for part, names in split.items()}
    if any(ids[x] & ids[y] for x,y in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('overlapping frame groups')
    if set(frames) != ids['train'] | ids['val'] | ids['test']:
        raise ValueError('incomplete split')
    sets = {part: [row for row in rows if row['frame_id'] in ids[part]] for part in ids}
    if any(not sets[part] for part in sets):
        raise ValueError('empty frame group')
    return sets


def _fit(rows: list[dict], selected: list[int], feature: str, seed: int):
    if not selected:
        raise ValueError('no reviewed pairs')
    xx = np.asarray([rows[i][feature] for i in selected], dtype=np.float64)
    yy = np.asarray([int(rows[i]['truth_same_track']) for i in selected], dtype=np.int64)
    if set(yy) != {0,1} or not np.isfinite(xx).all():
        raise ValueError('need both expert-reviewed pair classes with finite features')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000,
                                        class_weight='balanced', random_state=seed))
    model.fit(xx, yy)
    return model


def _prob(model, rows: list[dict], feature: str) -> np.ndarray:
    xx = np.asarray([r[feature] for r in rows], dtype=np.float64)
    return model.predict_proba(xx)[:,1]


def _scores(rows: list[dict], prob: np.ndarray, threshold: float, margin: float):
    truth = np.asarray([bool(r['truth_same_track']) for r in rows], dtype=bool)
    pred = _accept(rows, prob, threshold, margin)
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    frames = len({r['frame_id'] for r in rows})
    return {'positive_candidates': int(truth.sum()), 'accepted_connections': int(pred.sum()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': round(tp/max(tp+fn,1),6),
            'precision': round(tp/max(tp+fp,1),6),
            'fpr': round(fp/max(fp+tn,1),6),
            'false_connections_per_frame': round(fp/max(frames,1),6),
            'test_frames':frames}


def _initial_indices(rows: list[dict], n: int, seed: int):
    """Choose a label-blind random start. Reveal truth only AFTER selection."""
    order=np.random.default_rng(seed).permutation(len(rows))
    selected=order[:n].tolist()
    # Never use truth to force a positive or negative seed. If initial random
    # labels lack a class, reject this split rather than selecting oracle rows.
    if len({bool(rows[i]['truth_same_track']) for i in selected}) < 2:
        raise ValueError('initial random expert decisions lack a class')
    return selected


def propose_uncertain_reviews(rows: list[dict], probabilities: np.ndarray, seen: set[int],
                              count: int, max_per_frame: int = 3):
    """Score WITHOUT labels; distribute human queries over many original frames."""
    prob=np.asarray(probabilities, dtype=float)
    if len(prob) != len(rows) or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
        raise ValueError('invalid model probabilities')
    if count <= 0 or max_per_frame < 1:
        raise ValueError('invalid review batch count')
    by_frame=Counter()
    # Binary entropy maximized at p=.5. Tie-break stable by candidate identity.
    def entropy(p):
        p=float(np.clip(p,1e-12,1-1e-12))
        return -p*np.log(p)-(1-p)*np.log(1-p)
    order=sorted((i for i in range(len(rows)) if i not in seen),
                 key=lambda i: (-entropy(prob[i]), _row_id(rows[i])))
    chosen=[]
    for i in order:
        fid=str(rows[i]['frame_id'])
        if by_frame[fid]>=max_per_frame:continue
        chosen.append(i);by_frame[fid]+=1
        if len(chosen)>=count:break
    if len(chosen)<count:
        for i in order:
            if i not in chosen:
                chosen.append(i)
                if len(chosen)>=count:break
    return chosen


def propose_balanced_reviews(rows: list[dict], probabilities: np.ndarray, seen: set[int],
                             count: int, max_per_frame: int = 3):
    """Three-rail acquisition (likely-positive / boundary / likely-negative).

    No hidden truth is used. Queries remain human-review suggestions, not
    pseudo-verified relation labels. Frame caps protect acquisition diversity.
    """
    p=np.asarray(probabilities, dtype=float)
    if len(p) != len(rows) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or count<1:
        raise ValueError('invalid balanced acquisition inputs')
    available=set(range(len(rows)))-set(seen)
    if count>len(available):raise ValueError('budget exceeds available candidate pool')
    if max_per_frame<1:raise ValueError('invalid frame cap')
    rails=[(.75,round(count*.4)),(.5,round(count*.4)),(.25,0)]
    rails[2]=(.25,count-rails[0][1]-rails[1][1])
    selected=[];used_frames=Counter()
    for target,quota in rails:
        if quota<=0:continue
        order=sorted(available, key=lambda i:(abs(p[i]-target),_row_id(rows[i])))
        for i in order:
            if i in selected or used_frames[rows[i]['frame_id']]>=max_per_frame:continue
            selected.append(i);used_frames[rows[i]['frame_id']]+=1
            quota-=1
            if quota==0:break
    if len(selected)<count:
        for i in sorted(available, key=lambda i:(abs(p[i]-.5),_row_id(rows[i]))):
            if i in selected or used_frames[rows[i]['frame_id']]>=max_per_frame:continue
            selected.append(i);used_frames[rows[i]['frame_id']]+=1
            if len(selected)==count:break
    if len(selected)<count:
        for i in sorted(available, key=lambda i:(abs(p[i]-.5),_row_id(rows[i]))):
            if i in selected:continue
            selected.append(i)
            if len(selected)==count:break
    return selected


def _candidate_digest(rows: list[dict]) -> str:
    # The frame/source identifiers are from expert data but NEVER include labels.
    h=sha256()
    for row in rows:
        h.update((_row_id(row)+'|'+','.join(map(str,row['geometry_generic']))+'\n').encode())
    return h.hexdigest()


def experiment(manifest: dict, protocol: LabelBudgetProtocol = LabelBudgetProtocol()) -> dict[str, Any]:
    protocol.validate()
    if manifest.get('audit_schema') != 'visualmetro_large_audit_v1':
        raise ValueError('expected audited visualMetro measurements')
    if manifest['summary']['distinct_film_ids'] != ['070']:
        raise ValueError('this protocol is restricted to source film 070')
    frame_ids=[fr['frame_id'] for fr in manifest['frames']]
    split=frame_partition(frame_ids,protocol.seed)
    all_rows, stats = make_examples(manifest, SparseProtocol(cuts=protocol.cuts,
        k_candidates=protocol.k_candidates, missing_clicks=protocol.hidden_clicks,seed=protocol.seed))
    sets=_frames_split(all_rows,frame_ids,split)
    train, val, test=[sets[part] for part in ('train','val','test')]
    if len(train)<max(protocol.budgets):
        raise ValueError('insufficient candidates for review budget')

    seed_indices=_initial_indices(train,protocol.initial_reviews,protocol.seed)
    trajectories={name:{'indices':list(seed_indices),'rng':np.random.default_rng(protocol.seed+91),
                        'history':[]} for name in ('uncertainty','balanced','random')}
    chosen_index_trace={}
    for budget in protocol.budgets:
        for selector, item in trajectories.items():
            selected=item['indices']
            if len(selected)<budget:
                remaining=budget-len(selected)
                if selector in ('uncertainty','balanced'):
                    current_model=_fit(train,selected,'geometry_generic',protocol.seed)
                    current_prob=_prob(current_model,train,'geometry_generic')
                    if selector=='uncertainty':
                        picked=propose_uncertain_reviews(train,current_prob,set(selected),remaining,
                            max_per_frame=protocol.max_new_candidates_per_frame)
                    else:
                        picked=propose_balanced_reviews(train,current_prob,set(selected),remaining,
                            max_per_frame=protocol.max_new_candidates_per_frame)
                else:
                    left=np.asarray([i for i in range(len(train)) if i not in set(selected)],dtype=int)
                    picked=item['rng'].choice(left,size=remaining,replace=False).tolist()
                if len(set(picked))!=remaining or set(picked)&set(selected):raise AssertionError('duplicate review selection')
                selected.extend(picked)
            step={'budget_labels':budget,'reviewed_frames':len({train[i]['frame_id'] for i in selected}),
                  'reviewed_positive_pairs':sum(bool(train[i]['truth_same_track']) for i in selected),
                  'reviewed_negative_pairs':sum(not train[i]['truth_same_track'] for i in selected),
                  'models':{}}
            for name,features in [('geometry_baseline','geometry_generic'),('axial_geometry_proxy','geometry_axial_fuzzy')]:
                model=_fit(train,selected,features,protocol.seed)
                val_probs=_prob(model,val,features)
                val_truth=np.asarray([int(r['truth_same_track']) for r in val])
                threshold=_threshold_at_fpr(val_truth,val_probs,protocol.validation_max_candidate_fpr)
                test_probs=_prob(model,test,features)
                step['models'][name]={'val_threshold':float(threshold),
                   'val':_scores(val,val_probs,threshold,protocol.margin),
                   'test':_scores(test,test_probs,threshold,protocol.margin)}
            item['history'].append(step)
        chosen_index_trace[str(budget)]={name:len(set(t['indices'])) for name,t in trajectories.items()}
    # Save a query list for expert review, NOT candidate labels. These are from
    # the remaining training pool only; test/val group IDs never appear.
    selected=trajectories['balanced']['indices']
    current=_fit(train,selected,'geometry_generic',protocol.seed)
    probs=_prob(current,train,'geometry_generic')
    queries=propose_balanced_reviews(train,probs,set(selected),min(24,len(train)-len(selected)),max_per_frame=2)
    review_queue=[{'frame_id':train[i]['frame_id'],'film_id':train[i]['film_id'],
                   'view':int(train[i]['view']),'scene_id':train[i]['scene_id'],
                   'left_fragment':train[i]['left_id'],'right_fragment':train[i]['right_id'],
                   'candidate_key':_row_id(train[i]),'model_probability_not_confidence':round(float(probs[i]),6),
                   'label': 'UNKNOWN_UNREVIEWED',
                   'annotation_task': 'same_track | different_track | uncertain'} for i in queries]
    return {'schema':'S14_LOW_LABEL_GRAPH_LIRA_PROXY_V1',
      'provenance_sha256':manifest['provenance_sha256'],
      'candidate_table_sha256':_candidate_digest(all_rows),
      'protocol':asdict(protocol),
      'split':{name:{'frames':len(split[name]),'candidates':len(sets[name])} for name in split},
      'candidate_coverage':{'true_possible':stats['true_possible'],'candidate_true':stats['candidate_true'],
                  'ratio':round(stats['candidate_true']/max(stats['true_possible'],1),6)},
      'label_budget_unit':'pair review decisions, NOT annotated frames or pixel masks',
      'procedures':{name:item['history'] for name,item in trajectories.items()},
      'next_review_queue_example':review_queue,
      'decision':'PROXY_ONLY_CANNOT_AUTOMATICALLY_MERGE_REAL_TRACKS',
      'limitations':['The operator already labeled the original 070 track identities: this simulates withholding labels, not collecting new expert reviews.',
                     'Only measured geometries; missing clicks are artificial and no matching real film BMP or real pixel masks are present.',
                     'All frames are from one film: no propane/hydrogen or new camera/run transfer is demonstrated.',
                     'Axial features are hand-designed ANZA-inspired quantities, not original trained AZConv or image-feature Graph-LIRA.',
                     'Unreviewed candidates are never treated as different-track negatives; review queue stores no ground truth.',
                     'Validation frame identities calibrate thresholds at each budget; independent-source calibration is still needed.']}


def save_experiment(manifest:dict, dest:Path, protocol:LabelBudgetProtocol=LabelBudgetProtocol()):
    data=experiment(manifest,protocol)
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'metrics.json').write_text(json.dumps({k:v for k,v in data.items() if k!='next_review_queue_example'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (dest/'review_queue_example.json').write_text(json.dumps({'schema':'s14_expert_pair_queries_v1','source':data['candidate_table_sha256'],
       'queries':data['next_review_queue_example'],'instructions':'source 070 simulated gaps only; NOT actual real photo review'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return data
