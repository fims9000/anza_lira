"""S24 low-label, safety-first continuation consensus and human-review routing.

Inputs to `route_provisional_disagreements` are OBSERVED graph scores,
not truth labels. The oracle in `replay_review_answers` simulates a human only
for the pairs selected for review, using held-out visualMetro identities. It
is NOT a claim that real H50Z track identity can be determined automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections import defaultdict
import math

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class AuditConfig:
    calibration_reviews: int = 80
    independent_audit_reviews: int = 80
    training_reviews: int = 640
    review_budgets: tuple[int, ...] = (20, 40, 80)
    training_gaps: tuple[int, ...] = (1, 3, 5)
    test_gap: int = 5
    split_seeds: tuple[int, ...] = (20261030, 20261031, 20261032)
    matching_margin: float = 0.05
    support_quantile: float = .995

    def validate(self):
        if (self.calibration_reviews < 20 or self.independent_audit_reviews < 20
            or self.training_reviews < 40 or self.calibration_reviews > 500
            or self.independent_audit_reviews > 500
            or tuple(sorted(set(self.review_budgets))) != self.review_budgets
            or max(self.review_budgets) > self.training_reviews
            or self.training_gaps != (1, 3, 5) or self.test_gap != 5
            or len(set(self.split_seeds)) != len(self.split_seeds)
            or not 0 <= self.matching_margin < 1 or not .9 <= self.support_quantile < 1):
            raise ValueError('Invalid registered S24 low-label protocol')


def validate_scene_matching(rows: list[dict], accepted: np.ndarray) -> None:
    """Reject invalid graph merges: one terminal may connect at most once."""
    if len(accepted) != len(rows):
        raise ValueError('candidate count mismatch')
    if not np.asarray(accepted).dtype == bool:
        raise ValueError('expected boolean link decisions')
    used_left=set();used_right=set();seen=set()
    for r, yes in zip(rows, accepted):
        key=(str(r['scene_id']),str(r['left_id']),str(r['right_id']))
        if key in seen:raise ValueError('duplicate candidate identity')
        seen.add(key)
        if not yes:continue
        u=(key[0],key[1]);v=(key[0],key[2])
        if u in used_left or v in used_right:
            raise ValueError('inconsistent one-to-one scene graph')
        used_left.add(u);used_right.add(v)


def conservative_consensus(rows: list[dict], geometry: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Agree-only matching: any disagreement goes to human review, never auto joins."""
    g=np.asarray(geometry,dtype=bool);h=np.asarray(graph,dtype=bool)
    validate_scene_matching(rows,g);validate_scene_matching(rows,h)
    chosen=g&h
    validate_scene_matching(rows,chosen)
    return chosen


def split_validation_frames(rows:list[dict], seed:int)->tuple[np.ndarray,np.ndarray]:
    """Two genuinely disjoint sets of original frames inside ONE film.

    The audit is never used to select classifier, support, matching margin or
    decision threshold. Frame separation is necessary but insufficient for
    independent film/run generalization.
    """
    frames=sorted({str(r['frame_id']) for r in rows})
    if len(frames)<8:raise ValueError('at least 8 validation frames required')
    ranked=sorted(frames,key=lambda f:sha256(f'{seed}|s24-audit-frame|{f}'.encode()).digest())
    midpoint=len(ranked)//2
    cal=set(ranked[:midpoint]);audit=set(ranked[midpoint:])
    left=np.asarray([i for i,r in enumerate(rows) if str(r['frame_id']) in cal],dtype=int)
    right=np.asarray([i for i,r in enumerate(rows) if str(r['frame_id']) in audit],dtype=int)
    if len(left)==0 or len(right)==0 or set(left)&set(right):raise ValueError('bad holdout')
    return left,right


def unlabeled_audit_indices(rows:list[dict], budget:int,seed:int)->list[int]:
    """Deterministic hash sample; never reads hidden `truth_same_track`.

    Statistical population: uniformly sampled candidate pairs from audit
    frames (not independently sampled physical runs). A strong confidence
    bound on accepted-link error is NOT obtained from this alone.
    """
    if budget<=0 or budget>len(rows):raise ValueError('invalid audit budget')
    def key(i):
        r=rows[i]
        identity=f"{r['scene_id']}|{r['left_id']}|{r['right_id']}"
        return sha256(f'{seed}|s24-audit|{identity}'.encode()).digest()
    return sorted(range(len(rows)),key=key)[:budget]


def candidate_error_bound(failures:int, reviewed_accepted:int,alpha:float=.05):
    """One-sided exact iid Bernoulli upper confidence limit.

    Display is DESCRIPTIVE ONLY on within-film, nonindependent, potentially
    selection-biased reviews; not a certification of physical error risk.
    """
    if not (0 <= failures <= reviewed_accepted and 0 < alpha < 1):
        raise ValueError('invalid confidence interval counts')
    if reviewed_accepted==0:return None
    if failures==reviewed_accepted:return 1.0
    return float(beta.ppf(1-alpha,failures+1,reviewed_accepted-failures))


def zero_error_review_sample_size(target_fraction:float,alpha:float=.05)->int:
    """IID-only number of randomly audited *accepted joins* with 0 errors.

    Derived from (1-p)^n <= alpha; NOT an independent-run guarantee.
    Multiple related joins from one film invalidate naive iid interpretation.
    """
    if not (0 < target_fraction < 1 and 0 < alpha < 1):
        raise ValueError('target risk and alpha must be between 0 and 1')
    return int(math.ceil(math.log(alpha)/math.log1p(-target_fraction)))


def audit_labeled_subset(rows:list[dict],accepted:np.ndarray,reviewed:list[int])->dict:
    if len(accepted)!=len(rows) or len(set(reviewed))!=len(reviewed) or any(not 0<=i<len(rows) for i in reviewed):
        raise ValueError('invalid audit inputs')
    checked=np.asarray(reviewed,int)
    y=np.asarray([bool(rows[i]['truth_same_track']) for i in checked])
    sel=np.asarray(accepted,bool)[checked]
    n_selected=int(sel.sum());bad=int((sel&~y).sum());correct=int((sel&y).sum())
    return {'reviews':len(checked),'accepted_among_reviewed':n_selected,
            'correct_accepted':correct,'wrong_accepted':bad,
            'descriptive_iid_95pct_upper_error_fraction_among_accepted':candidate_error_bound(bad,n_selected),
            'physical_certification_possible':False,
            'sampling_note':'Hash-ranked candidate samples on validation frames of the same film; not i.i.d. photo-run observations'}


def route_provisional_disagreements(rows:list[dict],geom_accepted:np.ndarray,
                                    graph_accepted:np.ndarray,geom_score:np.ndarray,
                                    graph_score:np.ndarray,seed:int)->list[int]:
    """Review potentially useful one-model links; no hidden labels or IDs as features.

    1) Only proposals made by one model, but not both, are considered.
    2) Prefer strong disagreement in observable scores, then diversify frames.
    This ranking does not read candidate `truth_same_track`.
    """
    g=np.asarray(geom_accepted,bool);h=np.asarray(graph_accepted,bool)
    validate_scene_matching(rows,g);validate_scene_matching(rows,h)
    gs=np.asarray(geom_score,float);hs=np.asarray(graph_score,float)
    if len(gs)!=len(rows) or len(hs)!=len(rows) or not np.isfinite(gs).all() or not np.isfinite(hs).all():
        raise ValueError('invalid model scores')
    candidate_ids=np.where(g^h)[0].tolist()
    def key(i):
        r=rows[i]
        conf=float(max(gs[i],hs[i]));disagree=float(abs(gs[i]-hs[i]))
        digest=sha256(f"{seed}|{r['scene_id']}|{r['left_id']}|{r['right_id']}".encode()).hexdigest()
        return (-conf,-disagree,digest)
    ranked=sorted(candidate_ids,key=key)
    # Frame diversity is a weak guard against asking 80 questions about one
    # almost-identical local cluster. Process rounds until all are included.
    per_frame=defaultdict(list)
    for i in ranked:per_frame[str(rows[i]['frame_id'])].append(i)
    order=[]
    while any(per_frame.values()):
        for frame in sorted(per_frame):
            if per_frame[frame]:order.append(per_frame[frame].pop(0))
    return order


def replay_review_answers(rows:list[dict], consensus:np.ndarray,review_order:list[int],budgets:tuple[int,...])->dict:
    """OFFLINE oracle simulates human only for reviewed pairs (test-only study).

    Verified review positives may add a link only if no endpoint conflicts;
    no automatic review labels are inferred for the remaining candidates.
    """
    current=np.asarray(consensus,bool).copy()
    validate_scene_matching(rows,current)
    if len(set(review_order))!=len(review_order) or any(not 0<=i<len(rows) for i in review_order):
        raise ValueError('bad review queue')
    if tuple(sorted(set(budgets)))!=budgets or any(n<0 for n in budgets):
        raise ValueError('budgets must increase')
    used_l={(rows[i]['scene_id'],rows[i]['left_id']) for i in np.flatnonzero(current)}
    used_r={(rows[i]['scene_id'],rows[i]['right_id']) for i in np.flatnonzero(current)}
    reviewed=0;added=0;negatives=0;conflict=0;results={}
    for budget in budgets:
        upto=min(budget,len(review_order))
        for i in review_order[reviewed:upto]:
            r=rows[i]
            if not bool(r['truth_same_track']):
                negatives+=1
                continue
            u=(r['scene_id'],r['left_id']);v=(r['scene_id'],r['right_id'])
            if u in used_l or v in used_r:
                conflict+=1
                continue
            current[i]=True;used_l.add(u);used_r.add(v);added+=1
        reviewed=upto
        validate_scene_matching(rows,current)
        results[str(budget)]={'human_pairs_reviewed':reviewed,
                              'manually_verified_correct_joins_added':added,
                              'reviewed_different_track_rejected':negatives,
                              'reviewed_correct_but_conflicting':conflict,
                              'provisional_total_accepted':int(current.sum()),
                              'provisional_false_joins':int(sum(bool(x) and not bool(r['truth_same_track']) for x,r in zip(current,rows))),
                              'offline_oracle_only':True,
                              'automated_physical_merge_authorized':False}
    return results
