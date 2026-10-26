"""S37: conservative selection of *review suggestions* on synthetic track gaps.

All calibration/evaluation labels in S37 are created by drawing artificial strokes
onto two authentic but unlabeled H50Z photographs. A risk upper bound is valid
only for exchangeable, independently generated synthetic scenes conditional on
this simulation; it is NOT a guarantee for real events, physical tracks or runs.

Multiple calibration thresholds are covered by a Bonferroni-adjusted one-sided
Clopper-Pearson upper bound on the event ANY FALSE SUGGESTED LINK PER SCENE.
No external test labels are read for selection. If no nonempty cutoff is safe,
all candidates abstain.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class CalibrationDecision:
    threshold: float
    observed_false_scenes: int
    calibration_scenes: int
    calibration_correct: int
    upper_scene_false_probability: float
    all_abstain: bool
    eligible_safe_cutoffs_including_empty: int
    n_examined_cutoffs: int
    scope: str = 'SIMULATED_SCENES_ONLY__NOT_PHYSICAL_SAFETY'
    status: str = 'MACHINE_PROVISIONAL_REVIEW_ONLY'


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided binomial upper confidence endpoint, no asymptotic CI."""
    if not isinstance(k, (int,np.integer)) or not isinstance(n,(int,np.integer)) or not 0<=k<=n or n==0:
        raise ValueError('0 <= false scenes <= number of calibration scenes required')
    if not (0 < alpha < 1):
        raise ValueError('0 < alpha < 1 required')
    return 1.0 if k==n else float(beta.ppf(1-alpha, k+1, n-k))


def review_selection(rows, scores, threshold, *, margin=.04):
    """Source-independent mutual-best matching: never auto-connect a real track.

    Rows are indexed synthetic left/right candidates; all unknown pairs are
    omitted and no selection ever has physics_authorized=True.
    """
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or scores.shape!=(len(rows),) or not np.isfinite(scores).all():
        raise ValueError('Invalid score vector')
    if not 0 <= margin < 1:raise ValueError('Bad ambiguity margin')
    if np.isnan(threshold):raise ValueError('NaN threshold')
    left=sorted({int(r['left_index']) for r in rows})
    right=sorted({int(r['right_index']) for r in rows})
    pairs={(int(r['left_index']),int(r['right_index'])):float(v) for r,v in zip(rows,scores)}
    if len(pairs)!=len(rows) or not all(i>=0 for i in left+right):raise ValueError('Invalid candidate IDs')
    selected=[]
    for i in left:
        candidates=sorted(((v,j) for (li,j),v in pairs.items() if li==i),reverse=True)
        if not candidates:continue
        best,j=candidates[0]
        if best<threshold or len(candidates)>1 and best-candidates[1][0]<margin:continue
        others=[v for (li,rj),v in pairs.items() if rj==j and li!=i]
        if others and best-max(others)<margin:continue
        selected.append((i,j))
    return tuple(selected)


def scene_stats(scenes, scores, threshold, *, margin=.04):
    """Assess link selection per whole scene; no per-pair pseudoindependence."""
    lengths=[len(s['pairs']) for s in scenes]
    s=np.asarray(scores,dtype=float)
    if s.ndim!=1 or sum(lengths)!=len(s) or len(scenes)==0 or not np.isfinite(s).all():
        raise ValueError('Invalid complete scene scores')
    i=0;correct=wrong=any_wrong=accepted=0
    mask=[]
    for scene,n in zip(scenes,lengths):
        chosen=review_selection(scene['pairs'],s[i:i+n],threshold,margin=margin);i+=n
        known={(int(r['left_index']),int(r['right_index'])):int(r['label_SYNTHETIC_ONLY']) for r in scene['pairs']}
        if set(known.values())!={0,1}:raise ValueError('Scene needs explicit synthetic positives and negatives')
        nc=sum(known[p]==1 for p in chosen);nw=sum(known[p]==0 for p in chosen)
        correct+=nc;wrong+=nw;any_wrong+=int(nw>0);accepted+=len(chosen)
        mask.append((nc,nw))
    return {'n_scenes':len(scenes),'correct_links':correct,'wrong_links':wrong,
            'any_false_scene_count':any_wrong,'accepted_suggestions':accepted,
            'correct_links_per_scene':correct/len(scenes),
            'false_scenes_fraction':any_wrong/len(scenes),
            'false_links_per_100_scenes':100*wrong/len(scenes),
            'simulation_only':True,'native_track_ground_truth_count':0,
            'per_scene_counts_private':mask}


def candidate_cutoffs(train_scores, *, n_cutoffs=13):
    """Choose cutoffs exclusively from TRAIN scores, never from CAL/TEST."""
    s=np.asarray(train_scores,float)
    if s.ndim!=1 or len(s)<10 or not np.isfinite(s).all() or n_cutoffs<3:
        raise ValueError('Need training scores and >=3 cutoffs')
    q=np.linspace(.65,.999,n_cutoffs)
    thresholds=np.quantile(s,q)
    return tuple(sorted(set(float(x) for x in thresholds)))


def calibrate_review_risk(train_scores, calibration_scenes, calibration_scores,
                          *, target_scene_false_risk=.02, family_alpha=.05,
                          n_cutoffs=13, margin=.04):
    """Pick most productive safe cutoff with simultaneous finite-sample UCB.

    Family-wise error is controlled only under independent/exchangeable synthetic
    calibration scenes. Actual data repeat one physical run and must NEVER be
    described as having certified physical risk. `inf` is the safe fallback.
    """
    if not 0<target_scene_false_risk<1:raise ValueError('Target outside (0,1)')
    choices=candidate_cutoffs(train_scores,n_cutoffs=n_cutoffs)
    n=len(calibration_scenes)
    if n==0:raise ValueError('Missing calibration scenes')
    selected=[]
    for threshold in choices:
        counts=scene_stats(calibration_scenes,calibration_scores,threshold,margin=margin)
        hi=cp_upper(counts['any_false_scene_count'],n,family_alpha/len(choices))
        if hi<=target_scene_false_risk:
            selected.append((counts['correct_links'],-threshold,threshold,counts,hi))
    if selected:
        _,_,cut,stats,hi=max(selected,key=lambda x:(x[0],x[1]))
        return CalibrationDecision(threshold=cut,observed_false_scenes=stats['any_false_scene_count'],
            calibration_scenes=n,calibration_correct=stats['correct_links'],
            upper_scene_false_probability=hi,all_abstain=stats['accepted_suggestions']==0,
            eligible_safe_cutoffs_including_empty=len(selected),n_examined_cutoffs=len(choices))
    # Even an all-abstain rule cannot be 'certified' from too few scenes using
    # empirical confidence, though operationally it cannot create a false join.
    worst_safe=cp_upper(0,n,family_alpha/max(len(choices),1))
    return CalibrationDecision(threshold=float('inf'),observed_false_scenes=0,
        calibration_scenes=n,calibration_correct=0,
        upper_scene_false_probability=worst_safe,all_abstain=True,
        eligible_safe_cutoffs_including_empty=0,n_examined_cutoffs=len(choices))


def public_stats(stats:dict)->dict:
    """Strip all per-scene outcomes and private evidence before sharing."""
    return {k:v for k,v in stats.items() if k!='per_scene_counts_private'}
