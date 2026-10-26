"""S84 image-only extra-fragment filter based on local pixel + graph context.

The source S72 2D polylines stay intact, extra instances are *hypotheses*, and
no physical particle identity or synthetic reference enters inference. CAL
synthetic truth is allowed only in separate, post-inference evaluation.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .s56_global_path_assembly import pair_rows
from .s59_candidate_retrieval import filter_rows
from .s83_unsaturated_local_support import local_dark_stroke_support

ARMS=('baseline','s84_photo_q40','s84_context_single','s84_context_bridge')
THRESHOLD_KEYS=('q25','q40')
SINGLE_SCORE_DELTA=.04
BRIDGE_SCORE_DELTA=.02
SINGLE_RIVAL_MARGIN=.03


def thresholds_from_CAL_pixel_scores(scores:list[float])->dict[str,float]:
    x=np.asarray(scores,float)
    if x.ndim!=1 or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
        raise ValueError('Pixel-only CAL feature must be finite and normalized')
    return {'q25':float(np.quantile(x,.25)) if len(x) else 1.0,
            'q40':float(np.quantile(x,.40)) if len(x) else 1.0}


def _score_original_contacts(original:list[np.ndarray],extras:list[np.ndarray],scorer):
    """Only S56 score + pixel-extracted curve geometry; no reference nodes."""
    n=len(original)
    if n+len(extras)<3:return {i:[] for i in range(len(extras))}
    rows=filter_rows(pair_rows([*original,*extras]),'strict_s39')
    if not rows:return {i:[] for i in range(len(extras))}
    vals=np.asarray(scorer(rows),float)
    if vals.shape!=(len(rows),) or not np.isfinite(vals).all() or np.any((vals<0)|(vals>1)):
        raise ValueError('Frozen S56 scorer must return bounded probabilities')
    contacts=defaultdict(list)
    for row,score in zip(rows,vals):
        i,j=int(row['i']),int(row['j'])
        if i<n<=j:
            contacts[j-n].append({'score':float(score),'original_port':(i,int(row['end_i'])),
                                    'extra_port':int(row['end_j']),'original':i})
    return {i:sorted(contacts[i],key=lambda v:(-v['score'],v['original'],v['extra_port']))
            for i in range(len(extras))}


def gate_additional_lines(gray:np.ndarray,original:list[np.ndarray],consensus:list[np.ndarray],
                          baseline_selected,scorer,cutoff:float,arm:str,thresholds:dict[str,float])->dict:
    """Retain source objects, select only image-supported candidate extra lines.

    The existing S72 forest is used ONLY to avoid stealing its selected
    endpoints; the GT evaluator cannot affect this filter. No graph edge is
    forced even if an extra passes; graph re-optimization follows later.
    """
    if arm not in ARMS or not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Frozen S84 arm or source pixel image invalid')
    if set(thresholds)!=set(THRESHOLD_KEYS) or not np.isfinite(list(thresholds.values())).all() or any(
            v<0 or v>1 for v in thresholds.values()):
        raise ValueError('Source CAL q25/q40 thresholds invalid')
    if not np.isfinite(cutoff) or not 0<=cutoff<=1 or len(original)>32 or not len(original)<=len(consensus)<=32:
        raise ValueError('Invalid source-only graph policy / bounded candidates')
    if any(a is not b for a,b in zip(original,consensus)):
        raise ValueError('Original S72 2D curves must remain identical Python objects')
    out={'polylines':list(original),'extra_considered':len(consensus)-len(original),
         'extra_accepted':0,'extra_rejected':len(consensus)-len(original),
         'rejected_photo':0,'rejected_weak_graph':0,'rejected_conflict':0,
         'rejected_ambiguous':0,'physical_auto_joins_authorized':False,
         'native_GT_count':0}
    if arm=='baseline':
        out.update(extra_considered=0,extra_rejected=0)
        return out
    extras=consensus[len(original):]
    # Gating never alters any original S72 object and never uses a GT label.
    support=[local_dark_stroke_support(gray,e) for e in extras]
    candidate_contacts=(_score_original_contacts(original,extras,scorer)
                        if arm!='s84_photo_q40' else {})
    used_ports={port for e in baseline_selected for port in e.ports()}
    accepted=[]
    for j,(p,s) in enumerate(zip(extras,support)):
        threshold=thresholds['q25' if arm=='s84_context_bridge' else 'q40']
        if s<threshold:
            out['rejected_photo']+=1;continue
        if arm=='s84_photo_q40':
            accepted.append(p);continue
        contacts=candidate_contacts[j]
        strong=[c for c in contacts if c['score']>=cutoff+(
            BRIDGE_SCORE_DELTA if arm=='s84_context_bridge' else SINGLE_SCORE_DELTA)]
        free=[c for c in strong if c['original_port'] not in used_ports]
        if not strong:
            out['rejected_weak_graph']+=1;continue
        if not free:
            out['rejected_conflict']+=1;continue
        if arm=='s84_context_single':
            # Strongest admissible contact must be unambiguous among free contacts.
            if len(free)>1 and free[0]['score']-free[1]['score']<SINGLE_RIVAL_MARGIN-1e-12:
                out['rejected_ambiguous']+=1;continue
            accepted.append(p)
        else:
            pairs=[(a,b) for ii,a in enumerate(free) for b in free[ii+1:] if
                   a['original']!=b['original'] and a['extra_port']!=b['extra_port'] and
                   a['original_port']!=b['original_port']]
            if not pairs:
                out['rejected_ambiguous']+=1;continue
            accepted.append(p)
    out['polylines'].extend(accepted)
    out['extra_accepted']=len(accepted)
    out['extra_rejected']=len(extras)-len(accepted)
    assert out['extra_rejected']==sum(out[k] for k in (
        'rejected_photo','rejected_weak_graph','rejected_conflict','rejected_ambiguous'))
    assert len(out['polylines'])<=32 and all(a is b for a,b in zip(original,out['polylines']))
    return out


def choose_CAL_only(stats:dict)->dict:
    """This function can inspect synthetic CAL aggregate labels, NOT TEST GT."""
    conditions=('clean','blur_sigma_11')
    req=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
         'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
         'solver_failure','extra_accepted')
    if set(stats)!=set(ARMS) or any(set(stats[a])!=set(conditions) for a in ARMS):
        raise ValueError('Complete S84 source-CAL grid required')
    for a in ARMS:
        for c in conditions:
            if any(x not in stats[a][c] or type(stats[a][c][x]) is not int or stats[a][c][x]<0 for x in req):
                raise ValueError('Invalid CAL-only synthetic aggregate')
    b=stats['baseline'];passed=[]
    for arm in ARMS[1:]:
        x,y=stats[arm]['blur_sigma_11'],stats[arm]['clean']
        bx,by=b['blur_sigma_11'],b['clean']
        if (x['selected_synthetic_correct_links']>=bx['selected_synthetic_correct_links']+2 and
            x['shape_true_fragments_matched']>=bx['shape_true_fragments_matched']+2 and
            x['synthetic_noncorrect_selected']<=bx['synthetic_noncorrect_selected'] and
            x['complete_three_piece_paths_synthetic']>=bx['complete_three_piece_paths_synthetic'] and
            y['selected_synthetic_correct_links']>=by['selected_synthetic_correct_links']-1 and
            y['synthetic_noncorrect_selected']<=by['synthetic_noncorrect_selected'] and
            y['complete_three_piece_paths_synthetic']>=by['complete_three_piece_paths_synthetic'] and
            x['solver_failure']==0 and y['solver_failure']==0):
            passed.append(arm)
    chosen=min(passed,key=lambda arm:(-stats[arm]['blur_sigma_11']['selected_synthetic_correct_links'],
            stats[arm]['blur_sigma_11']['synthetic_noncorrect_selected'],
            stats[arm]['blur_sigma_11']['extra_accepted'],arm)) if passed else 'baseline'
    return {'CAL_only_selected':chosen,'eligible':passed,'original_default_retained':chosen=='baseline',
            'real_physics_auto_joins_authorized':False}
