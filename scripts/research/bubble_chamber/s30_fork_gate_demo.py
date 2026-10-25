#!/usr/bin/env python3
"""Deterministic contract experiment only, not a true particle-track benchmark."""
import argparse,json
from pathlib import Path
import numpy as np
from bubble_chamber.s30_trajectory_identity import gate_candidates


def item(y,score,support=.8):
    t=np.linspace(0,150,76)
    return {'path_points_original_xy':np.stack([t,t*0+y],axis=-1).tolist(),
            'model_selection_score_UNCALIBRATED':score,
            'fraction_steps_above_local_signal_0_15':support}


def run():
    cases={
      'duplicated_same_path':[item(0,.9),item(.4,.87)],
      'ambiguous_fork':[item(0,.9),item(16,.88)],
      'low_signal':[item(0,.8,.04)],
      'single_hypothesis':[item(0,.8,.8)],
    }
    rows=[]
    for name, candidates in cases.items():
        result=gate_candidates(candidates)
        rows.append({'case':name,'status':result['decision'], 'physical_joins_accepted':int(result['physical_join_authorized']),
                     'unique_hypotheses':result['unique_path_count'],'input_candidates':len(candidates)})
    return {'schema':'s30_deterministic_fork_contract_only','source':'constructed mathematical centerlines, NOT native H50Z photo labels',
            'physics_validation':False,'real_photo_false_link_rate':'NOT_MEASURED',
            'cases':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    r=run();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
