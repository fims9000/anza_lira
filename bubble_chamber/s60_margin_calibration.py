"""CAL-only risk-limited margin selection for global synthetic path assembly.

All labels are synthetic simulator labels; caller must NOT supply native physics
annotations.  Test results have no role in select_margin().
"""
from __future__ import annotations

MARGINS=(0., .01, .02, .03, .04, .06, .08)
BASELINE_MARGIN=.04


def select_margin(calibration_counts:dict)->dict:
    """Freeze source-photo policy using CAL counts, never TEST observations."""
    if set(calibration_counts)!={str(x) for x in MARGINS}:
        raise ValueError('Frozen calibration margin grid incomplete')
    required=('scenes','correct_selected','wrong_selected','any_wrong_scenes',
              'full_paths_selected','solver_failed_scenes')
    for key,row in calibration_counts.items():
        if any(v not in row for v in required):
            raise ValueError('Incomplete synthetic CAL statistics')
        if row['scenes']!=64 or any(not isinstance(row[x],int) or row[x]<0 for x in required):
            raise ValueError('Need nonnegative per-source 64-scene CAL counts')
    base=calibration_counts[str(BASELINE_MARGIN)]
    eligible=[]
    for margin in MARGINS:
        r=calibration_counts[str(margin)]
        if (margin!=BASELINE_MARGIN and r['solver_failed_scenes']==0
                and r['any_wrong_scenes']<=base['any_wrong_scenes']
                and r['correct_selected']>=base['correct_selected']+8
                and r['full_paths_selected']>=base['full_paths_selected']):
            eligible.append(margin)
    chosen=sorted(eligible,key=lambda m:(
        -calibration_counts[str(m)]['correct_selected'],
        calibration_counts[str(m)]['any_wrong_scenes'],
        -calibration_counts[str(m)]['full_paths_selected'],m))[0] if eligible else BASELINE_MARGIN
    return {'selected_margin_CAL_only':float(chosen),
            'baseline_margin':BASELINE_MARGIN,
            'eligible_challenger_margins_CAL_only':eligible,
            'cal_min_8_additional_correct_and_no_more_wrong_scenes':True}
