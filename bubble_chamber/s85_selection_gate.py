"""S85 CAL-only whole-graph promotion rule for learned fragment selectors.

Synthetic CAL aggregate labels are permitted here; no ground truth is used by
training feature extraction, frozen prediction or native H50Z inference.
"""
from __future__ import annotations

from .s85_fragment_ranker import ARMS

CONDITIONS = ('clean', 'blur_sigma_11', 'scratch_grain')
FIELDS = ('selected_synthetic_correct_links', 'synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic', 'shape_true_fragments_matched',
          'extra_accepted', 'solver_failure')


def choose_cal_only(stats, trained: bool):
    if set(stats) != set(ARMS) or any(set(stats[arm]) != set(CONDITIONS) for arm in ARMS):
        raise ValueError('Incomplete frozen S85 source-CAL grid')
    for arm in ARMS:
        for cond in CONDITIONS:
            row = stats[arm][cond]
            if any(key not in row or type(row[key]) is not int or row[key] < 0 for key in FIELDS):
                raise ValueError('Missing or invalid synthetic CAL aggregate')
    allowed = []
    if trained:
        ref = stats['baseline']
        for arm in ARMS[1:]:
            blur = stats[arm]['blur_sigma_11']
            base_blur = ref['blur_sigma_11']
            good = (blur['selected_synthetic_correct_links'] >= base_blur['selected_synthetic_correct_links']+2
                    and blur['synthetic_noncorrect_selected'] <= base_blur['synthetic_noncorrect_selected']
                    and blur['complete_three_piece_paths_synthetic'] >= base_blur['complete_three_piece_paths_synthetic']
                    and blur['shape_true_fragments_matched'] >= base_blur['shape_true_fragments_matched']+2)
            for cond in ('clean', 'scratch_grain'):
                a, b = stats[arm][cond], ref[cond]
                good &= (a['selected_synthetic_correct_links'] >= b['selected_synthetic_correct_links']-1
                         and a['synthetic_noncorrect_selected'] <= b['synthetic_noncorrect_selected']
                         and a['complete_three_piece_paths_synthetic'] >= b['complete_three_piece_paths_synthetic'])
            good &= all(stats[arm][c]['solver_failure'] == 0 for c in CONDITIONS)
            if good:
                allowed.append(arm)
    selected = (min(allowed,
                    key=lambda arm: (-stats[arm]['blur_sigma_11']['selected_synthetic_correct_links'],
                                     stats[arm]['blur_sigma_11']['synthetic_noncorrect_selected'],
                                     stats[arm]['blur_sigma_11']['extra_accepted'],
                                     0 if arm == 's85_learned_p95' else 1))
                if allowed else 'baseline')
    return {'CAL_selected': selected, 'eligible': allowed,
            'train_model_available': bool(trained), 'S72_remains_default': selected == 'baseline',
            'native_physics_auto_joins_authorized': False}
