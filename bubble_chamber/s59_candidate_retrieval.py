"""S59 locked geometry-only candidate retrieval; labels never enter this function.

All nodes are **machine/proxy** 2-D lines. Returns reversible pair hypotheses,
not particle identity. The original S39 gates are preserved as a reference.
"""
from __future__ import annotations

from typing import Any
import numpy as np

POLICIES = {
    'strict_s39': {'min_gap_exclusive': 3., 'max_gap_inclusive':36.,
                   'min_tangent_exclusive':.25, 'min_mutual_inclusive':0.},
    'relaxed_s59': {'min_gap_exclusive': 3., 'max_gap_inclusive':48.,
                   'min_tangent_exclusive':.05, 'min_mutual_inclusive':-.20},
    'all_pairs_oracle_upper_bound': None,
}


def accepts_pair(row:dict[str, Any], policy:str)->bool:
    """Test geometry only. Never reads annotations or pair-label fields."""
    if policy not in POLICIES:
        raise ValueError('Unknown frozen S59 candidate policy')
    keys=('endpoint_gap_px','outgoing_tangent_A_to_gap',
          'outgoing_tangent_B_to_gap','outgoing_tangent_mutual')
    vals=np.asarray([row[k] for k in keys],dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError('Candidate features must be finite')
    gap,ta,tb,mutual=(float(v) for v in vals)
    if gap<0 or any(abs(t)>1.00001 for t in (ta,tb,mutual)):
        raise ValueError('Invalid geometric measures')
    rule=POLICIES[policy]
    if rule is None:
        return True
    return (rule['min_gap_exclusive'] < gap <= rule['max_gap_inclusive']
            and min(ta,tb)>rule['min_tangent_exclusive']
            and mutual>=rule['min_mutual_inclusive'])


def filter_rows(rows:list[dict],policy:str)->list[dict]:
    """Preserves every original index and score field; never relabels records."""
    return [row for row in rows if accepts_pair(row,policy)]
