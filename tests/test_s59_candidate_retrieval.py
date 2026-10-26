"""S59 candidate gates: immutable frozen filters; no generator label dependency."""
import math
import pytest
from bubble_chamber.s59_candidate_retrieval import POLICIES,accepts_pair,filter_rows

def row(gap=18,ta=.75,tb=.65,mutual=.5, label=0):
    return {'endpoint_gap_px':gap,'outgoing_tangent_A_to_gap':ta,
            'outgoing_tangent_B_to_gap':tb,'outgoing_tangent_mutual':mutual,
            'label_SYNTHETIC_ONLY':label}

def test_strict_gate_matches_s39_inequalities():
    assert accepts_pair(row(), 'strict_s39')
    assert not accepts_pair(row(gap=3), 'strict_s39')
    assert accepts_pair(row(gap=36), 'strict_s39')
    assert not accepts_pair(row(gap=36.1), 'strict_s39')
    assert not accepts_pair(row(ta=.25), 'strict_s39')
    assert accepts_pair(row(mutual=0), 'strict_s39')
    assert not accepts_pair(row(mutual=-.01),'strict_s39')

def test_relaxed_includes_strict_and_restores_borderline_but_not_far():
    a=row(gap=40,ta=.2,tb=.3,mutual=-.1)
    assert not accepts_pair(a,'strict_s39')
    assert accepts_pair(a,'relaxed_s59')
    assert not accepts_pair(row(gap=49),'relaxed_s59')
    assert not accepts_pair(row(ta=0),'relaxed_s59')
    assert accepts_pair(row(gap=49,ta=-1,tb=-1,mutual=-1),
                        'all_pairs_oracle_upper_bound')

def test_label_blind_and_preserves_source_rows():
    a,b=row(label=0),row(label=1)
    assert accepts_pair(a,'strict_s39')==accepts_pair(b,'strict_s39')
    x=[row(label=0),row(gap=46,label=1),row(gap=100,label=0)]
    chosen=filter_rows(x,'relaxed_s59')
    assert chosen==x[:2] and all(v is x[k] for k,v in enumerate(chosen))

def test_fails_closed_bad_geometry():
    with pytest.raises(ValueError):accepts_pair(row(), 'never')
    for bad in (float('nan'),float('inf'),-1.):
        with pytest.raises(ValueError):accepts_pair(row(gap=bad),'strict_s39')
    with pytest.raises(ValueError):accepts_pair(row(ta=1.1),'strict_s39')
