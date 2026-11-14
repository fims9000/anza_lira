"""V28 port availability objective. No real unseen expert TEST accessed."""
from __future__ import annotations

from scripts.tver_v28_topology_calibration import (
    TOPOLOGY_THRESHOLDS,choose,_at_threshold,
)


def _metric(ntrue=0,nfalse=0,ncal=0,src=0,ncand=10,dense=0):
    return {
        "n_true_pairs":ntrue,
        "n_false_pairs":nfalse,
        "n_true_pair_gaps":ncal,
        "n_sources":src,
        "n_candidates":ncand,
        "n_dense_rejected":dense,
    }


def test_refuse_if_no_train_true_predicted_candidates():
    a={str(q):_metric(ntrue=0,nfalse=99) for q in TOPOLOGY_THRESHOLDS}
    b={str(q):_metric(ncal=4,src=4) for q in TOPOLOGY_THRESHOLDS}
    assert choose(a,b) is None


def test_refuse_if_cal_has_no_real_correct_candidate():
    a={str(q):_metric(ntrue=8,nfalse=35) for q in TOPOLOGY_THRESHOLDS}
    b={str(q):_metric(ncal=0,src=4) for q in TOPOLOGY_THRESHOLDS}
    assert choose(a,b) is None


def test_CAL_chooses_highest_true_candidate_coverage_then_lower_false_proposals():
    a={str(q):_metric() for q in TOPOLOGY_THRESHOLDS}
    b={str(q):_metric() for q in TOPOLOGY_THRESHOLDS}
    for q in (.6,.7,.8):
        a[str(q)]=_metric(ntrue=8,nfalse=25)
    b[str(.6)]=_metric(ncal=1,ncand=60,src=4)
    b[str(.7)]=_metric(ncal=2,ncand=55,src=3)
    b[str(.8)]=_metric(ncal=2,ncand=45,src=3)
    assert choose(a,b)==.8


def test_threshold_grid_exact_before_V28_data():
    assert TOPOLOGY_THRESHOLDS==(0.35,0.4,0.45,0.5,0.55,0.6,
         0.65,0.7,0.75,0.8,0.85,0.9)
