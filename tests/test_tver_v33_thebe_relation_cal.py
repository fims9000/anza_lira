"""V33 independent real-field source section CAL risk, not candidate-level pseudo-n."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.tver_v33_thebe_relation_cal import (
    upper_clopper_pearson,_section_calibrate,
    RELATION_CAL_SOURCE_IDS,RELATION_CAL_MIN_ELIGIBLE_SECTIONS,
    FRONTENDS,
)


def _scene(section,kind,y=1):
    return {
        "id":str(section),"section":section,"kind":kind,
        "proposals":[{"y":y}],"status":"OK",
        "source_count":1,"branch_count":2,
        "has_true_proposal":y>0,
    }


def test_exact_upper_confidence_binomial_one_sided():
    assert upper_clopper_pearson(0,4)==pytest.approx(0.5271291955)
    assert upper_clopper_pearson(0,29)<.10
    assert upper_clopper_pearson(0,28)>.10
    assert upper_clopper_pearson(29,29)==1.
    with pytest.raises(ValueError):
        upper_clopper_pearson(0,0)


def test_risk_CAL_distinct_source_section_gate_minimum():
    s=[];probs=[]
    for sid in range(40,68):  # 28 distinct eligible sections only
        s.extend((_scene(sid,"LINK"),_scene(sid,"NO_LINK",0)))
        probs.extend((np.asarray([.95]),np.asarray([.01])))
    result=_section_calibrate(s,probs)
    assert result["status"].startswith("STOP")
    assert result["accept_threshold"]==1.01
    assert result["n_sections"]==28


def test_risk_CAL_can_admit_only_correct_links_29_sections():
    s=[];probs=[]
    for sid in range(40,69):
        s.extend((_scene(sid,"LINK"),_scene(sid,"NO_LINK",0)))
        probs.extend((np.asarray([.95]),np.asarray([.05])))
    result=_section_calibrate(s,probs)
    assert result["status"]=="CAL_SECTION_RISK_CONDITIONALLY_FROZEN"
    assert result["source_section_count"]==29
    assert result["accept_threshold"]==pytest.approx(.95)
    assert result["CAL_only_section_risk_metrics"][
        "correctly_accepted_positive_gap_cases"]==29
    assert result["CAL_only_section_risk_metrics"][
        "nolink_CP_upper_95"]<.10


def test_one_wrong_link_in_each_section_triggers_never_link_abstention():
    s=[];scores=[]
    for sid in range(40,69):
        s.extend((_scene(sid,"LINK",0),_scene(sid,"NO_LINK",0)))
        scores.extend((np.asarray([.99]),np.asarray([.99])))
    result=_section_calibrate(s,scores)
    assert result["status"]=="CAL_SECTION_RISK_CONDITIONALLY_FROZEN"
    assert result["accept_threshold"]==1.01
    assert result["CAL_only_section_risk_metrics"][
        "correctly_accepted_positive_gap_cases"]==0


def test_fixed_relation_section_partition():
    assert FRONTENDS==("unet","anza_v1")
    assert list(RELATION_CAL_SOURCE_IDS)==list(range(40,100))
    assert RELATION_CAL_MIN_ELIGIBLE_SECTIONS==29
