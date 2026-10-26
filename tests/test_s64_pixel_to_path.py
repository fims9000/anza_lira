"""S64 zero-leak pixel-only inference and stricter one-to-one synthetic evaluation."""
import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s64_pixel_to_path import (
    ARMS,_candidate_associations,infer_from_pixels,audit_synthetic_scene)
from scripts.research.bubble_chamber.s64_benchmark_pixel_to_path import SEEDS,SCENES_PER_SEED


def nodes():
    out=[]
    for base_y in (36,96):
        for left in (20,70,120):
            out.append(np.column_stack((np.linspace(left,left+27,31),np.full(31,base_y))))
    out.append(np.column_stack((np.linspace(20,47,31),np.full(31,157))))
    return out


def test_inference_never_has_an_identity_argument(monkeypatch):
    import bubble_chamber.s64_pixel_to_path as m
    calls=[]
    def prediction(gray):
        calls.append(gray.shape)
        return {name:[a.copy() for a in nodes()[:3]] for name in ARMS}
    monkeypatch.setattr(m,'predict_variants',prediction)
    gray=np.full((192,192),201,dtype=np.uint8)
    calls_fit=[]
    def scorer(rows):
        assert all('label_SYNTHETIC_ONLY' not in row for row in rows)
        calls_fit.append(len(rows))
        return np.full(len(rows),.81)
    result=infer_from_pixels(gray,scorer,.8,arm='s39_original')
    assert calls==[(192,192)]
    assert result['physics_auto_joins_authorized'] is False
    assert result['pair_count']==2 and calls_fit==[2] # x 47->70 and 97->120
    with pytest.raises(TypeError):infer_from_pixels(gray,scorer,.8,arm='s39_original',reference=nodes())
    with pytest.raises(ValueError):infer_from_pixels(gray.astype(float),scorer,.8,arm='s39_original')


def test_matching_is_one_to_one_reversal_stable_and_not_union():
    reference=nodes()
    predicted=[reference[0][::-1].copy(),reference[1].copy(),reference[1].copy()]
    matched,qa=_candidate_associations(predicted,reference)
    assert len(matched)==2 and set(matched.values())=={0,1}
    assert qa['unmatched']==1
    # A line spanning both source fragments must NOT be made into one GT.
    mixed=np.vstack((reference[0][:16],reference[1][16:]))
    matched2,qa2=_candidate_associations([mixed],reference)
    assert matched2=={} and qa2['unmatched']==1


def test_audit_correct_ports_wrong_ports_and_unknown():
    source=nodes()
    base={'polylines':[x.copy() for x in source[:3]],'pair_count':2,
          'solver_success':True,'physics_auto_joins_authorized':False}
    good=dict(base,chosen=[Edge(0,1,-1,0,.92),Edge(1,2,-1,0,.91)])
    measured=audit_synthetic_scene(good,source)
    assert measured['selected_synthetic_correct_links']==2
    assert measured['complete_three_piece_paths_synthetic']==1
    assert measured['synthetic_noncorrect_selected']==0
    wrong=dict(base,chosen=[Edge(0,1,0,-1,.92)])
    tested=audit_synthetic_scene(wrong,source)
    assert tested['selected_synthetic_wrong_endpoint']==1
    assert tested['complete_three_piece_paths_synthetic']==0
    unnamed=dict(base,polylines=[source[0].copy(),source[1].copy(),
          np.column_stack((np.linspace(20,47,31),np.full(31,185)))],
          chosen=[Edge(0,2,-1,0,.99)])
    report=audit_synthetic_scene(unnamed,source)
    assert report['selected_synthetic_unmatched_or_ambiguous']==1
    assert report['selected_synthetic_correct_links']==0


def test_no_test_seed_overlap_and_frozen_sizes():
    assert set(SEEDS['005']).isdisjoint(SEEDS['006'])
    assert SCENES_PER_SEED==24
    assert SEEDS['005']==(20266403,20266404)
    assert SEEDS['006']==(20266413,20266414)
