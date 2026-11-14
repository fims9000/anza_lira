"""V51 source-block OOF, observed-only path gate and decoded-path risk tests."""
import numpy as np
import pytest
import scripts.tver_v51_path_accept_gate as gate


def test_contiguous_source_block_oof_is_disjoint():
    for fold in range(5):
        train={i for i in range(100) if gate.source_block(i)!=fold}
        held={i for i in range(100) if gate.source_block(i)==fold}
        assert len(train)==80 and len(held)==20 and not(train&held)
        assert sorted(held)==list(range(20*fold,20*(fold+1)))
    for invalid in (-1,100,999):
        with pytest.raises(ValueError):
            gate.source_block(invalid)


def test_path_features_do_not_use_hidden_line_or_label(monkeypatch):
    def decoded(net,scene,proposal,mode="learned_ribbon"):
        return [(48,x) for x in range(35,61)],{"anchor_status":"predicted_only"}
    monkeypatch.setattr(gate,"decode_candidate",decoded)
    monkeypatch.setattr(gate,"path_heatmap",
                        lambda net,scene,proposal:np.ones((96,96),np.float32)*.8)
    scene={"rectangle":(44,53,42,54)}
    proposal={"source":(48.,30.),"target":(48.,62.)}
    x,path,meta=gate.path_only_features(None,scene,proposal)
    assert x.shape==(6,) and np.isfinite(x).all()
    assert x[0]==pytest.approx(.8)
    assert meta["anchor_status"]=="predicted_only"
    scene["hidden_centerline"]=[(1,1)]
    scene["true_ends"]=((2,2),(3,3))
    proposal["y"]=0
    second,_,_=gate.path_only_features(None,scene,proposal)
    assert np.array_equal(x,second)


def cases(perfect=True):
    out=[]
    for section in range(60):
        for gap_index in range(2):
            idx=section*2+gap_index
            for world in ("LINK","NO_LINK"):
                correct=int(world=="LINK" and idx<45)
                good_path=int(correct and idx<10)
                score=(.95 if good_path else .20) if perfect else .8
                out.append({
                    "method":"path_OOF_gate",
                    "section":section,"gap_id":f"{section}:{gap_index}",
                    "world":world,
                    "candidate_available":int(world=="LINK" and idx<75),
                    "candidate_count":2,
                    "correct_top1_pair":correct,
                    "correct_decoded_path":good_path,
                    "accept_score":score,
                    "path_anchor_status":"observed_only",
                })
    return out


def test_path_success_gate_can_accept_only_successful_lines():
    result=gate.audit_path_gate(cases(perfect=True))
    assert result["correct_top1_pairs"]==45
    assert result["correct_hidden_paths_from_top1"]==10
    assert result["best_accepted_correct_paths_conditional_CP95_10pct"]==10
    assert result["best_accepted_correct_paths_both_empirical_errors_le10pct_posthoc"]==10
    assert result["risk_error_wrong_LINK_includes_wrong_pair_OR_wrong_decoded_path"]
    assert len(result["predeclared_threshold_sweep"])==11


def test_wrong_path_is_counted_as_wrong_even_when_pair_is_correct():
    result=gate.audit_path_gate(cases(perfect=False))
    assert result["correct_top1_pairs"]==45
    assert result["correct_hidden_paths_from_top1"]==10
    assert result["best_accepted_correct_paths_conditional_CP95_10pct"]==0
    with pytest.raises(RuntimeError,match="Original real"):
        gate.audit_path_gate(cases()[:-1])
    bad=cases()
    bad[0]["candidate_available"]=0
    with pytest.raises(RuntimeError,match="fixed source"):
        gate.audit_path_gate(bad)
