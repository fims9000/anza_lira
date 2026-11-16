"""Small V53 orchestration contract tests without original Thebe arrays."""
import json
import csv
import types
import sys
from pathlib import Path


def test_v53_gate_receipts_created_and_legacy_counts_locked(tmp_path,monkeypatch):
    import scripts.tver_v53_real_gate as m
    classes=types.ModuleType('scripts.tver_v53_path_accept_gate')
    classes.fit_path_success_gate=lambda *args: (object(),{"positive_successful_OOF_paths":20})
    classes.score_source_path_worlds=lambda *_: [{"world":"LINK","correct_decoded_path":1}]
    classes.audit_path_gate=lambda rows: {
        "correct_hidden_paths_from_top1":33,
        "best_accepted_correct_paths_conditional_CP95_10pct":4,
        "best_accepted_correct_paths_at_zero_observed_wrong_sections_posthoc":5,
        "best_accepted_correct_paths_both_empirical_errors_le10pct_posthoc":7,
    }
    monkeypatch.setitem(sys.modules,'scripts.tver_v53_path_accept_gate',classes)
    v52=types.ModuleType('scripts.tver_v52_thebe_comparison')
    v52.evaluate_paired_cal=lambda *args:{"n_v52_correct_paths":33,"n_original_v50_correct_paths":18}
    monkeypatch.setitem(sys.modules,'scripts.tver_v52_thebe_comparison',v52)
    result=m.run_v53_comparison([],None,None,None,{},tmp_path,augmented_train=[])
    assert result['source_CI_conditional_CAL_audit']['best_accepted_correct_paths_conditional_CP95_10pct']==4
    saved=json.loads((tmp_path/'V53_PATH_GATE_RESULT.json').read_text())
    assert saved['raw_curvature_paths_reproduced']==33
    assert len(list(csv.DictReader((tmp_path/'V53_PATH_GATE_CASES.csv').open())))==1