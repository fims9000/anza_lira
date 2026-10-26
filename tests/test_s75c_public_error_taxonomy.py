"""S75C all synthetic error categories add up and historic S75 remains intact."""
from hashlib import sha256
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'


def test_original_unmodified_test_replayed_and_categories_exhaustive():
    a=json.loads((DOC/'S75C_POSTHOC_ERROR_TAXONOMY_PUBLIC_2026_10_26.json').read_text())
    original=DOC/'S75_PUBLIC_FRESH_TEST_2026_10_26.json'
    assert a['S75_TEST_sha256']==sha256(original.read_bytes()).hexdigest()
    assert a['analysis_status']=='retrospective_only_NOT_NEW_TEST'
    assert a['new_training_or_threshold_selection'] is False
    assert a['native_particle_auto_joins_authorized'] is False
    assert a['native_physics_GT_count']==0
    old=json.loads(original.read_text())
    for condition,methods in a['aggregate_96_scenes'].items():
        for arm,result in methods.items():
            assert (result['selected_synthetic_wrong_identified']+
                result['selected_synthetic_unmatched_or_ambiguous']+
                result['selected_synthetic_wrong_endpoint'])==result['synthetic_noncorrect_selected']
            assert result['selected_synthetic_correct_links']==old['overall_conditions'][condition][arm]['selected_synthetic_correct_links']
            assert result['synthetic_noncorrect_selected']==old['overall_conditions'][condition][arm]['synthetic_noncorrect_selected']


def test_registry_has_strict_new_diagnostic_id_and_content_hashes():
    r=json.loads((DOC/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    s=[x for x in r['experiments'] if x['id']=='ANZA-LIRA-BC-S75C-20261026']
    assert len(s)==1
    v=s[0]
    for k in ('predictor','report','results'):
        p=ROOT/v[k]
        assert p.is_file() and sha256(p.read_bytes()).hexdigest()==v[k+'_sha256']
    assert not v['new_independent_scenes']
    assert v['native_physics_GT_count']==0
