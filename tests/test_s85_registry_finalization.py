"""S85 is registered as source-ready, never falsely as scientific-ready."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/bubble_chamber'

def test_S85_provenance_has_exact_source_sha_and_no_fake_scientific_result():
    ledger=json.loads((DOCS/'S85_SOURCE_ONLY_RELEASE_VALIDATION_2026_10_27.json').read_text())
    registry=json.loads((DOCS/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    entry=[x for x in registry['experiments'] if x.get('id')=='ANZA-LIRA-BC-S85-20261027']
    assert len(entry)==1
    rec=entry[0]
    assert rec['private_TRAIN_complete'] is False and rec['private_CAL_complete'] is False and rec['private_TEST_complete'] is False
    assert rec['native_GT_count']==0 and rec['physical_auto_join_authorized'] is False
    assert ledger['native_physics_GT_count']==0 and ledger['synthetic_smoke_is_NOT_H50Z']
    assert 'SCIENTIFIC_TRAIN_CAL_TEST_NOT_RUN' in ledger['phase']
    assert ledger['source_frozen_S72_S80_S84_locks_valid']
    for name,relative in ledger['source_paths'].items():
        h=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        assert rec[name+'_sha256']==ledger['source_sha256'][name]==h
    assert rec['validation_ledger_sha256']==hashlib.sha256((ROOT/rec['validation_ledger']).read_bytes()).hexdigest()
