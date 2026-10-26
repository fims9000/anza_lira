"""S72 public snapshot and CAL-first, no native physics claim."""
import json
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
TEST=ROOT/'docs/bubble_chamber/S72_PUBLIC_FRESH_TEST_2026_10_26.json'


def test_immutable_cal_provenance_and_source_split():
    a=json.loads(CAL.read_text())
    b=json.loads(TEST.read_text())
    assert b['frozen_CAL_sha256']==sha256(CAL.read_bytes()).hexdigest()
    assert a['experiment_id']==b['experiment_id']=='ANZA-LIRA-BC-S72-20261026'
    assert b['source_frame_split_sha256']=='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef'
    assert a['phase']=='cal' and b['phase']=='test'
    assert not a['physical_auto_join_authorized'] and not b['physical_auto_join_authorized']
    assert b['physical_verified_H50Z_GT']==0


def test_fresh_test_observed_and_frozen_gate():
    j=json.loads(TEST.read_text())
    arms=j['all_TEST_arms']
    assert len(j['directions'])==2
    assert j['directions']['005_to_006']['CAL_selected']=='s68_branch_endpoint_refine'
    assert j['directions']['006_to_005']['CAL_selected']=='s39_original'
    assert sum(len(d['TEST_seeds']) for d in j['directions'].values())==4
    assert sum(x['scene_count'] for d in j['directions'].values() for x in d['TEST_seeds'].values() for k,x in x['arms'].items() if k=='s39_original')==96
    assert [(arms[k]['selected_synthetic_correct_links'],arms[k]['synthetic_noncorrect_selected'],
             arms[k]['complete_three_piece_paths_synthetic']) for k in arms] == [
        (56,52,12),(49,45,10),(52,45,12),(53,35,13)]
    correct=incorrect=paths=0
    for d in j['directions'].values():
        selected=d['CAL_selected']
        for row in d['TEST_seeds'].values():
            vals=row['arms'][selected]
            correct+=vals['selected_synthetic_correct_links']
            incorrect+=vals['synthetic_noncorrect_selected']
            paths+=vals['complete_three_piece_paths_synthetic']
    assert (correct,incorrect,paths)==(56,41,13)


def test_public_s72_does_not_embed_original_photos_or_coordinates():
    for path in (CAL,TEST):
        t=path.read_text()
        for bad in ('/mnt/data/','file_0000','data:image','base64,','_PRIVATE_source_frames',
                    '.png"','pixel_array','original_photograph'):
            assert bad not in t
    registry=json.loads((ROOT/'docs/bubble_chamber/EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    ids=[r['id'] for r in registry['experiments']]
    assert ids.count('ANZA-LIRA-BC-S72-20261026')==1
