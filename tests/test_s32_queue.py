import json,hashlib
from pathlib import Path
import pytest
from bubble_chamber.s32_review_queue import build_review_queue,inject_review_queue


def test_private_real_pack_queue_has_no_confirmed_truth():
 base=Path('/mnt/data/s32')
 f=base/'annotations/S31_MACHINE_DRAFTS.json'
 if not f.is_file():pytest.skip('private S31 dataset unavailable')
 pack=json.loads(f.read_text())
 q=build_review_queue(pack,hashlib.sha256(f.read_bytes()).hexdigest())
 assert q['original_photos']==39 and q['machine_drafts']==133 and q['possible_pairs']==28
 assert q['expert_certified_lines']==0 and q['expert_certified_pairs']==0
 assert all(x['machine_label']=='UNKNOWN' for x in q['candidate_questions'])
 assert set(x['photo_id'] for x in q['candidate_questions'])=={'005','006'}
 html=(base/'review/S31_MACHINE_FIRST_REVIEWER.html').read_text()
 revised=inject_review_queue(html,pack,q)
 assert revised.count('id="s31-draft-payload"')==1
 assert revised.count('id="s32-review-priority"')==1
 assert revised.count('window.s31ExportCorrections')==1
 assert 's31_machine_corrections_v1' in revised
 with pytest.raises(ValueError,match='Already enhanced'):
  inject_review_queue(revised,pack,q)


def test_queue_source_digest_enforced():
 base=Path('/mnt/data/s32')
 f=base/'annotations/S31_MACHINE_DRAFTS.json'
 if not f.is_file():pytest.skip('private S31 dataset unavailable')
 pack=json.loads(f.read_text())
 q=build_review_queue(pack,'0'*64)
 html=(base/'review/S31_MACHINE_FIRST_REVIEWER.html').read_text()
 q['source_S29_sha256']='0'*64
 with pytest.raises(ValueError,match='provenance'):
  inject_review_queue(html,pack,q)


def test_synthetic_s31_pack_review_queue_without_private_files(tmp_path):
 from tests.test_s31_contracts import fixture
 _,pack,_=fixture(tmp_path)
 q=build_review_queue(pack,'e'*64)
 assert q['machine_drafts']==40 and q['possible_pairs']==1
 assert q['expert_certified_lines']==0 and q['expert_certified_pairs']==0
 assert all(x['machine_label']=='UNKNOWN' for x in q['candidate_questions'])
 assert q['relation_questions'][0]['human_certified'] is False
