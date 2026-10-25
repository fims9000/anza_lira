from pathlib import Path
import json,pytest
from bubble_chamber.s32_relation_bridge import compile_local_relation_reviews
from tests.test_s31_contracts import fixture,corrections


def test_empty_real_local_review_emits_zero_pairs():
 f=Path('/mnt/data/s32/annotations/S31_MACHINE_DRAFTS.json')
 if not f.is_file():pytest.skip('private S31 absent')
 pack=json.loads(f.read_text())
 result=compile_local_relation_reviews(pack,None)
 assert result['total_candidate_questions']==28
 assert not result['reviewed_rows']
 assert result['expert_certified_relation_count']==0
 assert result['eligible_for_physically_automatic_joins'] is False
 assert result['unknown_candidates_not_used_as_negatives']


def test_forged_local_pairs_cannot_be_marked_human_truth(tmp_path):
 folder,pack,photos=fixture(tmp_path)
 c=corrections(pack)
 c['relation_decisions']['rel_001']={'status':'LOCAL_SAME_PROVISIONAL'}
 r=compile_local_relation_reviews(pack,c)
 assert len(r['reviewed_rows'])==1
 assert r['locally_reviewed_same']==1
 assert r['eligible_for_provisional_graph_training'] is False
 assert r['reviewed_rows'][0]['expert_certified'] is False
 assert r['reviewed_rows'][0]['physics_join_allowed'] is False
 c['relation_decisions']['rel_001']={'status':'LOCAL_UNSURE'}
 assert compile_local_relation_reviews(pack,c)['reviewed_rows']==[]
 c['relation_decisions']['rel_001']={'status':'LOCAL_SAME_PROVISIONAL'}
 c['decisions']['draft_000']={'status':'USER_REJECTED_PROVISIONAL'}
 with pytest.raises(ValueError,match='rejected'):
  compile_local_relation_reviews(pack,c)
