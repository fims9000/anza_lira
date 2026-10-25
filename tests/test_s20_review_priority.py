import pytest
from bubble_chamber.s19_real_review import load_package,empty_review,compile_review,make_offline_review_html
from bubble_chamber.s20_review_priority import rank_review_questions,integrate_priority_ui
from tests.test_s19_real_review import fixture


def test_candidate_questions_are_unknown_and_photo_diverse(tmp_path):
    source,folder=fixture(tmp_path)
    pack=load_package(source,folder)
    q=rank_review_questions(pack,limit_candidates=6)
    assert q['source_package_sha256']==pack['package_sha256']
    assert q['all_candidates']==6
    assert len(q['candidate_questions'])==6
    assert len(set(x['photo_id'] for x in q['candidate_questions'][:4]))==2
    assert len(set(x['candidate_id'] for x in q['candidate_questions']))==6
    assert q['expert_approved_connections']==0
    assert q['expert_approved_lines']==0
    assert all(x['expert_label']=='UNKNOWN' for x in q['candidate_questions'])
    assert all(x['needs_human_pair_label'] is True for x in q['pair_questions'])
    assert q['calibrated_probability'] is False
    confirmed=compile_review(pack,empty_review(pack))
    assert confirmed['confirmed_positive_pixels']==0
    assert confirmed['confirmed_negative_pixels']==0
    assert confirmed['reviewed_relation_labels']==0
    assert confirmed['trainable_real_image_baseline'] is False


def test_prioritized_review_page_keeps_same_export_json(tmp_path):
    source,folder=fixture(tmp_path)
    pack=load_package(source,folder)
    page=make_offline_review_html(pack,folder)
    q=rank_review_questions(pack)
    new=integrate_priority_ui(page,q)
    assert '<script id="payload"' in new and '<script id="s20-priority-payload"' in new
    assert 's20-next-line' in new and 's20-next-pair' in new
    assert 'function exported()' in new
    assert pack['package_sha256'] in new
    assert 'Object.prototype.hasOwnProperty.call' in new
    assert new.count('</body></html>')==1
    assert 'fetch(' not in new and 'https://' not in new


def test_invalid_review_package_or_html_rejected(tmp_path):
    source,folder=fixture(tmp_path)
    pack=load_package(source,folder)
    q=rank_review_questions(pack)
    with pytest.raises(ValueError):rank_review_questions({'schema':'invalid','tiles':[]})
    with pytest.raises(ValueError):rank_review_questions(pack,limit_candidates=0)
    with pytest.raises(ValueError):integrate_priority_ui('bad html',q)
    with pytest.raises(ValueError):integrate_priority_ui(make_offline_review_html(pack,folder),{'schema':'bad'})
