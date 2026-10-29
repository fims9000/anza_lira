"""S91 tests use synthetic photos and simulated reviewer votes, never H50Z GT."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp
from bubble_chamber.s88_generated_benchmark import scene
from bubble_chamber.s89_private_candidate_audit import AuditConfig, run as audit_s89
from bubble_chamber.s90_physicist_review import (
    RESPONSE_SCHEMA, ReviewConfig, build_review_pack, verify_reviews,
)
from bubble_chamber.s91_review_consensus import (
    ConsensusConfig, REPORT_SCHEMA, REVIEW_STATES, analyze_reviews,
    pairwise_agreement,
)


@pytest.fixture(scope='module')
def generated_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('generated_s91')
    bmp = root / 'GENERATED_NOT_H50Z.bmp'
    assert cv2.imwrite(str(bmp), scene(9191, 'blurred_scratches').gray)
    s86, s89, s90 = (root / x for x in ('s86', 's89', 's90'))
    run_full_bmp(bmp, s86, FullBMPConfig(
        method='s88-steerable', core_size=128, halo=32,
        s88_low_snr=.816, s88_high_snr=1.7,
        min_segment_length_px=5, preview_limit=0))
    audit = audit_s89(s86, bmp, s89, AuditConfig(max_tiles=4, preview_tiles=0))
    assert audit['sampled_fragment_count'] >= 5
    pack = build_review_pack(s86, s89, bmp, s90,
                             ReviewConfig(max_cases=6, max_per_tile=4, crop_size=128))
    assert pack['case_count'] >= 3
    return root, bmp, s86, s89, s90


def answer(pack_dir, reviewer, case_ids, choices):
    return {
        'schema': RESPONSE_SCHEMA,
        'review_pack_sha256': sha256_file(pack_dir / 'review_pack.json'),
        'reviewer_id': reviewer,
        'decisions': [
            {'case_id': cid, 'choice': choice, 'note': ''}
            for cid, choice in zip(case_ids, choices)
        ],
    }


@pytest.fixture
def reviewed(generated_inputs, tmp_path):
    _, _, _, s89, pack = generated_inputs
    ids = [x['case_id'] for x in json.loads((pack / 'review_pack.json').read_text())['cases']]
    one = tmp_path / 'expert_1.json'
    two = tmp_path / 'expert_2.json'
    one.write_text(json.dumps(answer(pack, 'physicist_01', ids[:2], ['TRACK_LIKE', 'ARTIFACT_LIKE'])))
    two.write_text(json.dumps(answer(pack, 'physicist_02', ids[:2], ['TRACK_LIKE', 'TRACK_LIKE'])))
    verified_dir = tmp_path / 'verified'
    verify_reviews(pack, [one, two], verified_dir)
    return s89, pack, verified_dir, ids, one, two


def test_s91_full_private_review_to_blinded_followup(reviewed, tmp_path):
    s89, pack, verified, ids, one, two = reviewed
    original_hashes = {
        'pack': sha256_file(pack / 'review_pack.json'),
        'vote': sha256_file(verified / 'case_votes.ndjson'),
        'audit': sha256_file(s89 / 'candidate_audit.ndjson'),
    }
    out = tmp_path / 'S91_PRIVATE'
    report = analyze_reviews(pack, verified, s89, out,
                             ConsensusConfig(max_followup_cases=5))
    assert report['schema'] == REPORT_SCHEMA
    assert report['unique_reviewer_count'] == 2
    assert report['selected_candidate_count'] == len(ids)
    assert report['cases_reviewed_by_at_least_two'] == 2
    assert report['image_appearance_state_counts'][REVIEW_STATES[2]] == 1
    assert report['image_appearance_state_counts'][REVIEW_STATES[1]] == 1
    assert report['reviewer_decision_count'] == 4
    assert report['unresolved_selected_case_count'] == len(ids) - 1
    assert report['followup_queue_selected_count'] == min(5, len(ids) - 1)
    assert report['pairwise_agreement'][0]['overlap_cases'] == 2
    assert report['pairwise_agreement'][0]['matching_votes'] == 1
    assert report['pairwise_agreement'][0]['cohen_kappa'] == 0
    assert report['no_representative_physical_precision_or_recall'] is True
    assert report['native_physics_track_GT_count'] == 0
    assert report['no_training_labels_produced'] is True
    assert report['physical_auto_join_or_reject_authorized'] is False
    queue = json.loads((out / 'followup_queue_private.json').read_text())
    assert ids[1] == queue['cases'][0]['case_id']  # disagreement before insufficient reviews
    assert ids[0] not in {row['case_id'] for row in queue['cases']}
    html = (out / 'followup_blinded/index.html').read_text()
    assert 'PRIVATE S91' in html and ids[1] in html
    assert original_hashes['pack'] in html  # exports SAME original S90 pack SHA
    assert 'review_pack_sha256:packHash' in html
    for secret in ('REVIEW_FIRST', 'REVIEW_LATER_NOT_VERIFIED',
                   'DISAGREEMENT_OR_UNDECIDABLE_REVIEW', 'physicist_01',
                   'physicist_02', 's88_ridge_score_proxy'):
        assert secret not in html
    assert len(list((out / 'followup_blinded/images').glob('*.png'))) == 2 * len(queue['cases'])
    for row in queue['cases']:
        cid = row['case_id']
        for suffix in ('original', 'overlay'):
            path = f'images/{cid}_{suffix}.png'
            assert (out / 'followup_blinded' / path).read_bytes() == (pack / path).read_bytes()
    assert (out / 'followup_blinded').stat().st_mode & 0o077 == 0
    assert (out / 'report.json').stat().st_mode & 0o077 == 0
    assert all(file.stat().st_mode & 0o077 == 0
               for file in (out / 'followup_blinded/images').iterdir())
    assert sha256_file(pack / 'review_pack.json') == original_hashes['pack']
    assert sha256_file(verified / 'case_votes.ndjson') == original_hashes['vote']
    assert sha256_file(s89 / 'candidate_audit.ndjson') == original_hashes['audit']
    with pytest.raises(FileExistsError, match='new empty'):
        analyze_reviews(pack, verified, s89, out)


def test_s91_followup_answer_is_accepted_by_original_s90_verify(reviewed, tmp_path):
    s89, pack, verified, ids, one, two = reviewed
    out = tmp_path / 'analyzed'
    analyze_reviews(pack, verified, s89, out)
    queue = json.loads((out / 'followup_queue_private.json').read_text())['cases']
    new = tmp_path / 'expert_3.json'
    new.write_text(json.dumps(answer(pack, 'physicist_03', [queue[0]['case_id']], ['ARTIFACT_LIKE'])))
    merged_dir = tmp_path / 'verified_with_third'
    merged = verify_reviews(pack, [one, two, new], merged_dir)
    assert merged['unique_reviewer_count'] == 3
    assert merged['native_physics_track_GT_count'] == 0
    second = analyze_reviews(pack, merged_dir, s89, tmp_path / 'analyzed_round_two')
    assert second['reviewer_decision_count'] == 5
    assert second['unique_reviewer_count'] == 3
    assert second['unresolved_selected_case_count'] == len(ids) - 1
    assert second['physical_auto_join_or_reject_authorized'] is False


def test_s91_single_reviewer_no_kappa_and_no_followup_option(generated_inputs, tmp_path):
    _, _, _, s89, pack = generated_inputs
    ids = [x['case_id'] for x in json.loads((pack / 'review_pack.json').read_text())['cases']]
    path = tmp_path / 'solo.json'
    path.write_text(json.dumps(answer(pack, 'solo', ids[:2], ['TRACK_LIKE', 'UNDECIDABLE'])))
    verified = tmp_path / 'single_verified'
    verify_reviews(pack, [path], verified)
    out = tmp_path / 'summary_only'
    report = analyze_reviews(pack, verified, s89, out, ConsensusConfig(build_blinded_followup=False))
    assert report['pairwise_agreement'] == []
    assert report['undecidable_decision_count'] == 1
    assert not (out / 'followup_blinded').exists()
    assert report['followup_pack_available'] is False
    assert all(c['reviewer_decisions'] >= 0 for c in report['selected_candidate_strata'].values())
    with pytest.raises(ValueError, match='max_followup_cases'):
        ConsensusConfig(max_followup_cases=201).validate()


def test_s91_all_agree_kappa_not_fabricated(generated_inputs, tmp_path):
    _, _, _, s89, pack = generated_inputs
    ids = [x['case_id'] for x in json.loads((pack / 'review_pack.json').read_text())['cases']]
    paths = []
    for reviewer in ('a', 'b'):
        f = tmp_path / f'{reviewer}.json'
        f.write_text(json.dumps(answer(pack, reviewer, ids, ['TRACK_LIKE'] * len(ids))))
        paths.append(f)
    verified = tmp_path / 'all_consensus'
    verify_reviews(pack, paths, verified)
    report = analyze_reviews(pack, verified, s89, tmp_path / 'reviewed_all')
    assert report['pairwise_agreement'][0]['cohen_kappa'] is None
    assert report['pairwise_agreement'][0]['observed_agreement'] == 1
    assert report['unresolved_selected_case_count'] == 0
    assert not (tmp_path / 'reviewed_all/followup_blinded').exists()
    assert report['followup_pack_available'] is False
    assert report['image_appearance_state_counts'] == {REVIEW_STATES[2]: len(ids)}


@pytest.mark.parametrize('tamper', ['wrong_vote', 'wrong_summary', 'wrong_audit', 'wrong_preview', 'wrong_priority'])
def test_s91_rejects_tampered_inputs_before_creating_output(reviewed, tmp_path, tamper):
    s89, pack, verified, ids, one, two = reviewed
    s89_copy, pack_copy, verified_copy = [tmp_path / x for x in ('s89_copy', 'pack_copy', 'verified_copy')]
    shutil.copytree(s89, s89_copy)
    shutil.copytree(pack, pack_copy)
    shutil.copytree(verified, verified_copy)
    if tamper == 'wrong_vote':
        p = verified_copy / 'case_votes.ndjson'
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows[0]['native_track_GT'] = True
        p.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    elif tamper == 'wrong_summary':
        p = verified_copy / 'review_summary.json'
        obj = json.loads(p.read_text())
        obj['unique_reviewer_count'] = 100
        p.write_text(json.dumps(obj))
    elif tamper == 'wrong_audit':
        with (s89_copy / 'candidate_audit.ndjson').open('a') as handle:
            handle.write('{}\n')
    elif tamper == 'wrong_preview':
        p = pack_copy / f'images/{ids[0]}_original.png'
        p.write_bytes(b'FAKE_PRIVATE_BYTES')
    elif tamper == 'wrong_priority':
        p = s89_copy / 'report.json'
        obj = json.loads(p.read_text())
        obj['physics_track_GT_count'] = 300
        p.write_text(json.dumps(obj))
    target = tmp_path / 'NEVER_CREATED'
    with pytest.raises((ValueError, OSError)):
        analyze_reviews(pack_copy, verified_copy, s89_copy, target)
    assert not target.exists()


def test_s91_destination_inside_repo_is_blocked(reviewed):
    s89, pack, verified, *_ = reviewed
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='inside public source'):
        analyze_reviews(pack, verified, s89, repo / 'PRIVATE_NOT_ALLOWED')


def test_s91_pairwise_excludes_nonoverlap_and_not_measured_from_one_vote():
    cases = {
        'c_aa': {'votes': [{'reviewer_id': 'a', 'choice': 'TRACK_LIKE'}]},
        'c_bb': {'votes': [{'reviewer_id': 'b', 'choice': 'ARTIFACT_LIKE'}]},
    }
    pair = pairwise_agreement(cases)
    assert pair[0]['overlap_cases'] == 0
    assert pair[0]['observed_agreement'] is None
    assert pair[0]['cohen_kappa'] is None
