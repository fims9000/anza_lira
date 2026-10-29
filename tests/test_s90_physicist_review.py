"""S90 exercises GENERATED images only; never real H50Z pixels/identities."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp
from bubble_chamber.s89_private_candidate_audit import AuditConfig, run as audit_s89
from bubble_chamber.s90_physicist_review import (
    PACK_SCHEMA, RESPONSE_SCHEMA, ReviewConfig,
    build_review_pack, select_candidate_rows, verify_reviews,
)
from bubble_chamber.s88_generated_benchmark import scene


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp('s90_generated_fixture')
    fake_bmp = root / 'GENERATED_NOT_H50Z.bmp'
    synthetic = scene(9090, 'blurred_scratches')
    assert cv2.imwrite(str(fake_bmp), synthetic.gray)
    s86, s89 = root / 'S86_PRIVATE', root / 'S89_PRIVATE'
    base = FullBMPConfig(method='s88-steerable', core_size=128, halo=32,
                         s88_low_snr=.816, s88_high_snr=1.7,
                         min_segment_length_px=5, preview_limit=0)
    prior = run_full_bmp(fake_bmp, s86, base)
    assert prior['candidate_fragments'] > 2
    report = audit_s89(s86, fake_bmp, s89, AuditConfig(max_tiles=4, preview_tiles=0))
    assert report['sampled_fragment_count'] > 2
    return root, fake_bmp, s86, s89


@pytest.fixture(scope='module')
def pack(source):
    root, bmp, s86, s89 = source
    out = root / 'S90_PRIVATE'
    output = build_review_pack(s86, s89, bmp, out,
                               ReviewConfig(max_cases=8, max_per_tile=3, crop_size=128))
    return out, output


def _response(pack_dir, reviewer, cases, choices):
    return {'schema': RESPONSE_SCHEMA, 'review_pack_sha256': sha256_file(pack_dir/'review_pack.json'),
            'reviewer_id': reviewer,
            'decisions': [{'case_id': cid, 'choice': ch, 'note': ''}
                          for cid, ch in zip(cases, choices)]}


def test_s90_source_locked_private_pack_blinded_and_resumeless(source, pack, tmp_path):
    _, bmp, s86, s89 = source
    review_dir, result = pack
    manifest_sha = sha256_file(s86/'manifest.json')
    audit_sha = sha256_file(s89/'candidate_audit.ndjson')
    mask_hashes = {p.name: sha256_file(p) for p in (s86/'masks').glob('*.png')}
    saved = json.loads((review_dir/'review_pack.json').read_text())
    assert result['schema'] == saved['schema'] == PACK_SCHEMA
    assert result['review_pack_sha256'] == sha256_file(review_dir/'review_pack.json')
    assert 1 <= result['case_count'] <= 8
    assert saved['source_S86_manifest_sha256'] == manifest_sha
    assert saved['source_S89_audit_sha256'] == audit_sha
    assert saved['source_BMP_sha256'] == sha256_file(bmp)
    assert saved['native_track_GT_count'] == 0
    assert saved['physical_auto_joins_authorized'] is False
    assert saved['S89_review_flags_hidden_from_review_UI'] is True
    assert saved['does_not_change_detector_masks'] is True
    assert saved['case_count'] == len(saved['cases'])
    assert len({x['case_id'] for x in saved['cases']}) == len(saved['cases'])
    assert len(list((review_dir/'images').glob('*_original.png'))) == saved['case_count']
    assert len(list((review_dir/'images').glob('*_overlay.png'))) == saved['case_count']
    page = (review_dir/'index.html').read_text()
    assert 'TRACK_LIKE' in page and 'ARTIFACT_LIKE' in page and 'UNDECIDABLE' in page
    assert 'URL.createObjectURL' in page and 'review_pack_sha256' in page
    assert 'fetch(' not in page and 'localStorage' not in page
    assert 'REVIEW_FIRST' not in page and 'LOW_TRANSVERSE_DARKNESS_REVIEW' not in page
    assert 's88_ridge_score_proxy' not in page
    assert (review_dir/'index.html').stat().st_mode & 0o077 == 0
    assert (review_dir/'images').stat().st_mode & 0o077 == 0
    assert all(x.stat().st_mode & 0o077 == 0 for x in (review_dir/'images').iterdir())
    assert all('images/' in c['original_preview'] for c in saved['cases'])
    assert sha256_file(s86/'manifest.json') == manifest_sha
    assert sha256_file(s89/'candidate_audit.ndjson') == audit_sha
    assert mask_hashes == {p.name: sha256_file(p) for p in (s86/'masks').glob('*.png')}
    with pytest.raises(FileExistsError, match='new empty'):
        build_review_pack(s86, s89, bmp, review_dir)
    assert (review_dir/'review_pack.json').is_file()


def test_s90_visuals_are_original_resolution_with_coordinate_provenance(source, pack):
    _, bmp, _, _ = source
    review_dir, _ = pack
    img = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    manifest = json.loads((review_dir/'review_pack.json').read_text())
    for case in manifest['cases']:
        x0,y0,x1,y1 = case['crop_xyxy_source_px']
        original = cv2.imread(str(review_dir/case['original_preview']),cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(str(review_dir/case['overlay_preview']),cv2.IMREAD_COLOR)
        assert np.array_equal(original, img[y0:y1,x0:x1])
        assert overlay.shape[:2] == original.shape
        assert original.ndim == 2 and original.dtype == np.uint8
        assert sha256_file(review_dir/case['original_preview']) == case['original_preview_sha256']
        assert sha256_file(review_dir/case['overlay_preview']) == case['overlay_preview_sha256']


def test_s90_two_independent_reviews_and_disagreements_never_gt(pack, tmp_path):
    review_dir, _ = pack
    ids = [x['case_id'] for x in json.loads((review_dir/'review_pack.json').read_text())['cases']]
    assert len(ids) >= 2
    r1 = _response(review_dir,'expert_01',ids[:2],['TRACK_LIKE','ARTIFACT_LIKE'])
    r2 = _response(review_dir,'expert_02',ids[:2],['TRACK_LIKE','TRACK_LIKE'])
    p1 = tmp_path/'answer_1.json';p1.write_text(json.dumps(r1))
    p2 = tmp_path/'answer_2.json';p2.write_text(json.dumps(r2))
    out = tmp_path/'VERIFIED_PRIVATE'
    result = verify_reviews(review_dir,[p1,p2],out)
    assert result['unique_reviewer_count']==2
    assert result['cases_reviewed_by_at_least_two']==2
    assert result['review_state_counts']['IMAGE_APPEARANCE_TRACK_LIKE_AGREEMENT_NOT_GT']==1
    assert result['review_state_counts']['DISAGREEMENT_OR_UNDECIDABLE_REVIEW']==1
    assert result['native_physics_track_GT_count']==0
    assert result['physical_track_accuracy_measured'] is False
    assert result['no_training_labels_or_detector_mask_modifications']
    votes=[json.loads(x) for x in (out/'case_votes.ndjson').read_text().splitlines()]
    assert len(votes)==len(ids)
    assert votes[0]['source_fragment_id']
    assert not any(x['native_track_GT'] or x['physical_auto_accept'] for x in votes)
    assert (out/'review_summary.json').stat().st_mode&0o077==0
    with pytest.raises(FileExistsError,match='new empty'):
        verify_reviews(review_dir,[p1,p2],out)


@pytest.mark.parametrize('bad_change',[
    lambda x: x.update(review_pack_sha256='0'*64),
    lambda x: x.update(reviewer_id='person@example.com'),
    lambda x: x['decisions'][0].update(case_id='c_'+'0'*20),
    lambda x: x['decisions'][0].update(choice='TRUE_PHYSICAL_GT'),
    lambda x: x['decisions'][0].update(note='X'*301),
    lambda x: x['decisions'].append(x['decisions'][0].copy()),
])
def test_s90_rejects_spoofed_response(pack,tmp_path,bad_change):
    review_dir,_=pack
    cid=json.loads((review_dir/'review_pack.json').read_text())['cases'][0]['case_id']
    reply=_response(review_dir,'reviewer1',[cid],['TRACK_LIKE'])
    bad_change(reply)
    p=tmp_path/'tampered_response.json';p.write_text(json.dumps(reply))
    with pytest.raises(ValueError):
        verify_reviews(review_dir,[p],tmp_path/'NEVER_WRITTEN')
    assert not (tmp_path/'NEVER_WRITTEN').exists()


def test_s90_rejects_duplicate_reviewer_and_tampered_preview(pack,tmp_path):
    review_dir,_=pack
    cid=json.loads((review_dir/'review_pack.json').read_text())['cases'][0]['case_id']
    p1=tmp_path/'a.json';p2=tmp_path/'b.json'
    reply=_response(review_dir,'same_person',[cid],['TRACK_LIKE'])
    p1.write_text(json.dumps(reply));p2.write_text(json.dumps(reply))
    with pytest.raises(ValueError, match='duplicate reviewer_id'):
        verify_reviews(review_dir,[p1,p2],tmp_path/'BAD')
    assert not (tmp_path/'BAD').exists()
    copied=tmp_path/'COPY_REVIEW_PACK'
    shutil.copytree(review_dir,copied)
    first=json.loads((copied/'review_pack.json').read_text())['cases'][0]
    altered=copied/first['original_preview']
    altered.write_bytes(b'TAMPERED')
    with pytest.raises(ValueError,match='modified or missing'):
        verify_reviews(copied,[p1],tmp_path/'BAD2')


def test_s90_rejects_wrong_source_and_public_destination(source,pack,tmp_path):
    _,bmp,s86,s89=source
    wrong=tmp_path/'WRONG.bmp';img=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
    img[5,5]^=30
    assert cv2.imwrite(str(wrong),img)
    with pytest.raises(ValueError,match='SHA256'):
        build_review_pack(s86,s89,wrong,tmp_path/'BAD')
    assert not (tmp_path/'BAD').exists()
    import bubble_chamber.s90_physicist_review as module
    repo=Path(module.__file__).resolve().parents[1]
    with pytest.raises(ValueError,match='inside public source'):
        build_review_pack(s86,s89,bmp,repo/'BAD')
    with pytest.raises(ValueError,match='overwrite or contain'):
        build_review_pack(s86,s89,bmp,s89/'BAD')


def test_s90_balancing_and_blinding_never_use_true_labels():
    rows = [{'fragment_id':f't{i % 2}_f{i}', 'tile_id':f't{i % 2}',
             'status':'UNKNOWN','review_priority':'REVIEW_FIRST' if i<6 else 'REVIEW_LATER_NOT_VERIFIED'}
            for i in range(12)]
    cfg=ReviewConfig(max_cases=6,max_per_tile=3,crop_size=128)
    chosen=select_candidate_rows(rows,'a'*64,cfg)
    assert len(chosen)==6
    assert chosen==select_candidate_rows(rows,'a'*64,cfg)
    assert len({x['fragment_id'] for x in chosen})==6
    assert {x['review_priority'] for x in chosen}=={'REVIEW_FIRST','REVIEW_LATER_NOT_VERIFIED'}
    assert all(sum(x['tile_id']==tid for x in chosen)<=3 for tid in ('t0','t1'))
    with pytest.raises(ValueError,match='duplicate'):
        select_candidate_rows([rows[0],rows[0]],'a'*64,cfg)
    with pytest.raises(ValueError,match='crop_size'):
        ReviewConfig(crop_size=91).validate()
    with pytest.raises(ValueError,match='max_cases'):
        ReviewConfig(max_cases=0).validate()
