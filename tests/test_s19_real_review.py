"""Real-source sparse annotation contracts: simulated reviewer *fixtures* only.

No fixture decision is claimed to have come from a bubble-chamber physicist.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber import s19_real_review as s19
from scripts.research.bubble_chamber.s14_train_sparse_anza import read_reviewed_manifest


def fixture(tmp_path):
    source=tmp_path/'tiles';source.mkdir()
    records=[];proposals=[]
    for i in range(2):
        fname=f'H50Z_test_{i}_start0_tile.png'
        im=np.full((64,64),195,np.uint8)
        cv2.line(im,(4,16),(57,16),20,1)
        cv2.line(im,(4,40),(57,40),30,1)
        assert cv2.imwrite(str(source/fname),im)
        dig=sha256(im.tobytes()).hexdigest()
        candidates=[]
        for k,(a,b) in enumerate([([[4,16],[10,16],[16,16],[22,16]],0),
                                   ([[31,16],[37,16],[43,16],[49,16]],1),
                                   ([[4,40],[10,40],[16,40],[22,40]],2)]):
            candidates.append({'id':f'cid:{i}:{k}', 'centerline_local_xy_px':a,
                               'centerline_source_xy_endpoints':[a[0],a[-1]],
                               'length_px':18.0,'origin':'frangi_not_truth'})
        src={'label':'start','source_index':0, 'parent':f'H50Z_030_01_{5+i:03d}.bmp',
             'image_sha256':'e'*64,'roi_original_xyxy':[0,0,64,64],
             'tile_sha256':dig,'center_tile_xy':[32,32]}
        p={'schema':s19.OLD_SCHEMA,'source':src,'tile_shape_yx':[64,64],
           'pixels_sha256':dig,'candidate_method':'frangi_p2_fixed_fraction_0.16',
           'candidates':candidates,'label_semantics':'expert only','unmarked_pixels':'UNKNOWN','automatic_labels_count':0}
        p['proposal_digest']=s19._sha(p)
        proposals.append(p)
        records.append({'parent':src['parent'],'crop_file':fname,
                        'crop_gray_pixels_sha256':dig,'proposal_digest':p['proposal_digest'],
                        'source_kind':'test image not real'})
    path=tmp_path/'proposals.json'
    path.write_text(json.dumps({'schema':'s15_real_photo_sparse_proposals_v1','proposals':proposals,'records':records}))
    return path,source


def test_load_real_source_validates_provenance_and_empty_all_unknown(tmp_path):
    p,root=fixture(tmp_path)
    pack=s19.load_package(p,root)
    assert len(pack['tiles'])==2
    empty=s19.empty_review(pack)
    result=s19.compile_review(pack,empty,out_dir=tmp_path/'compiled',image_root=root)
    assert result['confirmed_positive_pixels']==0
    assert result['confirmed_negative_pixels']==0
    assert result['reviewed_relation_labels']==0
    assert not result['trainable_real_image_baseline']
    manifest=json.loads((tmp_path/'compiled/s14_sparse_image_reviews.json').read_text())
    assert all(x['split']=='unlabeled' and x['mask'] is None for x in manifest['entries'])
    assert not list((tmp_path/'compiled').glob('*expert_sparse_mask.png'))
    assert all(x['unknown_pixels']==64*64 for x in result['per_tile'])
    assert result['auto_certified_track_pixels']==0


def test_sparse_manual_and_candidates_compiled_to_0_1_255_and_graph_pair(tmp_path):
    p,root=fixture(tmp_path)
    pack=s19.load_package(p,root)
    review=s19.empty_review(pack)
    for i,tile in enumerate(pack['tiles']):
        entry=review['entries'][i]
        cand=tile['candidates']
        entry['decisions']=[{'candidate_id':cand[0]['id'],'label':'track_centerline'},
                            {'candidate_id':cand[1]['id'],'label':'track_centerline'},
                            {'candidate_id':cand[2]['id'],'label':'track_centerline'}]
        # Expert manually marks an artifact away from the candidate track.
        entry['strokes']=[{'id':f'manual:{tile["tile_id"]}:0','label':'certified_artifact',
                           'radius_px':2,'points_xy':[[8,51],[9,51],[10,51]]},
                          {'id':f'manual:{tile["tile_id"]}:1','label':'track_centerline',
                           'radius_px':1,'points_xy':[[49,30],[52,30],[55,30]]}]
        entry['links']=[{'a':cand[0]['id'],'b':cand[1]['id'],'decision':'same_track'},
                        {'a':cand[0]['id'],'b':cand[2]['id'],'decision':'different_track'},
                        {'a':cand[2]['id'],'b':entry['strokes'][1]['id'],'decision':'uncertain'}]
    out=tmp_path/'compiled';r=s19.compile_review(pack,review,out_dir=out,image_root=root)
    assert r['trainable_real_image_baseline'] is True
    assert r['confirmed_positive_pixels']>0 and r['confirmed_negative_pixels']>0
    assert r['reviewed_relation_labels']==4
    manifest=json.loads((out/'s14_sparse_image_reviews.json').read_text())
    assert {e['split'] for e in manifest['entries']}=={'train','val'}
    masks=[cv2.imread(str(out/e['mask']),cv2.IMREAD_UNCHANGED) for e in manifest['entries']]
    assert all(set(np.unique(m)).issubset({0,1,255}) and m[0,0]==255 for m in masks)
    loaded,summary=read_reviewed_manifest(out/'s14_sparse_image_reviews.json')
    assert len(loaded['train'])==len(loaded['val'])==1
    assert summary['source_run_count']==1  # not an independent physical run
    pairs=json.loads((out/'s19_pair_labels.json').read_text())['rows']
    assert {q['expert_label'] for q in pairs}=={'same_track','different_track'}
    assert all(q['verified_human'] and q['orientation']=='minimum_endpoint_distance_unverified' for q in pairs)


def test_sparse_single_class_tile_is_kept_when_global_class_balance_sufficient(tmp_path):
    p,root=fixture(tmp_path);pack=s19.load_package(p,root)
    review=s19.empty_review(pack)
    for i,tile in enumerate(pack['tiles']):
        review['entries'][i]['strokes']=[{'id':f'manual:{tile["tile_id"]}:0','label':'track_centerline',
            'radius_px':2,'points_xy':[[5,10],[12,10]]}]
    result=s19.compile_review(pack,review,out_dir=tmp_path/'one_class',image_root=root)
    assert result['trainable_real_image_baseline'] is False
    manifest=json.loads((tmp_path/'one_class/s14_sparse_image_reviews.json').read_text())
    assert {x['split'] for x in manifest['entries']}=={'train','val'}
    with pytest.raises(ValueError,match='both expert foreground and certified artifacts'):
        read_reviewed_manifest(tmp_path/'one_class/s14_sparse_image_reviews.json')


def test_conflict_does_not_silently_assign_foreground(tmp_path):
    p,root=fixture(tmp_path);pack=s19.load_package(p,root)
    r=s19.empty_review(pack);a=r['entries'][0]
    a['decisions']=[{'candidate_id':pack['tiles'][0]['candidates'][0]['id'], 'label':'track_centerline'}]
    a['strokes']=[{'id':f'manual:{pack["tiles"][0]["tile_id"]}:1','label':'certified_artifact',
                   'radius_px':2,'points_xy':[[10,16],[16,16]]}]
    out=tmp_path/'conflict';s19.compile_review(pack,r,out_dir=out,image_root=root)
    mask=cv2.imread(str(out/f'{pack["tiles"][0]["tile_id"]}_expert_sparse_mask.png'),cv2.IMREAD_UNCHANGED)
    assert mask[16,10]==255 and mask[16,16]==255
    assert mask[16,6]==1 and mask[19,10]==0


@pytest.mark.parametrize('mutation,reason',[
    (lambda r,p: r.update({'package_sha256':'0'*64}), 'does not match'),
    (lambda r,p: r['entries'][0]['strokes'].append({'id':'manual:bad:3','label':'track_centerline','radius_px':1,'points_xy':[[1,2]]}), 'forged'),
    (lambda r,p: r['entries'][0]['strokes'].append({'id':f'manual:{p["tiles"][0]["tile_id"]}:0','label':'track_centerline','radius_px':1,'points_xy':[[999,0]]}), 'invalid expert stroke'),
    (lambda r,p: r['entries'][0]['links'].append({'a':'bogus','b':'unknown','decision':'same_track'}), 'missing/different'),
])
def test_tampering_or_forged_human_marks_rejected(tmp_path,mutation,reason):
    path,root=fixture(tmp_path);pack=s19.load_package(path,root);review=s19.empty_review(pack)
    mutation(review,pack)
    with pytest.raises(ValueError,match=reason):s19.compile_review(pack,review)


def test_unknown_pair_is_not_supervision_and_artifact_pair_rejected(tmp_path):
    p,root=fixture(tmp_path);pack=s19.load_package(p,root)
    r=s19.empty_review(pack);c=pack['tiles'][0]['candidates'];e=r['entries'][0]
    e['links']=[{'a':c[0]['id'],'b':c[1]['id'],'decision':'uncertain'}]
    assert s19.compile_review(pack,r)['reviewed_relation_labels']==0
    e['links'][0]['decision']='same_track'
    with pytest.raises(ValueError,match='not confirmed'):
        s19.compile_review(pack,r)


def test_package_rejects_changed_tile_or_proposal(tmp_path):
    path,root=fixture(tmp_path);first=next(root.glob('*.png'))
    im=cv2.imread(str(first),cv2.IMREAD_GRAYSCALE);im[1,1]=0;cv2.imwrite(str(first),im)
    with pytest.raises(ValueError,match='pixels changed'):
        s19.load_package(path,root)
    newer=tmp_path/'new';newer.mkdir()
    path,root=fixture(newer)
    raw=json.loads(path.read_text());raw['proposals'][0]['candidates'][0]['length_px']+=1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='digest mismatch'):
        s19.load_package(path,root)


def test_review_html_embeds_images_without_network_or_auto_labels(tmp_path):
    path,root=fixture(tmp_path);pack=s19.load_package(path,root)
    html=s19.make_offline_review_html(pack,root)
    assert 'data:image/png;base64,' in html
    assert 'fetch(' not in html and 'https://' not in html
    assert '___JSON_PAYLOAD___' not in html
    assert 'manual:' in html and 'different_track' in html
    assert len(html)>10000
