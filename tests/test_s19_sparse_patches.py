import json
from pathlib import Path
import cv2,numpy as np,pytest
from tests.test_s19_real_review import fixture
from bubble_chamber.s19_real_review import load_package,empty_review,compile_review
from bubble_chamber.s19_sparse_patches import make_sparse_patches
from scripts.research.bubble_chamber.s14_train_sparse_anza import read_reviewed_manifest


def reviewed_fixture(tmp_path):
    path,root=fixture(tmp_path);pack=load_package(path,root);review=empty_review(pack)
    for tile,e in zip(pack['tiles'],review['entries']):
        e['strokes']=[{'id':f'manual:{tile["tile_id"]}:0','label':'track_centerline',
                      'radius_px':1,'points_xy':[[12,12],[18,12],[25,12]]},
                     {'id':f'manual:{tile["tile_id"]}:1','label':'certified_artifact',
                      'radius_px':1,'points_xy':[[40,40],[46,40],[52,40]]}]
    out=tmp_path/'compiled';r=compile_review(pack,review,out_dir=out,image_root=root)
    assert r['trainable_real_image_baseline']
    return out/'s14_sparse_image_reviews.json'


def test_crop_preserves_255_and_original_frame_grouping(tmp_path):
    manifest=reviewed_fixture(tmp_path)
    a,s=make_sparse_patches(manifest,tmp_path/'patches',size=32)
    rows=json.loads(a.read_text())['entries']
    assert rows
    assert {x['split'] for x in rows}=={'train','val'}
    for e in rows:
        im=cv2.imread(str(a.parent/e['image']),cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(str(a.parent/e['mask']),cv2.IMREAD_UNCHANGED)
        assert im.shape==mask.shape==(32,32)
        assert set(np.unique(mask)).issubset({0,1,255})
        assert (mask==255).any() and ((mask==0)|(mask==1)).any()
        assert e['s19_patch_origin']['sampling_only_from_certified_pixels']
    train,val={e['frame_id'] for e in rows if e['split']=='train'},{e['frame_id'] for e in rows if e['split']=='val'}
    assert train and val and train.isdisjoint(val)
    assert s['auto_human_labels_created']==0
    loaded,summary=read_reviewed_manifest(a)
    assert summary['source_run_count']==1


def test_deterministic_patch_crops_and_invalid_metadata(tmp_path):
    path=reviewed_fixture(tmp_path)
    a,stat_a=make_sparse_patches(path,tmp_path/'p1',size=32)
    b,stat_b=make_sparse_patches(path,tmp_path/'p2',size=32)
    assert stat_a['patch_examples_by_split']==stat_b['patch_examples_by_split']
    ax=json.loads(a.read_text())['entries'];bx=json.loads(b.read_text())['entries']
    assert [(r['image_sha256'],r['mask_sha256']) for r in ax]==[(r['image_sha256'],r['mask_sha256']) for r in bx]
    with pytest.raises(ValueError):make_sparse_patches(path,tmp_path/'bad',size=13)
    source=json.loads(path.read_text());source['entries'][0]['mask_sha256']='0'*64;path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='checksum drift'):
        make_sparse_patches(path,tmp_path/'bad2',size=32)
