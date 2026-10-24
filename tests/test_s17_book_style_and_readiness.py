import json
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.research.bubble_chamber.s17_match_unlabeled_photo_style import (
    grayscale_quantile_match, run,
)
from scripts.research.bubble_chamber.s17_audit_book_readiness import audit
from scripts.research.bubble_chamber.s17_make_book_prior_dataset import generate
from scripts.research.bubble_chamber.s14_train_sparse_anza import read_reviewed_manifest


def test_monotonic_photo_style_does_not_change_rank_order():
    source=np.arange(256,dtype='uint8').reshape(16,16)
    real=np.asarray([*np.zeros(200),*np.ones(56)*180],dtype='uint8').reshape(16,16)
    styled=grayscale_quantile_match(source,real)
    assert styled.shape==source.shape and styled.dtype==np.uint8
    assert np.all(np.diff(styled.ravel().astype(int))>=0)
    with pytest.raises(ValueError):
        grayscale_quantile_match(np.zeros((2,2),np.float32),real)
    with pytest.raises(ValueError,match='degenerate'):
        grayscale_quantile_match(np.zeros((16,16),np.uint8),real)


def test_style_images_retain_partial_labels_and_no_real_gt(tmp_path):
    raw=tmp_path/'source';r=tmp_path/'real';out=tmp_path/'styled'
    generate(raw,size=64,train=4,val=2,unlabeled=2,test=2)
    r.mkdir()
    img=np.zeros((64,64),np.uint8)+170
    cv2.line(img,(6,11),(54,41),30,2)
    cv2.imwrite(str(r/'unlabeled_real_crop.png'),img)
    stats=run(raw,r,out)
    assert stats['book_samples_restyled']==10
    assert stats['new_real_track_annotations']==0
    groups,info=read_reviewed_manifest(out/'s17_book_synthetic_sparse_manifest.json')
    assert info['split_examples']=={'train':4,'val':2,'unlabeled':2}
    for x in groups['train']+groups['val']:
        _,mask,row=x
        original=cv2.imread(str(raw/row['mask']),cv2.IMREAD_UNCHANGED)
        assert np.array_equal(mask,original)
        assert set(np.unique(mask)).issubset({0,1,255})
    test=json.loads((out/'s17_book_test_truth_only.json').read_text())
    assert all(e['is_real_photo'] is False for e in test['examples'])


def test_single_view_H50Z_not_three_view_or_mass_input(tmp_path):
    photos=tmp_path/'photos';photos.mkdir()
    labels={}
    for frame in ('005','006'):
        f=f'H50Z_030_01_{frame}.bmp'
        (photos/f).write_bytes(b'BM_FAKE_NO_IMAGE_OPEN_REQUIRED')
        labels[f]={'original_height':16,'original_width':16,
                   'keypoints':[{'label':'start','x_px':1,'y_px':2},
                                {'label':'f1','x_px':3,'y_px':4}]}
    p=tmp_path/'labels.json';p.write_text(json.dumps(labels))
    report=audit(p,photos)
    assert report['available_bmps']==2
    assert report['complete_three_view_events']==0
    assert report['available_starts']==2
    assert all(not row['can_run_book_physical_mass_fit'] for row in report['events'])
    assert all('OTHER_TWO_PROJECTIONS_NOT_AVAILABLE' in row['blocking_conditions'] for row in report['events'])
    assert report['expert_track_mask_count']==0


def test_three_views_alone_not_enough_without_calibration_and_stop(tmp_path):
    photos=tmp_path/'photos';photos.mkdir()
    labels={}
    for view in (1,2,3):
        name=f'H50Z_030_0{view}_005.bmp';(photos/name).write_bytes(b'BM')
        labels[name]={'keypoints':[{'label':'start','x_px':1,'y_px':1}]}
    p=tmp_path/'labels.json';p.write_text(json.dumps(labels))
    report=audit(p,photos)
    assert report['complete_three_view_events']==1
    event=report['events'][0]
    assert not event['can_run_book_physical_mass_fit']
    assert 'OPTICAL_FIDUCIAL_OR_BFIELD_CALIBRATION_UNVERIFIED' in event['blocking_conditions']
    assert 'STOPPING_PARTICLE_AND_STEREO_TRACK_CORRESPONDENCE_NOT_EXPERT_CONFIRMED' in event['blocking_conditions']


def test_invalid_photo_filenames_rejected(tmp_path):
    x={'../H50Z_030_01_005.bmp':{'keypoints':[]}}
    p=tmp_path/'labels.json';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='unsafe'):
        audit(p,tmp_path)
