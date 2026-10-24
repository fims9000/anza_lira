from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from bubble_chamber.h50z_seed_analysis import audit_landmarks,make_keypoint_pilot,_nearest_mask


def _fixture(tmp_path):
    root=tmp_path/'frames';root.mkdir()
    image=np.full((320,320),228,dtype=np.uint8)
    cv2.line(image,(20,160),(310,160),60,2)
    cv2.line(image,(160,20),(160,310),70,2)
    cv2.imwrite(str(root/'filmA.bmp'),image)
    d={'filmA.bmp':{'original_width':320,'original_height':320,'keypoints_count':2,
                     'keypoints':[{'label':'start','x_px':160.0,'y_px':160.0},
                                  {'label':'f1','x_px':30.0,'y_px':40.0}]},
       'filmB.bmp':{'original_width':320,'original_height':320,'keypoints_count':1,
                    'keypoints':[{'label':'start','x_px':30.0,'y_px':40.0}]}}
    p=tmp_path/'labels.json';p.write_text(json.dumps(d))
    return p,root


def test_catalog_counts_missing_images_and_valid_centers(tmp_path):
    labels,root=_fixture(tmp_path)
    audit=audit_landmarks(labels,root)
    assert audit['available_original_bmps']==1
    assert audit['annotated_frames_in_json']==2
    assert audit['missing_original_bmps']==1
    assert audit['available_start_points']==1
    assert audit['available_fiducial_points']==1
    assert audit['segmentation_mask_available'] is False


def test_tile_and_real_image_diagnostic_not_segmentation_accuracy(tmp_path):
    labels,root=_fixture(tmp_path)
    result=make_keypoint_pilot(labels,root,tmp_path/'out',tile_size=256)
    assert len(result['labelled_tiles'])==2
    assert result['real_segmentation_accuracy']=='NOT_MEASURABLE'
    assert result['keypoint_proximity_diagnostics']['contrast']['start_landmark_count']==1
    assert all((tmp_path/'out'/r['tile_filename']).is_file() for r in result['labelled_tiles'])
    assert all(r['no_dense_target'] for r in result['labelled_tiles'])


def test_no_ridge_returns_none():
    assert _nearest_mask(np.zeros((64,64),bool),(32,32)) is None


def test_reject_malformed_keypoints(tmp_path):
    labels,root=_fixture(tmp_path)
    d=json.loads(labels.read_text());d['filmA.bmp']['keypoints'][0]['x_px']=999
    labels.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='out of image'):
        audit_landmarks(labels,root)
