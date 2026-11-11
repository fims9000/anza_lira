"""Synthetic/public smoke tests; no H50Z photo or operator annotation required."""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from bubble_chamber.s113_review_output import _group_runs, _cross_components, validate_review_decisions
from bubble_chamber.s113_frame_coverage import frame_roi_coverage
from bubble_chamber.s113_fiducials import detect_cross_candidates
from scripts.research.bubble_chamber.s113_fullframe_qa import tile_starts, run


def _write_bmp(path, gray):
    import struct
    h,w=gray.shape
    stride=(w+3)//4*4
    raw=b''.join(bytes(gray[y])+bytes(stride-w) for y in range(h-1,-1,-1))
    header=struct.pack('<2sIHHI',b'BM',1078+len(raw),0,0,1078)
    dib=struct.pack('<IiiHHIIiiII',40,w,h,1,8,0,len(raw),2835,2835,256,256)
    palette=b''.join(bytes((i,i,i,0)) for i in range(256))
    path.write_bytes(header+dib+palette+raw)


def test_public_full_coverage_grid_synthetic(tmp_path):
    assert tile_starts(9020,1024)[-1]==9020-1024
    assert tile_starts(26151,1024)[-1]==26151-1024
    bmp=tmp_path/'demo.bmp'
    gray=np.full((512,512),215,np.uint8)
    cv2.line(gray,(20,30),(450,470),25,1)
    _write_bmp(bmp,gray)
    marks={bmp.name:{'original_width':512,'original_height':512,'keypoints_count':2,
             'keypoints':[{'label':'f4','x_px':150,'y_px':160},
                          {'label':'start','x_px':400,'y_px':400}]}}
    labels=tmp_path/'labels.json';labels.write_text(json.dumps(marks))
    output=run(bmp,tmp_path/'private',tile_size=256,labels_json=labels,preview_side=64)
    assert output['total_tiles']==4
    assert output['cross_label_count']==1
    assert output['full_frame_pixel_coverage'] and not output['AZUNet_or_S112_used_on_full_frame']
    with pytest.raises(FileExistsError):run(bmp,tmp_path/'private',tile_size=256,labels_json=labels)


def test_public_cross_not_operator_identity():
    image=np.full((240,240),190,np.uint8)
    cv2.line(image,(70,70),(170,170),15,2)
    cv2.line(image,(170,70),(70,170),15,2)
    found=detect_cross_candidates(image,radii=(20,26,38,45),min_score=.55)
    assert any(np.hypot(z['x']-120,z['y']-120)<10 for z in found)
    assert all(z['operator_fiducial_id'] is None and not z['physical_fiducial_gt'] for z in found)
    assert detect_cross_candidates(np.full_like(image,190),min_score=.6)==[]


def test_public_never_convert_gap_to_observed_photo():
    pts=np.stack([np.arange(10),np.zeros(10)],axis=1)
    flags=np.array([True]*4+[False]*3+[True]*3)
    result=_group_runs(pts,flags)
    assert sum(x['n_edges'] for x in result)==9
    assert result[0]['kind']=='observed_candidate'
    assert any(x['kind']=='inferred_inside_curve' for x in result)


def test_public_fiducial_not_large_scan_artifact():
    mask=np.zeros((240,340),np.uint8)
    cv2.line(mask,(30,30),(70,70),255,4)
    cv2.line(mask,(70,30),(30,70),255,4)
    mask[120:230,100:330]=255
    regions=_cross_components(mask)
    assert len(regions)==2
    assert any(x['status']=='SMALL_FIDUCIAL_CANDIDATE_UNVERIFIED' for x in regions)
    assert any(x['status']=='LARGE_PROTECTED_SCAN_ARTIFACT_OR_VIGNETTE' for x in regions)


def test_public_review_requires_matching_provenance():
    stub={'roi':'ROI','source_png_sha256':'s','s111_global_tracks_sha256':'m',
          's112_bubble_proposals_sha256':'b','operator_labels_sha256':None,'tracks':[{'id':'t'}],
          'protected_regions':[],'bubble_chain_review_only':[],'bridge_review_only':[]}
    good={'schema':'S113_EXPERT_REVIEW_V1','roi':'ROI','source_sha256':'s',
          's111_sha256':'m','s112_sha256':'b','operator_labels_sha256':None,'decisions':[{'kind':'track','id':'t','decision':'track'}]}
    assert validate_review_decisions(stub,good)['accepted_review_count']==1
    for key in ['roi','source_sha256','s111_sha256','s112_sha256']:
        bad={**good,key:'wrong'}
        with pytest.raises(ValueError):validate_review_decisions(stub,bad)