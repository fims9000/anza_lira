"""Synthetic source-locked full-tile reviewer tests; never use real H50Z pixels."""
import json
import cv2
import numpy as np
import pytest
from bubble_chamber.bitmap_tiles import sha256_file
from scripts.research.bubble_chamber.s113_fullframe_review import build
from tests.test_s113_public_smoke import _write_bmp


def _fixture(tmp_path, shift=False):
    image=tmp_path/'synthetic.bmp'
    pixels=np.full((256,256),210,np.uint8)
    cv2.line(pixels,(12,15),(220,200),22,2)
    _write_bmp(image,pixels)
    src=tmp_path/'model';src.mkdir()
    sample={'roi_xyxy_full':[0,0,256,256],
            'proposals':[{'x_full':15.5+(130 if shift else 0),'y_full':20.25,
                'x_local':15.5,'y_local':20.25,'score_uncalibrated':.86}]}
    (src/'candidates_R00_C00.json').write_text(json.dumps(sample))
    (src/'summary_PRIVATE.json').write_text(json.dumps({
        'source_image_sha256':sha256_file(image),'frame_size_wh':[256,256],
        'full_frame_tiled_pixel_coverage':True,'tiles_processed':1,'grid_rows_cols':[1,1],
        'tiles':[{'row':0,'column':0,'xyxy_full':[0,0,256,256],
                 'candidate_coordinates_private':'candidates_R00_C00.json'}]}))
    labels=tmp_path/'labels.json'
    labels.write_text(json.dumps({'synthetic.bmp':{'original_width':256,
        'original_height':256,'keypoints_count':2,'keypoints':[
            {'label':'f4','x_px':90,'y_px':100},
            {'label':'start','x_px':160,'y_px':140}]}}))
    return image,src,labels


def test_private_reviewer_exact_original_geometry(tmp_path):
    bmp,src,labels=_fixture(tmp_path)
    out=tmp_path/'review'
    report=build(bmp,src,out,labels,max_linelets=5)
    assert report['tiles']==1 and report['full_frame']
    assert report['bead_candidates']==1 and report['operator_marks']==2
    assert (out/'R00_C00.jpg').is_file()
    evidence=json.loads((out/'evidence_PRIVATE.json').read_text())
    assert evidence['source_sha256']==sha256_file(bmp)
    assert evidence['s112_evidence_sha256']
    assert evidence['not_physical_track_GT']
    assert 'source_sha256' in (out/'index.html').read_text()
    with pytest.raises(FileExistsError):build(bmp,src,out,labels)


def test_wrong_source_or_shifted_candidate_rejected(tmp_path):
    bmp,src,labels=_fixture(tmp_path,shift=True)
    with pytest.raises(ValueError,match='transformed'):build(bmp,src,tmp_path/'bad',labels,max_linelets=0)
    other=tmp_path/'other.bmp'
    _write_bmp(other,np.full((256,256),72,np.uint8))
    with pytest.raises(ValueError,match='provenance'):build(other,src,tmp_path/'wrong',max_linelets=0)


def test_incomplete_full_frame_cannot_claim_coverage(tmp_path):
    bmp,src,labels=_fixture(tmp_path)
    p=src/'summary_PRIVATE.json'
    rep=json.loads(p.read_text());rep['tiles_processed']=2
    p.write_text(json.dumps(rep))
    with pytest.raises(ValueError,match='Missing'):build(bmp,src,tmp_path/'incomplete',labels)
