"""Synthetic and negative tests for unbiased S122 review crops."""
import json
from pathlib import Path
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research.bubble_chamber.make_s122_20_roi_review import grid20,create_pack


def test_grid20_is_exactly_20_distinct_full_frame_source_rois():
    for wh in ((9103,26317),(9268,25324),(1400,1600)):
        points=grid20(*wh,roi=1024)
        assert len(points)==20
        assert len(set(points))==20
        for _,x0,y0,x1,y1 in points:
            assert 0<=x0<x1<=wh[0]
            assert 0<=y0<y1<=wh[1]
            assert x1-x0==1024 and y1-y0==1024
    assert grid20(9103,26317)==grid20(9103,26317)
    assert grid20(9103,26317)!=grid20(9103,26317,seed=5)


def test_grid20_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        grid20(1000,2000,roi=1024)
    with pytest.raises(ValueError):
        grid20(1200,2000,roi=40)


def test_make_20_reads_only_original_bmp_and_manifest(tmp_path):
    bmp=tmp_path/'source.bmp'
    raw=np.full((1600,1400),205,np.uint8)
    cv2.line(raw,(100,100),(1300,1500),50,2)
    cv2.imwrite(str(bmp),raw)
    import hashlib
    hash=hashlib.sha256(bmp.read_bytes()).hexdigest()
    run=tmp_path/'run';tile=run/'tiles'/'T00000';tile.mkdir(parents=True)
    (run/'RUN_CONFIG.json').write_text(json.dumps({
        'source_image_sha256':hash,'shape_hw':[1600,1400],
        'bbox_xyxy':[0,0,1400,1600],'grid_count':1}))
    c={'candidate_id':'one','observed_pixel_path_xy':[[100,100],[1300,1500]]}
    (tile/'S115_CURVES.json').write_text(json.dumps({'curves':[c]}))
    (run/'S118_MANIFEST_00001.json').write_text(json.dumps({'tiles':[
        {'tile_id':'T00000','xyxy':[0,0,1400,1600],
         'curves':'tiles/T00000/S115_CURVES.json'}]}))
    stitch=run/'S118_STITCH_00001';stitch.mkdir()
    (stitch/'S118_STITCH_NOT_GT.json').write_text(json.dumps({'groups':[
        {'global_group_id':'g','source_curve_keys':['T00000:one']}] }))
    out=tmp_path/'qa';report=create_pack(bmp,run,out)
    assert report['images_created']==20
    assert (out/'R20_SOURCE_VS_PROPOSALS.jpg').is_file()
    assert (out/'REVIEW_20_REGIONS.csv').read_bytes().startswith(b'\xef\xbb\xbf')
    labels=json.loads((out/'SOURCE_AND_REVIEW_INFO.json').read_text())
    assert labels['manual_labels_used_for_selection'] is False
    assert labels['regions'][0]['source_sha256']==hash
    wrong=tmp_path/'wrong.bmp';cv2.imwrite(str(wrong),np.full_like(raw,100))
    with pytest.raises(ValueError,match='SHA'):
        create_pack(wrong,run,tmp_path/'bad')
