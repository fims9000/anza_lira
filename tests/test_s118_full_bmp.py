"""Short reproducible full BMP pipeline contract tests on synthetic scenes."""
import json
import cv2
import numpy as np
import pytest
from scripts.research.bubble_chamber.run_s118_full_bmp import run, _tiles

def test_tile_grid_covers_last_seam_and_is_deterministic():
    tiles=_tiles([0,0,205,150],96,32)
    assert tiles==_tiles([0,0,205,150],96,32)
    assert tiles[0]['xyxy']==[0,0,96,96]
    assert any(t['xyxy'][2]==205 for t in tiles)
    assert any(t['xyxy'][3]==150 for t in tiles)
    assert len({tuple(t['xyxy']) for t in tiles})==len(tiles)


def test_small_complete_photo_pipeline_and_resume(tmp_path):
    # Synthetic photo is intentionally not a source of particle GT assertions.
    photo=np.full((150,205),190,np.uint8)
    cv2.line(photo,(12,36),(190,44),20,2)
    cv2.line(photo,(15,90),(195,90),30,2)
    bmp=tmp_path/'synthetic.bmp'
    assert cv2.imwrite(str(bmp),photo)
    output=tmp_path/'private_pipeline'
    a=run(bmp,output,[0,0,205,150],tile_size=96,overlap=32,max_tiles=2)
    assert a['processed_full_requested_bbox'] is False
    assert a['stitch_report']['input_curve_instances']>=0
    assert (output/'S118_STITCH_00002'/'S118_VECTOR_REVIEW.svg').exists()
    with pytest.raises(FileExistsError):
        run(bmp,output,[0,0,205,150],tile_size=96,overlap=32,max_tiles=2)
    b=run(bmp,output,[0,0,205,150],tile_size=96,overlap=32,max_tiles=3,resume=True)
    assert b['processed_tile_count']==3
    assert b['tiles'][0]['status']=='REUSED_VERIFIED_TILE'
    assert b['tiles'][2]['status']=='PROCESSED_FROM_BMP'
    assert b['stitch_report']['tile_count']==3
    c=run(bmp,output,[0,0,205,150],tile_size=96,overlap=32,max_tiles=3,resume=True)
    assert c['processed_tile_count']==3
    assert all(t['status']=='REUSED_VERIFIED_TILE' for t in c['tiles'])
    assert json.loads((output/'S118_RUN_REPORT_00003.json').read_text())['processed_tile_count']==3


def test_resume_rejects_tampered_bmp_and_bad_bbox(tmp_path):
    bmp=tmp_path/'a.bmp'
    cv2.imwrite(str(bmp),np.full((96,96),190,np.uint8))
    output=tmp_path/'private'
    run(bmp,output,tile_size=96,overlap=32,max_tiles=1)
    cv2.imwrite(str(bmp),np.full((96,96),180,np.uint8))
    with pytest.raises(ValueError,match='different source image'):
        run(bmp,output,tile_size=96,overlap=32,max_tiles=1,resume=True)
    with pytest.raises(ValueError,match='BBox exceeds'):
        run(bmp,tmp_path/'other',bbox=[0,0,100,100],tile_size=96)
