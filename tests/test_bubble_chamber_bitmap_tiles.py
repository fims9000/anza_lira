import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import BitmapFormatError, GrayBMP8, export_h50z_tiles, export_visualmetro_triplet_tiles


def _bmp(path: Path, h=256,w=288):
    image=np.fromfunction(lambda y,x:(2*y+x)%255,(h,w),dtype=int).astype(np.uint8)
    assert cv2.imwrite(str(path),image)
    return image


def test_uncompressed_bottom_up_mapping_and_coordinates(tmp_path):
    src=tmp_path/'a.bmp'; original=_bmp(src)
    reader=GrayBMP8(src)
    assert reader.width==original.shape[1] and reader.height==original.shape[0]
    crop,loc=reader.crop((90.5,110.2),size=64)
    x0,y0,x1,y1=loc.original_xyxy
    assert np.array_equal(crop,original[y0:y1,x0:x1])
    assert loc.center_xy_local==(90.5-x0,110.2-y0)
    with pytest.raises(ValueError,match="center"):
        reader.crop((2000,2),64)


def test_non_image_rejected(tmp_path):
    p=tmp_path/'f.bmp';p.write_bytes(b'hello')
    with pytest.raises(BitmapFormatError):GrayBMP8(p)


def test_h50z_keypoint_tiles_keep_partial_semantics_and_source_dims(tmp_path):
    d=tmp_path/'source';d.mkdir();img=_bmp(d/'frame.bmp')
    labels={"frame.bmp":{"original_width":img.shape[1],"original_height":img.shape[0],
            "keypoints_count":3,"keypoints":[{"label":"f4","x_px":110.0,"y_px":100.0},
                                             {"label":"start","x_px":90.5,"y_px":110.2},
                                             {"label":"start","x_px":250.0,"y_px":240.0}]}}
    l=tmp_path/'labels.json';l.write_text(json.dumps(labels))
    report=export_h50z_tiles(d,l,tmp_path/'out',tile_size=64)
    assert len(report['tiles'])==3
    assert report['supervision']=='FIDUCIALS_AND_START_KEYPOINTS_ONLY_NOT_FULL_TRACK_MASKS'
    assert report['unmarked_pixels']=='UNLABELED_NOT_TRUE_NEGATIVE'
    for entry in report['tiles']:
        assert (tmp_path/'out'/entry['filename']).is_file()
        x,y=entry['point_xy_local_px'];assert 0<=x<64 and 0<=y<64
    label_local=[e['point_xy_local_px'] for e in report['tiles'] if e['selected_label']=='start'][0]
    assert isinstance(label_local,list)


def test_h50z_reject_mismatched_dimensions(tmp_path):
    d=tmp_path/'source';d.mkdir();_bmp(d/'frame.bmp')
    l=tmp_path/'labels.json';l.write_text(json.dumps({'frame.bmp':{'original_width':123,'original_height':256,'keypoints_count':0,'keypoints':[]}}))
    with pytest.raises(ValueError,match="dimensions mismatch"):
        export_h50z_tiles(d,l,tmp_path/'out',tile_size=64)


def test_triplet_matched_index_and_original_pixels(tmp_path):
    sources={}
    for view in (1,2,3):
        path=tmp_path/f'view{view}.bmp';_bmp(path);sources[view]=path
    track={'frame_id':'070_008','track_id':'e1t01','n_points':4,'file_sha256':'a'*64,
           'xyz_cm':[[1,2,3],[2,3,4],[3,4,5],[4,5,6]],
           'measured_pixel_xy':{str(v):[[70+i*40,80+i*30] for i in range(4)] for v in (1,2,3)}}
    out=export_visualmetro_triplet_tiles(track,sources,tmp_path/'out',indices=(0,2),tile_size=64)
    assert len(out['selected_correspondences'])==2
    assert out['camera_geometry_calibration_verified'] is False
    assert out['annotation_semantics']=='SPARSE_MEASURED_TRACK_COORDINATES_NO_DENSE_FOREGROUND_MASK'
    for entry in out['selected_correspondences']:
        assert set(entry['views'])=={'1','2','3'}
        for view in (1,2,3):
            v=entry['views'][str(view)]
            xy=v['point_xy_original_px'];x0,y0,x1,y1=v['crop_xyxy_in_source']
            assert np.allclose(v['point_xy_local_px'],[xy[0]-x0,xy[1]-y0])
            crop=cv2.imread(str(tmp_path/'out'/v['filename']),cv2.IMREAD_GRAYSCALE)
            assert crop.shape==(64,64)
    with pytest.raises(ValueError,match="outside view"):
        track['measured_pixel_xy']['3'][0]=[800,2]
        export_visualmetro_triplet_tiles(track,sources,tmp_path/'bad',indices=(0,),tile_size=64)
