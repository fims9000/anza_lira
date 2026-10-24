import csv
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from bubble_chamber.small_bundle import create_small_bundle


def _inputs(tmp_path):
    src=tmp_path/'bmp';src.mkdir()
    img=np.fromfunction(lambda y,x:(3*y+x)%255,(256,288)).astype('uint8')
    cv2.imwrite(str(src/'a.bmp'),img)
    cv2.imwrite(str(src/'view.bmp'),img)
    rows=[dict(source='H50Z_LABELS_JSON',source_image='a.bmp',frame_id='H50Z_030_01_005',view_id='',point_id='start_001',
               label_semantics='VERTEX_START_KEYPOINT_NOT_TRACK_MASK',x0=64,y0=96,x1=128,y1=160,source_x_px=96,source_y_px=128),
          dict(source='VISUALMETRO_SPARSE_3VIEW',source_image='070_008 VIEW 1',frame_id='070_008',view_id='1',point_id='e1t01_index_01',
               label_semantics='EXPERT_TRACK_CLICK_SPARSE_NOT_DENSE_MASK',x0=96,y0=80,x1=160,y1=144,source_x_px=128,source_y_px=112)]
    req=tmp_path/'requests.csv'
    with req.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    mapping=tmp_path/'mapping.csv';mapping.write_text('frame_id,view_id,file\n070_008,1,view.bmp\n')
    return src,img,req,mapping


def test_exact_original_pixel_crops_with_explicit_mapping(tmp_path):
    src,img,req,mapping=_inputs(tmp_path)
    dest=tmp_path/'pack.zip'
    m=create_small_bundle(req,dest,h50z_root=src,visualmetro_root=src,visualmetro_view_map=mapping,strict=True)
    assert m['tile_count']==2 and m['missing_count']==0
    assert all(rec['label_semantics'].endswith(('NOT_TRACK_MASK','NOT_DENSE_MASK')) for rec in m['tiles'])
    with ZipFile(dest) as z:
        for row in m['tiles']:
            a,b,c,d=row['crop_xyxy_in_full_bmp']
            decoded=cv2.imdecode(np.frombuffer(z.read(row['zip_name']),np.uint8),cv2.IMREAD_GRAYSCALE)
            assert np.array_equal(decoded,img[b:d,a:c])


def test_missing_image_is_not_guessed(tmp_path):
    src,img,req,mapping=_inputs(tmp_path)
    dest=tmp_path/'pack.zip'
    m=create_small_bundle(req,dest,h50z_root=src)
    assert m['tile_count']==1 and m['missing_count']==1
    with pytest.raises(FileNotFoundError,match='no partial archive'):
        create_small_bundle(req,tmp_path/'strict.zip',h50z_root=src,strict=True)
    assert not (tmp_path/'strict.zip').exists()


def test_refuses_wrong_matched_view_or_unexpected_image_dimensions(tmp_path):
    src,img,req,mapping=_inputs(tmp_path)
    bad=(np.zeros((100,100),dtype=np.uint8))
    cv2.imwrite(str(src/'view.bmp'),bad)
    with pytest.raises((ValueError,Exception)):
        create_small_bundle(req,tmp_path/'bad.zip',h50z_root=src,visualmetro_root=src,visualmetro_view_map=mapping)


def test_rejects_path_traversal_in_view_map(tmp_path):
    src,img,req,mapping=_inputs(tmp_path)
    mapping.write_text('frame_id,view_id,file\n070_008,1,../a.bmp\n')
    with pytest.raises(ValueError,match='within source directory'):
        create_small_bundle(req,tmp_path/'bad.zip',h50z_root=src,visualmetro_root=src,visualmetro_view_map=mapping)
