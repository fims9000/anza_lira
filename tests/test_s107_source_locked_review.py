"""S107 guard tests: unlabelled source pixels are UNKNOWN, never negatives."""
import json

import cv2
import numpy as np
import pytest

from bubble_chamber.s107_review import (
    SCHEMA, EXPORT_SCHEMA, _blind_rects, _manifest_identity,
    verify_pack, validate_annotation, evaluate,
)
from bubble_chamber.bitmap_tiles import sha256_file


def _fixture(tmp_path):
    pack = tmp_path / 'pack'
    (pack / 'images').mkdir(parents=True)
    image = np.full((160,160), 205, dtype=np.uint8)
    baseline = np.zeros_like(image)
    cv2.line(image,(5,50),(120,50),140,2)
    cv2.line(baseline,(5,50),(32,50),255,1)
    cv2.line(baseline,(55,50),(120,50),255,1)
    for name,pixels in [('raw.png',image),('contrast.png',image),('s105.png',baseline)]:
        assert cv2.imwrite(str(pack/'images'/name),pixels)
    row={'tile_id':'A','source_name':'H50Z_030_01_005.bmp',
         'source_sha256':'a'*64,'source_dimensions_xy':[250,250],
         'source_rect_xyxy':[25,30,185,190],'width':160,'height':160,
         'original_png':'images/raw.png',
         'original_png_sha256':sha256_file(pack/'images/raw.png'),
         'contrast_png':'images/contrast.png',
         'contrast_png_sha256':sha256_file(pack/'images/contrast.png'),
         's105_mask_png':'images/s105.png',
         's105_mask_png_sha256':sha256_file(pack/'images/s105.png'),
         's105_display_png':None,'selection':'BLIND_FIXED_GRID_NEVER_SCORED_FOR_SELECTION'}
    manifest={'schema':SCHEMA,'tiles':[row],
              'source_images':[{'name':row['source_name'],'sha256':row['source_sha256']}],
              'unmarked_pixels_mean':'UNKNOWN'}
    manifest['manifest_sha256']=_manifest_identity(manifest)
    (pack/'manifest.json').write_text(json.dumps(manifest))
    return pack,manifest


def test_fixed_blind_roi_without_model():
    a=list(_blind_rects('a'*64,4400,8700,7))
    assert a==list(_blind_rects('a'*64,4400,8700,7))
    assert a!=list(_blind_rects('a'*64,4400,8700,8))
    assert len(a)==4
    for _,(x0,y0,x1,y1) in a:
        assert x1-x0==y1-y0==1024


def test_empty_reference_is_not_perfect_model(tmp_path):
    pack,m=_fixture(tmp_path)
    assert verify_pack(pack)['manifest_sha256']==m['manifest_sha256']
    export={'schema':EXPORT_SCHEMA,'manifest_sha256':m['manifest_sha256'],
            'strokes':[],'reviewer_confirmed':False}
    report=evaluate(export,pack,tmp_path/'evaluation')
    assert report['s105_centerline_coverage_on_explicitly_marked_strokes'] is None
    assert report['global_precision_or_recall_supported'] is False
    labels=cv2.imread(str(tmp_path/'evaluation/sparse_labels/A_partial_stroke_targets.png'),0)
    assert set(np.unique(labels))=={0}


def test_gap_and_source_safety(tmp_path):
    pack,m=_fixture(tmp_path)
    row={'tile_id':'A','stroke_id':'r1','kind':'track','certainty':'clear',
         'visual_instance_id':'T001','points_source_xy':[[30,80],[55,80],[85,80]]}
    export={'schema':EXPORT_SCHEMA,'manifest_sha256':m['manifest_sha256'],
            'strokes':[row],'reviewer_confirmed':True}
    validate_annotation(export,m)
    row['points_source_xy']=[[30,80],[110,80]]
    with pytest.raises(ValueError,match='gap'):
        validate_annotation(export,m)
    row['points_source_xy']=[[30,80],[200,80]]
    with pytest.raises(ValueError,match='outside'):
        validate_annotation(export,m)
    row['points_source_xy']=[[30,80],[45,80]]
    export['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='mismatch'):
        validate_annotation(export,m)


def test_changed_image_bytes_are_rejected(tmp_path):
    pack,m=_fixture(tmp_path)
    (pack/'images/raw.png').write_bytes(b'tampered image')
    with pytest.raises(ValueError,match='modified'):
        verify_pack(pack)
