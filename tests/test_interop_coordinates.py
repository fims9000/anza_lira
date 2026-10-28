import numpy as np
import pytest
from bubble_chamber.interop_api import ScoreBinding,assemble_lines
from bubble_chamber.interop_coordinates import CropTransform,private_review_payload


def curve():return np.column_stack((np.linspace(10,101,37), np.full(37,63.)))


def test_crop_roundtrip_exactly_and_immutable_source_frame():
    src=np.arange(512*512,dtype=np.uint32).reshape(512,512).astype(np.uint8)
    copy=src.copy();roi=CropTransform(160,160,512,512)
    result=roi.extract(src)
    assert result.shape==(192,192) and not result.flags.writeable
    assert result[0,0]==src[160,160]
    assert np.array_equal(src,copy)
    original=curve()
    parent=roi.to_source((original,))[0]
    assert not parent.flags.writeable
    assert np.allclose(parent[0],original[0]+np.array([160,160]))
    assert np.allclose(roi.to_crop((parent,))[0],original)
    assert parent[0][0] != original[0][0]


def test_explicit_bounds_and_no_silent_scale_or_coord_clipping():
    with pytest.raises(ValueError):CropTransform(160,160,512,512,384,384)
    with pytest.raises(ValueError):CropTransform(500,500,512,512)
    with pytest.raises(ValueError):CropTransform(160.,160,512,512)
    a=CropTransform(160,160,512,512)
    with pytest.raises(ValueError):a.extract(np.zeros((192,192),np.uint8))
    with pytest.raises(ValueError):a.to_crop((curve()+300,))


def test_opt_in_private_overlay_has_UNKNOWN_links_but_aggregate_public_doesnt():
    t=CropTransform(160,160,512,512)
    b=ScoreBinding('fake','toy-only',lambda rows:np.ones(len(rows)),.7,
                   source_cal_id='TOY_ONLY',strategy='none')
    result=assemble_lines((curve(),curve()+np.array([0,20]),curve()+np.array([0,45])),b,detector_id='toy-only')
    public=result.as_public_dict();assert 'lines_PRIVATE' not in public
    private=private_review_payload(result,t,source_image_id='LOCAL_OFFLINE_ONLY')
    assert private['publication_safe'] is False
    assert len(private['lines_PRIVATE'])==3
    assert private['native_physics_gt_count']==0
    assert not private['physical_auto_joins_authorized']
    assert 'source_image_id_PRIVATE' not in public
    with pytest.raises(ValueError):private_review_payload(result,t,source_image_id='')
