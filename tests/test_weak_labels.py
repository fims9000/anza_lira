from pathlib import Path
import numpy as np
import cv2
import pytest
from bubble_chamber.weak_labels import sparse_patch, validate_view_alignment, pairing_table, attach_bmp_inventory, AnnotationAlignmentError
from bubble_chamber.visualmetro_import import SCHEMA


def small_manifest():
    frame={"frame_id":"070_008","film_id":"070",
           "fiducials":{"view_pixel_xy":{str(v):{f'f{i}':[10+i,10+i] for i in range(1,9)} for v in (1,2,3)}},
           "tracks":[{"track_id":"e1t01","n_points":3,"xyz_cm":[[1,2,3],[2,3,4],[3,4,5]],
                      "measured_pixel_xy":{str(v):[[10,15],[25,24],[40,32]] for v in (1,2,3)}}]}
    return {"schema":SCHEMA,"frames":[frame]}


def test_sparse_positive_is_not_complete_binary_mask():
    frame=small_manifest()['frames'][0]
    image=np.full((80,80),185,dtype=np.uint8)
    out=sparse_patch(image,frame,1,(20,24),patch_size=64,point_radius_px=2)
    assert out['image'].shape==(64,64)
    assert out['clicked_foreground'].sum()>0
    assert out['visual_line_guide_untrusted'].sum()>0
    assert out['negative_supervision']=='UNAVAILABLE'
    assert out['dense_segmentation_ground_truth'] is False
    assert out['observed_pixels'].sum()<64*64
    assert out['containing_track_ids']==['e1t01']
    assert out['roi_xyxy_original']==[-12,-8,52,56]


def test_out_of_bounds_fids_and_views_rejected():
    frame=small_manifest()['frames'][0]
    assert validate_view_alignment(frame,3,(80,80))['measured_points_inside_image']==3
    frame['tracks'][0]['measured_pixel_xy']['2'][-1]=[800,20]
    with pytest.raises(AnnotationAlignmentError,match='outside image'):
        validate_view_alignment(frame,2,(80,80))


def test_stereo_correspondences_retain_track_identity():
    tuples=pairing_table(small_manifest())
    assert len(tuples)==3
    assert tuples[1]['v1_x_px']==25
    assert tuples[1]['v2_x_px']==25
    assert tuples[1]['v3_y_px']==24
    assert tuples[1]['track_id']=='e1t01'


def test_explicit_bmp_mapping_detects_missing_views(tmp_path):
    manifest=small_manifest()
    img=np.full((80,80),220,np.uint8)
    cv2.imwrite(str(tmp_path/'1.bmp'),img)
    mapping=tmp_path/'files.csv'
    mapping.write_text('frame_id,view_id,file\n070_008,1,1.bmp\n')
    with pytest.raises(AnnotationAlignmentError,match='missing BMP mappings'):
        attach_bmp_inventory(manifest,mapping,tmp_path)


def test_duplicate_bmp_mapping_rejected(tmp_path):
    manifest=small_manifest()
    img=np.full((80,80),220,np.uint8)
    cv2.imwrite(str(tmp_path/'1.bmp'),img)
    mapping=tmp_path/'files.csv'
    mapping.write_text('frame_id,view_id,file\n070_008,1,1.bmp\n070_008,1,1.bmp\n')
    with pytest.raises(AnnotationAlignmentError,match='duplicate'):
        attach_bmp_inventory(manifest,mapping,tmp_path)
