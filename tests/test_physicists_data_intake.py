"""Only generated BMP/film070-shaped fixtures; NO authentic private H50Z data."""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from bubble_chamber.physicists_intake import (
    inventory_sources, inventory_bmps, preview_crosses, check_manual_stereo_map)
from scripts.research.bubble_chamber.physicists_data_intake import main


def fixture(tmp_path):
    folder=tmp_path/'bmps';folder.mkdir()
    im=np.full((320,320),206,np.uint8)
    cv2.line(im,(20,150),(300,160),23,2)
    assert cv2.imwrite(str(folder/'TEST_005.bmp'),im)
    assert cv2.imwrite(str(folder/'NO_LABEL.bmp'),im)
    labels={'TEST_005.bmp':{'original_width':320,'original_height':320,'keypoints_count':2,
        'keypoints':[{'label':'start','x_px':150,'y_px':157},
                     {'label':'f1','x_px':135,'y_px':140}]},
        'NOT_MOUNTED.bmp':{'original_width':320,'original_height':320,'keypoints_count':1,
                           'keypoints':[{'label':'start','x_px':100,'y_px':100}]}}
    p=tmp_path/'labels.json';p.write_text(json.dumps(labels))
    return folder,p


def fake_film(root):
    folder=root/'070_008';folder.mkdir(parents=True)
    (folder/'070_008_trkList.txt').write_text('e1t01\n')
    (folder/'070_008_fids.txt').write_text('\n'.join(
        f'{view} -- view\n'+'\n'.join(f'{50+i} {60+i} f{i}' for i in range(1,9))
        for view in (1,3,2)))
    head='7 1 2 3 4 5 6 1'
    xyz=[f'{i} 0.1 {i+1} 0.1 {i+2} 0.1 '+' '.join(['0']*13) for i in range(7)]
    pixels=[f'{view} $000000FF {100+i} {200+i}' for view in (3,1,2) for i in range(7)]
    (folder/'070_008_e1t01.txt').write_text('\n'.join([head,*xyz,*pixels,'2026.06.24 11:40:06'])+'\n')
    return root


def test_public_count_has_no_source_coordinates_and_does_not_invent_masks(tmp_path):
    bmps,labels=fixture(tmp_path)
    pub,private=inventory_sources(bmp_root=bmps,crosses=labels)
    stats=pub['bmp_crosses']
    assert stats['valid_bmps']==2 and stats['matched_bmp_with_crosses']==1
    assert stats['cross_label_frames_missing_unambiguous_bmp']==1
    assert stats['start_marks_on_matching_valid_bmps']==1
    assert stats['fiducial_marks_on_matching_valid_bmps']==1
    assert not pub['H50Z_crosses_are_particle_masks'] and pub['native_physics_GT_count']==0
    assert not pub['physical_auto_joins_authorized']
    assert 'TEST_005.bmp' not in json.dumps(pub)
    assert 'TEST_005.bmp' in json.dumps(private)


def test_duplicates_and_wrong_size_fail_closed(tmp_path):
    folder,labels=fixture(tmp_path)
    sub=folder/'repeat';sub.mkdir()
    assert cv2.imwrite(str(sub/'TEST_005.bmp'),np.full((320,320),222,np.uint8))
    public,_=inventory_bmps(folder,labels_json=labels)
    assert public['ambiguous_names_count']==1
    assert public['matched_bmp_with_crosses']==0
    (sub/'TEST_005.bmp').unlink()
    data=json.loads(labels.read_text());data['TEST_005.bmp']['original_width']=999
    labels.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='shape differs'):inventory_bmps(folder,labels_json=labels)


def test_native_points_cannot_be_labelled_as_h50z_gt(tmp_path):
    folder,labels=fixture(tmp_path)
    measured=fake_film(tmp_path/'070')
    pub,private=inventory_sources(bmp_root=folder,crosses=labels,visualmetro=measured)
    counts=pub['visualmetro_sparse']
    assert counts['frames_complete_used']==1 and counts['tracks_used']==1
    assert counts['spatial_points_used']==7 and counts['stereo_pixel_measurements_used']==21
    assert counts['verified_BMP_view_correspondences']==0
    assert pub['auto_paired_070_with_H50Z'] is False
    assert '070_008' not in json.dumps(pub)
    assert '070_008' in json.dumps(private)


def test_manual_stereo_mapping_requires_explicit_three_distinct_views(tmp_path):
    folder,_=fixture(tmp_path)
    frames=fake_film(tmp_path/'070')
    for v in (1,2,3):
        assert cv2.imwrite(str(folder/f'VIEW_{v}.bmp'),np.full((320,320),200,np.uint8))
    mapping={'schema':'ANZA_LIRA_OPERATOR_STEREO_IMAGE_BINDINGS_V1',
             'bindings':[{'frame_id':'070_008','physical_run_id':'PRIVATE_RUN_A',
               'verified_by':'operator-qa-reference','operator_declares_same_physical_frame_and_view':True,
               'views':{str(v):f'VIEW_{v}.bmp' for v in (1,2,3)}}]}
    p=tmp_path/'PRIVATE_BINDING.json';p.write_text(json.dumps(mapping))
    pub,private=inventory_sources(bmp_root=folder,visualmetro=frames,stereo_map=p)
    assert pub['visualmetro_sparse']['operator_declared_paired_stereo_frames']==1
    assert not pub['visualmetro_sparse']['calibration_and_camera_projection_validated']
    assert 'PRIVATE_RUN_A' not in json.dumps(pub)
    assert 'PRIVATE_RUN_A' in json.dumps(private)
    mapping['bindings'][0]['operator_declares_same_physical_frame_and_view']=False
    p.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='explicit operator'):inventory_sources(
        bmp_root=folder,visualmetro=frames,stereo_map=p)


def test_cli_outputs_public_only_and_private_preview(tmp_path):
    bmps,labels=fixture(tmp_path)
    public=tmp_path/'public_summary.json'
    secret=tmp_path/'private'/'index.json'
    images=tmp_path/'private'/'images'
    main(['--bmp-root',str(bmps),'--crosses',str(labels),
          '--public-summary',str(public),'--private-index',str(secret),
          '--private-cross-preview',str(images),'--tile-size','128'])
    assert public.exists() and secret.exists()
    assert 'TEST_005.bmp' not in public.read_text()
    assert 'TEST_005.bmp' in secret.read_text()
    assert secret.stat().st_mode & 0o777 == 0o600
    assert len(list(images.glob('*.png')))==1
    with pytest.raises(FileExistsError):main(['--bmp-root',str(bmps),'--public-summary',str(public)])


def test_unlabeled_bmps_cannot_be_claimed_as_labels(tmp_path):
    bmps,labels=fixture(tmp_path)
    stats,_=inventory_sources(bmp_root=bmps)
    assert stats['bmp_crosses']['matched_bmp_with_crosses']==0
    assert stats['native_physics_GT_count']==0
    with pytest.raises(ValueError):inventory_sources(crosses=labels)
    with pytest.raises(ValueError):inventory_sources()
