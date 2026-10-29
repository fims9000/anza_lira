"""Only synthetic BMP drawings. Never publish real H50Z pixels in tests."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.research.bubble_chamber.physicists_batch_preview import run,_centers


def fake_image(tmp_path):
    root=tmp_path/'bmps';root.mkdir()
    gray=np.full((320,320),208,np.uint8)
    cv2.line(gray,(40,125),(284,137),30,2)
    cv2.line(gray,(38,205),(286,195),44,2)
    assert cv2.imwrite(str(root/'FAKE.bmp'),gray)
    labels={'FAKE.bmp':{'original_width':320,'original_height':320,'keypoints_count':2,
                       'keypoints':[{'label':'start','x_px':142,'y_px':127},
                                    {'label':'f1','x_px':150,'y_px':202}]}}
    p=tmp_path/'labels.json';p.write_text(json.dumps(labels))
    return root,p


def test_private_real_pixel_processing_path_cannot_auto_certify_joins(tmp_path):
    source,marks=fake_image(tmp_path)
    out=tmp_path/'private_preview'
    result=run(bmp_root=source,crosses=marks,out_dir=out,
               max_images=1,per_image=2,roi_mode='both')
    assert result['crops_previewed'] in (1,2)
    assert len(list((out/'images').glob('*.png')))==3*result['crops_previewed']
    assert result['native_particle_GT_count']==0 and not result['physical_auto_joins_authorized']
    info=json.loads((out/'PRIVATE_MANIFEST.json').read_text())
    assert info['machine_outputs_all_UNKNOWN']
    assert len(info['sources'])==result['crops_previewed']
    assert 'FAKE.bmp' not in json.dumps(result)
    html=(out/'index.html').read_text()
    assert 'не предсказание обученной AZUNet' in html
    assert 'private_00001_original.png' in html
    with pytest.raises(FileExistsError):
        run(bmp_root=source,crosses=marks,out_dir=out)


def test_unannotated_BMP_still_has_deterministic_grid_and_no_GT(tmp_path):
    source,marks=fake_image(tmp_path)
    result=run(bmp_root=source,crosses=None,out_dir=tmp_path/'grid',
               max_images=1,per_image=4,roi_mode='grid')
    assert result['crops_previewed']==4
    assert len(_centers(320,320,4))==4
    assert result['native_particle_GT_count']==0
    with pytest.raises(ValueError):
        run(bmp_root=source,crosses=None,out_dir=tmp_path/'bad',roi_mode='wrong')


def test_public_checkout_must_not_contain_private_overlay(tmp_path):
    source,marks=fake_image(tmp_path)
    root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError,match='source repository'):
        run(bmp_root=source,crosses=marks,out_dir=root/'PRIVATE_NOT_ALLOWED')
