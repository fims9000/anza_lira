"""Tests for label integrity and data acquisition without fabricated masks."""
import csv
import json
import numpy as np
import pytest

from bubble_chamber.film_data_planner import _spread_subset, _load, audit_fiducials,build_crop_request,save_plan


def test_ransac_identifies_wrong_fiducial_name_without_relabeling(tmp_path):
    landmarks=[(100,100),(300,100),(500,100),(100,300),(500,300),(100,500),(500,500),(300,600)]
    frames={}
    for n, shift in (('001',0),('002',10)):
        pts=[]
        for i,(x,y) in enumerate(landmarks):
            xx=x+shift;yy=y-5
            if n=='002' and i==7:yy-=300
            pts.append({'label':f'f{i+1}','x_px':xx,'y_px':yy})
        pts.append({'label':'start','x_px':200,'y_px':300})
        frames[f'H50Z_030_01_{n}.bmp']={'original_width':800,'original_height':800,'keypoints_count':len(pts),'keypoints':pts}
    path=tmp_path/'labels.json';path.write_text(json.dumps(frames))
    out=audit_fiducials(path,residual_limit_px=25)
    assert 'H50Z_030_01_002.bmp' in out['outlier_id_frames']
    flagged=next(x for x in out['frames'] if x['image'].endswith('002.bmp'))
    assert flagged['outlying_fiducial_ids']==['f8']
    assert flagged['fitted_residual_px']['f8']>250


def test_duplicate_id_records_as_ambiguous(tmp_path):
    frame={'original_width':300,'original_height':300,'keypoints':[
        {'label':'f4','x_px':10,'y_px':10},{'label':'f4','x_px':40,'y_px':40},
        {'label':'f1','x_px':110,'y_px':20},{'label':'start','x_px':20,'y_px':100}]}
    frame['keypoints_count']=len(frame['keypoints']);f=tmp_path/'l.json';f.write_text(json.dumps({'H50Z_030_01_001.bmp':frame}))
    qc=audit_fiducials(f)
    assert qc['duplicate_id_frames']==['H50Z_030_01_001.bmp']
    assert qc['frames'][0]['duplicate_fiducial_ids']==['f4']


def test_spread_sampler_selects_no_more_than_target_and_spreads():
    pts=[(i,{'x_px':float(i*100),'y_px':float(200+i*30)}) for i in range(7)]
    choice=_spread_subset(pts,900,900,3)
    assert len(choice)==3
    assert len(set(i for i,_ in choice))==3
    assert choice==sorted(choice,key=lambda x:x[0])


def test_rejects_outside_labels(tmp_path):
    q={'H50Z_030_01_001.bmp':{'original_width':200,'original_height':100,
       'keypoints_count':1,'keypoints':[{'label':'start','x_px':500,'y_px':0}]}}
    p=tmp_path/'x.json';p.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='Out-of-frame'):_load(p)


def test_source_plan_with_real_manifest(tmp_path):
    from pathlib import Path
    src=Path('/mnt/data/bubble_h50z/labels.json')
    measurements=Path('/mnt/data/bubble_new_dataset')
    if not src.is_file() or not measurements.is_dir():
        pytest.skip('real source archives unavailable outside bundled environment')
    qc=audit_fiducials(src)
    plan=build_crop_request(src,measurements,available_root=Path('/mnt/data/bubble_h50z'))
    assert len(plan['h50z_image_priority'])==4
    files={e['image'] for e in plan['h50z_image_priority']}
    assert 'H50Z_030_02_705.bmp' in files
    assert 'H50Z_030_02_758.bmp' in files
    assert 'H50Z_030_01_007.bmp' in files
    assert len(plan['requests'])==4*6+3*3*3
    assert all(0<=row['local_x_px']<1024 and 0<=row['local_y_px']<1024 for row in plan['requests'])
    save_plan(plan,qc,tmp_path/'plan')
    with (tmp_path/'plan'/'requested_crops.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f)))==len(plan['requests'])
    assert len(qc['duplicate_id_frames'])==2
    # Do not demand a specific number of RANSAC outliers across the whole archive;
    # only flag suspect correspondences without modifying the source labels.
