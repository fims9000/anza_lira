import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from bubble_chamber.stereo_dataset import build_manifest, verified_triplet, InputContractError
from bubble_chamber.pilot import PilotConfig


def sample_dataset(tmp_path, *, groups=3):
    root = tmp_path/'photos';root.mkdir()
    table = tmp_path/'triplets.csv'
    fields=['event_id','chamber','run_id','film_id','frame_id','view1','view2','view3']
    rows=[]
    for r in range(groups):
        names=[]
        for view in (1,2,3):
            name=f'run_{r}_v{view}.bmp'
            photo = np.full((128, 160), 210+r*8+view, np.uint8)
            cv2.line(photo, (15,20+view*10), (150,100+r), (30+view*10),2)
            assert cv2.imwrite(str(root/name),photo)
            names.append(name)
        rows.append({'event_id':f'event{r}','chamber':'H2' if r%2==0 else 'C3H8','run_id':f'r{r}',
                     'film_id':f'f{r}', 'frame_id':f'00{r}',
                     **{f'view{i}':names[i-1] for i in (1,2,3)}})
    with table.open('w',encoding='utf-8',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return root,table


def test_three_views_checksum_grouping_and_true_run_disjoint_split(tmp_path):
    root,table=sample_dataset(tmp_path,groups=4)
    manifest=build_manifest(table,root)
    assert manifest['schema']=='bubble_stereo_dataset_v1'
    assert manifest['split_status']=='RUN_DISJOINT_ASSIGNED'
    assert manifest['event_count']==4
    assert len({e['group_id'] for e in manifest['events']})==4
    assert {e['split'] for e in manifest['events']}=={'train','validation','test'}
    assert all(e['calibration_status']=='INCOMPLETE' for e in manifest['events'])
    views=verified_triplet(manifest,'event1')
    assert len(views)==3 and all(x.shape==(128,160) for x in views)


def test_dataset_with_one_run_does_not_fabricate_holdout(tmp_path):
    root,table=sample_dataset(tmp_path,groups=1)
    manifest=build_manifest(table,root)
    assert manifest['split_status']=='INSUFFICIENT_RUN_GROUPS_FOR_HOLDOUT'
    assert manifest['events'][0]['split']=='unassigned'


def test_photo_mutation_invalidates_manifest(tmp_path):
    root,table=sample_dataset(tmp_path,groups=1)
    manifest=build_manifest(table,root)
    path=root/'run_0_v2.bmp'
    img=cv2.imread(str(path),0);img[0,0]=0;cv2.imwrite(str(path),img)
    with pytest.raises(InputContractError,match='checksum changed'):
        verified_triplet(manifest,'event0')


def test_reused_photo_and_path_traversal_rejected(tmp_path):
    root,table=sample_dataset(tmp_path,groups=2)
    with table.open(encoding='utf-8',newline='') as fd:
        rows=list(csv.DictReader(fd))
    rows[1]['view1']=rows[0]['view2']
    with table.open('w',encoding='utf-8',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with pytest.raises(InputContractError,match='reused'):
        build_manifest(table,root)
    rows[1]['view1']='../out_of_root.bmp'
    with table.open('w',encoding='utf-8',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with pytest.raises(InputContractError,match='escapes'):
        build_manifest(table,root)


def test_end_to_end_three_independent_views_but_no_fake_3d(tmp_path):
    root,table=sample_dataset(tmp_path,groups=1)
    manifest=build_manifest(table,root)
    path=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/analyze_triplet.py'
    spec=importlib.util.spec_from_file_location('bc_analyze_triplet',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=module.analyze(manifest,'event0',tmp_path/'out',PilotConfig(segmentation_method='frangi'))
    assert out['auto_joins']==0
    assert out['physical_result']=='NO_STEREO_MATCHING_NO_3D_NO_MOMENTUM'
    assert len(out['views'])==3
    assert (tmp_path/'out/triplet_result.json').is_file()
    assert all((tmp_path/'out'/f'view{i}'/'result.json').is_file() for i in (1,2,3))


def test_calibration_reference_checksum_is_pinned(tmp_path):
    root, table = sample_dataset(tmp_path, groups=1)
    (root/'optics.txt').write_text('VERSION1')
    (root/'field.txt').write_text('B(x,y,z) TABULATED')
    (root/'fids.txt').write_text('FIDS')
    with table.open(encoding='utf-8', newline='') as fd:
        rows = list(csv.DictReader(fd))
    rows[0].update({'optical_file':'optics.txt','fiducial_file':'fids.txt','magnetic_field_file':'field.txt'})
    with table.open('w',encoding='utf-8',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    manifest=build_manifest(table,root)
    assert manifest['events'][0]['calibration_status']=='FILES_PRESENT_UNVERIFIED'
    assert len(verified_triplet(manifest,'event0'))==3
    (root/'field.txt').write_text('B CHANGED WITHOUT MEASUREMENT')
    with pytest.raises(InputContractError,match='calibration reference checksum changed'):
        verified_triplet(manifest,'event0')
