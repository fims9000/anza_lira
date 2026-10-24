from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
from bubble_chamber.pilot import PilotConfig
from bubble_chamber.s14_fewshot_graph import json_model_parameters,fit_fewshot

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s14_predict_new_photo.py'
spec=importlib.util.spec_from_file_location('s14_predict',SCRIPT)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def toy_pairs():
    out=[]
    for i in range(30):
        x=np.arange(7,dtype=float)
        label='same_track' if i%2==0 else 'different_track'
        out.append({'film_id':'toy','run_id':'run1','frame_id':f'f{i//3}',
           'scene_id':f'f{i//3}','left_segment':i*2,'right_segment':i*2+1,
           'left_points_xy':np.column_stack((x, .02*x*x)).tolist(),
           'right_points_xy':np.column_stack((x+8, .02*(x+8)**2+(0 if i%2==0 else 12))).tolist(),
           'source_sha256':'1'*64,'expert_label':label})
    return out


def test_inference_without_human_never_fabricates_certified_tracks():
    model,stat=fit_fewshot(toy_pairs(),source_run_id='run1')
    params=json_model_parameters(model)
    image=np.full((128,128),200,dtype=np.uint8)
    image[32:95,32]=20;image[35:95,72]=20
    original,r,overlay=m.predict_image(image,params,film_id='film2',run_id='run2',frame_id='frame7',
                                        config=PilotConfig(segmentation_method='contrast',min_segment_length_px=12))
    assert r['real_auto_merged_fragments']==0
    assert r['calibration_receipt_usable'] is False
    assert all(not x['physical_join_certified'] for x in r['candidate_links'])
    assert overlay.shape[:2]==image.shape
    with pytest.raises(ValueError):m.predict_image(image,params,film_id='',run_id='run2',frame_id='f')
