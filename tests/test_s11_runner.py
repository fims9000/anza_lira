from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.research.bubble_chamber import s7_original_retrain as s7
from scripts.research.bubble_chamber import s11_train_axial_topology as s11


def cfg(seed=1234):
    return s7.RunConfig(source='procedural',seed=seed,data_seed=20261024,epochs=1,
       size=32,batch_size=2,widths=(4,4,4,4),num_rules=2,
       train_limit=8,val_limit=6,test_limit=8,steps_per_epoch=2,
       cpu_threads=2,az_residual=True,robust_train=True)


def test_source_split_heldout_and_fixed_by_data_seed():
    a,m=s11.prepare_source(cfg(10),None)
    b,n=s11.prepare_source(cfg(11),None)
    assert m['split_fingerprints']==n['split_fingerprints']
    assert m['train_test_disjoint']
    assert m['all_pixel_masks_simulated']
    assert all(np.array_equal(u[0],v[0]) for u,v in zip(a['test'],b['test']))
    sets=[{row[2] for row in a[p]} for p in ['train','val','test']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


def test_ablation_runner_two_real_original_architectures(tmp_path):
    c=cfg()
    samples,_=s11.prepare_source(c,None)
    a,comp=s11.train_arms(c,samples,tmp_path,arm_ids=('unet_base','az_cldice_axis'),
                          cldice_iterations=2,axis_iterations=2)
    assert set(a)=={'unet_base','az_cldice_axis'}
    assert (tmp_path/'az_cldice_axis'/'best_final.pt').is_file()
    assert (tmp_path/'unet_base'/'s11_arm.json').is_file()
    assert 'clean_az_cldice_axis_minus_unet_base' in comp
    assert a['az_cldice_axis']['evaluation']['clean']['test_images']==c.test_limit
    assert comp['clean_az_cldice_axis_minus_unet_base']['dice_paired_frame']['frames']==c.test_limit


def test_invalid_arm_and_weights_fail():
    assert [arm.name for arm in s11.ARMS]==['unet_base','unet_cldice','az_base','az_cldice','az_cldice_axis']
    with pytest.raises(ValueError):
        s11._bootstrap_frame_cldice(np.zeros(2),np.zeros(3),['a','b'],seed=1)
