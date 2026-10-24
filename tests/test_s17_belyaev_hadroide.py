"""Book equations, applicability gates, and synthetic label provenance."""
import math
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from bubble_chamber.s17_belyaev_hadroide import (
    MEDIA, critical_length_cm, phi_from_s_cm, simulate_stopping_track, fit_hadroide_a,
)
from scripts.research.bubble_chamber.s17_make_book_prior_dataset import generate, make_example
from scripts.research.bubble_chamber.s14_train_sparse_anza import read_reviewed_manifest


def test_book_tabulated_phi_h_and_a_units():
    # Paper table 3, |Hz|=18 kG: phiH values 20.112 (H2), 5.635 (C3H8).
    w=2/7
    for medium,expected in [('liquid_hydrogen',20.112),('liquid_propane',5.635)]:
        props=MEDIA[medium]
        h0=props['density_g_per_cm3']*math.sqrt(2)/(0.3*props['kappa_g_per_mev_cm2'])
        phi_h=18/h0/(1-w)
        assert abs(phi_h-expected)<0.02
    assert 3.6<critical_length_cm(1,0,'liquid_hydrogen')<3.9
    assert 1.0<critical_length_cm(1,0,'liquid_propane')<1.1


def test_simulated_curve_has_physical_origin_and_charge_symmetry():
    p=simulate_stopping_track(mass_mev=139.57,alpha_rad=.2,medium='liquid_propane',
                              mean_hz_kgauss=-18,charge_sign=1,max_s_over_a=.6)
    n=simulate_stopping_track(mass_mev=139.57,alpha_rad=.2,medium='liquid_propane',
                              mean_hz_kgauss=-18,charge_sign=-1,max_s_over_a=.6)
    assert p.xy_cm.shape==(512,2)
    assert np.allclose(p.xy_cm[0],[0,0])
    assert np.allclose(p.phi_rad,-n.phi_rad)
    assert np.allclose(p.xy_cm[:,0],n.xy_cm[:,0])
    assert np.allclose(p.xy_cm[:,1],-n.xy_cm[:,1])
    assert p.is_real_photo_label is False
    assert np.diff(p.s_cm).min()>0


def test_domain_and_calibration_abort():
    a=critical_length_cm(139.57,.2,'liquid_propane')
    with pytest.raises(ValueError,match='s < a'):
        phi_from_s_cm(np.asarray([0,a]),a,medium='liquid_propane',mean_hz_kgauss=-18,charge_sign=1)
    with pytest.raises(ValueError,match='negative'):
        phi_from_s_cm(np.asarray([0,.1]),a,medium='liquid_propane',mean_hz_kgauss=18,charge_sign=1)
    with pytest.raises(ValueError,match='angle'):
        critical_length_cm(100, math.pi/2,'liquid_hydrogen')
    out=fit_hadroide_a(s_cm=np.arange(7.),observed_phi_rad=np.zeros(7),medium='liquid_propane',
                       mean_hz_kgauss=-18,charge_sign=1,stopping_confirmed=False,
                       stereo_calibrated=False,optical_calibration_id='',field_map_id='',
                       alpha_rad=.2,max_a_cm=100)
    assert out['status'].startswith('ABSTAIN') and out['physical_mass_certified'] is False


def test_synthetic_full_roundtrip_model_fit_is_not_physical_certification():
    medium='liquid_propane'
    p=simulate_stopping_track(mass_mev=493.67,alpha_rad=.3,medium=medium,
                mean_hz_kgauss=-18,charge_sign=1,max_s_over_a=.6)
    sample=np.arange(0,512,30)
    out=fit_hadroide_a(s_cm=p.s_cm[sample],observed_phi_rad=p.phi_rad[sample],
         medium=medium,mean_hz_kgauss=-18,charge_sign=1,
         stopping_confirmed=True,stereo_calibrated=True,
         optical_calibration_id='SYNTHETIC_ONLY',field_map_id='SIMULATED_CONSTANT_FIELD',
         alpha_rad=.3,max_a_cm=1500)
    assert out['status']=='PHYSICS_PRIOR_FIT_FOR_REVIEW'
    assert abs(out['approx_mass_proxy_mev']-493.67)<.05
    assert out['physical_mass_certified'] is False


def test_generated_images_and_masks_are_explicitly_synthetic():
    rng=np.random.default_rng(4)
    im,visible,sparse,full,info=make_example(128,rng,'liquid_hydrogen')
    assert im.shape==visible.shape==sparse.shape==full.shape==(128,128)
    assert set(np.unique(sparse)).issubset({0,1,255})
    assert np.any(sparse==1) and np.any(sparse==0) and np.any(sparse==255)
    assert not np.any((sparse==1)&(visible==0))
    assert np.all(full[visible>0]==1) or np.mean(full[visible>0])>.98
    assert info['never_expert_certified'] is True
    assert info['pixel_projection'].startswith('SYNTHETIC_')


def test_s17_manifest_s14_consumable_and_test_not_training(tmp_path):
    rec=generate(tmp_path,seed=13,size=64,train=4,val=2,unlabeled=2,test=2)
    groups,meta=read_reviewed_manifest(tmp_path/'s17_book_synthetic_sparse_manifest.json')
    assert len(groups['train'])==4 and len(groups['val'])==2 and len(groups['unlabeled'])==2
    assert not any(x['frame_id'].startswith('test') for xs in groups.values() for _,_,x in xs)
    assert rec['expert_labeled_pixels']==0 and rec['synthetic_full_test_masks']==2
    manifest=json.loads((tmp_path/'s17_book_synthetic_sparse_manifest.json').read_text())
    assert manifest['dataset_origin'].startswith('SIMULATED_')
    for ent in manifest['entries']:
        assert ent['image_sha256'] and ent['source_kind'].startswith('GENERATED_')
