"""S54 source-context parity and synthetic OOD safety gates."""
import cv2
import numpy as np
import pytest

from bubble_chamber.s54_context_ood import (
    source_context_crop,calibration_ood_threshold,score_context_aligned,
)
from bubble_chamber.s43_artifact_triage import FEATURE_NAMES as OLD
from bubble_chamber.s53_bead_texture import FEATURE_NAMES as NEW, SCHEMA


def _path(x0=30.,y0=70.,extent=120.):
    return np.column_stack((np.linspace(x0,x0+extent,42),
                           np.linspace(y0,y0+3.,42)))


def _model(target,extended):
    names=list(OLD)+(list(NEW) if extended else [])
    return dict(schema=SCHEMA,synthetic_only=True,
      arm='S43_plus_bead_texture_S53' if extended else 'existing_S43',
      feature_names=names,source_photo='006',test_photo=target,
      native_physics_classes_known=False,native_auto_join_allowed=False,
      scaler_mean=[0.]*len(names),scaler_scale=[1.]*len(names),
      coefficients=[0.]*len(names),intercept=0.)


def test_crop_is_lossless_with_exact_translated_coordinates():
    gray=np.random.default_rng(41).integers(0,256,(512,512),dtype='uint8')
    path=_path(207,220,140)
    value=source_context_crop(gray,path)
    assert value is not None
    crop,points,(x,y)=value
    assert crop.shape==(192,192)
    assert np.array_equal(crop,gray[y:y+192,x:x+192])
    assert np.allclose(points+np.array([x,y]),path)
    assert np.all(points>=12) and np.all(points<=180)


def test_nonfitting_or_edge_paths_abstain_not_fake_a_negative():
    a=np.zeros((512,512),dtype='uint8')
    assert source_context_crop(a,_path(30,80,210)) is None
    assert source_context_crop(a,_path(1,80,120)) is None
    with pytest.raises(ValueError):source_context_crop(a.astype(float),_path())
    with pytest.raises(ValueError):source_context_crop(a,_path(480,80,120))


def test_calibration_threshold_uses_only_supplied_features():
    x=np.random.default_rng(40).standard_normal((64,14))
    model=_model('005',False)
    threshold=calibration_ood_threshold(x,model,quantile=.99)
    assert threshold==pytest.approx(np.quantile(np.max(np.abs(x),axis=1),.99))
    with pytest.raises(ValueError):calibration_ood_threshold(x[:2],model)


def test_native_scores_are_never_physical_probs_or_autojoin():
    a=np.full((512,512),185,dtype='uint8')
    cv2.line(a,(30,80),(150,83),50,2)
    c=[{'draft_id':'one','points_xy':_path().tolist()},
       {'draft_id':'border','points_xy':_path(1.,78.).tolist()}]
    models={'existing_S43':_model('005',False),
            'S43_plus_bead_texture_S53':_model('005',True)}
    thresh={k:1.e9 for k in models}
    results=score_context_aligned(a,c,photo='005',checkpoints=models,thresholds=thresh)
    assert len(results)==len(c)
    assert results[0]['status']=='UNKNOWN_WITHIN_SIMULATOR_SUPPORT_ONLY'
    assert results[1]['status']=='UNKNOWN_UNSCORABLE_CONTEXT'
    assert all(r['native_particle_identity']=='UNKNOWN' and r['candidate_preserved']
               and not r['physical_auto_join_authorized'] for r in results)
    assert all('native_particle_probability' not in r for r in results)
    wrong={k:dict(m) for k,m in models.items()}
    wrong['existing_S43']['source_photo']='005'
    with pytest.raises(ValueError):score_context_aligned(a,c[:1],photo='005',checkpoints=wrong,thresholds=thresh)
