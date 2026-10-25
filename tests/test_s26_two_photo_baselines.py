"""S26 protects against invented real-photo ground truth and test leakage."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np,pytest
from scripts.research.bubble_chamber.s26_two_photo_baselines import (
  render_one,pixel_input,classical_score,select_threshold,score_measures,
  sample_two_photos)
from scripts.research.bubble_chamber.s26_guided_anza import two_channel,build
from scripts.research.bubble_chamber.s26_track_proposals import segment_consensus

EXAMPLES=Path('/mnt/data/s25_work/private/1mH2Data/S1')
BMPS=Path('/mnt/data/s25_private')

@pytest.mark.parametrize('method',['darkness','local_contrast','blackhat','clahe_blackhat','frangi','sato','meijering','steerable_ridge'])
def test_deterministic_filters_and_finite(method):
    rng=np.random.default_rng(54);im=rng.integers(35,235,(64,64),dtype='uint8')
    a=classical_score(im,method);b=classical_score(im,method)
    assert a.shape==im.shape and np.isfinite(a).all()
    np.testing.assert_allclose(a,b)


def test_genuine_photo_pixels_with_synthetic_only_annotation():
    if not EXAMPLES.is_dir() or not BMPS.is_dir():pytest.skip('private H50Z required for this test')
    ds,meta=sample_two_photos(BMPS,EXAMPLES,20261027,train_n=4,val_n=3,test_n=3)
    assert meta['train_and_validation_photo']=='005'
    assert meta['heldout_test_photo']=='006'
    assert not meta['real_native_track_labels_available']
    assert all(x.photo=='005' for x in ds['train']+ds['val'])
    assert all(x.photo=='006' for x in ds['test'])
    for e in ds['train']+ds['val']+ds['test']:
        assert e.gray.shape==(64,64)==e.sparse.shape
        assert set(np.unique(e.sparse))=={0,1,255}
        assert np.all(e.synthetic_track_mask[e.sparse==1]>0)
        assert np.all(e.synthetic_scratch_mask[e.sparse==0]>0)
        assert np.mean(e.sparse==255)>.9
    assert meta['independent_physical_run'] is False


def test_local_normalization_brightness_robust_for_constant_offset():
    im=np.full((64,64),170,'uint8');cv2.line(im,(2,12),(61,53),90,2)
    shifted=im+30
    np.testing.assert_allclose(pixel_input(im,'local'),pixel_input(shifted,'local'),atol=1e-5)
    assert np.abs(pixel_input(im,'raw')-pixel_input(shifted,'raw')).mean()>.1


def test_two_channel_preserves_original_source_contract():
    im=np.full((64,64),190,'uint8');cv2.line(im,(4,40),(59,40),70,2)
    features=two_channel(im)
    assert features.shape==(2,64,64) and np.isfinite(features).all()
    for architecture in ('azunet','unet'):
        m=build(architecture)
        assert m is not None


def test_validation_only_threshold_and_false_positive_reporting():
    score=np.asarray([.99,.70,.18,.03])
    y=np.asarray([1,1,0,0])
    th=select_threshold(score,y,max_fpr=.1)
    assert th['validation_selected_only_on_photo_005']
    assert th['val_fpr']==0 and th['val_tpr']==1
    metrics=score_measures(y,score,th['threshold'])
    assert metrics['roc_auc_synthetic_marks']==1.0
    assert metrics['n_known_negative']==2
    with pytest.raises(ValueError):select_threshold(score,np.zeros(4,int))


def test_consensus_never_joins_and_quadratics_are_pixel_space_only():
    img=np.full((512,512),220,dtype=np.uint8)
    score=np.zeros(img.shape,np.float32)
    cv2.line(score,(50,100),(215,140),1.,2)
    cv2.line(score,(232,145),(290,163),1.,2)
    fragments,questions,mask=segment_consensus(img,{'unet':score,'azunet':score},
                                            {'unet':.5,'azunet':.5},min_len=15)
    assert len(fragments)>=2
    assert all(not x['human_confirmed_trace'] for x in fragments)
    assert all('quadratic_pca_local' in x for x in fragments)
    assert all(x['automatic_merge'] is False for x in questions)
    assert all(x['status']=='needs_review' for x in questions)
    assert not np.any(mask[0])


def test_one_model_false_line_cannot_enter_consensus():
    im=np.full((512,512),220,'uint8')
    a=np.zeros(im.shape,np.float32);b=np.zeros_like(a)
    cv2.line(a,(20,20),(250,20),1,3)
    f,q,m=segment_consensus(im,{'unet':a,'azunet':b},{'unet':.5,'azunet':.5})
    assert not f and not q and not np.any(m)


def test_review_html_has_no_manually_confirmed_strokes_and_valid_original_sha():
    p=Path('/mnt/data/s26_work/results/real_photo_review/S26_manual_sparse_review.html')
    if not p.exists():pytest.skip('no private visualization artifact')
    import re
    x=p.read_text()
    match=re.search(r'<script type="application/json" id="payload">(.*?)</script>',x,re.S)
    assert match
    ob=json.loads(match.group(1))
    assert ob['schema']=='s26_review_visual_payload_v1'
    assert len(ob['photos'])==9
    assert all(q['sha256']==hashlib.sha256((Path('/mnt/data/s25_work/private')/(q['crop_id']+'.png')).read_bytes()).hexdigest() for q in ob['photos'])
    assert 'true' not in [q.get('expert_approved',False) for q in ob['photos']]
    assert ob['no_expert_labels_in_predictions'] is True


def test_junction_questions_are_not_secondary_star_truth():
    from scripts.research.bubble_chamber.s26_track_proposals import uncertain_junctions
    x=np.zeros((64,64),np.uint8)
    cv2.line(x,(8,30),(54,30),1,1)
    cv2.line(x,(32,10),(32,54),1,1)
    c=uncertain_junctions(x)
    assert len(c)>=1
    assert all(r['human_verified_secondary_vertex'] is False for r in c)
    assert all('NOT_PROVEN_INTERACTION' in r['semantic_status'] for r in c)
