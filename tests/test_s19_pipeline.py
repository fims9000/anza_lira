import json
from pathlib import Path
import pytest
from tests.test_s19_real_review import fixture
from bubble_chamber.s19_real_review import load_package,empty_review
from scripts.research.bubble_chamber.s19_run_small_label_pipeline import run_pipeline


def test_no_expert_review_means_zero_real_training(tmp_path):
    p,root=fixture(tmp_path)
    package=load_package(p,root)
    review=tmp_path/'empty.json';review.write_text(json.dumps(empty_review(package)))
    out=tmp_path/'out'
    result=run_pipeline(proposals=p,images=root,reviews=review,out_dir=out,epochs=1)
    assert result['status']=='NEEDS_MORE_CONFIRMED_EXPERT_MARKS'
    assert result['trained_models']=={}
    assert result['graph']['human_confirmed_pairs']==0
    assert result['real_automatic_track_merges_allowed'] is False
    assert not (out/'azunet/best.pt').exists()


def test_with_fixture_expert_decisions_runs_original_both_architectures(tmp_path):
    p,root=fixture(tmp_path)
    package=load_package(p,root)
    reviews=empty_review(package)
    for tile, entry in zip(package['tiles'],reviews['entries']):
        ident=tile['tile_id']
        entry['strokes']=[{'id':f'manual:{ident}:0','label':'track_centerline',
                           'radius_px':1,'points_xy':[[5,14],[9,14],[15,14]]},
                          {'id':f'manual:{ident}:1','label':'certified_artifact',
                           'radius_px':1,'points_xy':[[5,47],[9,47],[15,47]]}]
    reviewed=tmp_path/'reviewed.json';reviewed.write_text(json.dumps(reviews))
    out=tmp_path/'trained'
    r=run_pipeline(proposals=p,images=root,reviews=reviewed,out_dir=out,epochs=1,widths=(4,4,4,4),rules=2,patch_size=32)
    assert r['status']=='REAL_REVIEW_TRAINING_COMPLETED_PROVISIONAL_ONLY'
    assert r['graph']['status']=='INSUFFICIENT_EXPERT_RELATION_LABELS_ABSTAIN'
    assert (out/'azunet/best.pt').is_file()
    assert (out/'unet/best.pt').is_file()
    assert r['trained_models']['azunet']['real_full_image_pixel_dice']=='NOT_MEASURED'
    assert r['real_automatic_track_merges_allowed'] is False
    assert r['pseudo_teacher_used_for_training'] is False
