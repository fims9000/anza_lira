import copy
import hashlib
import numpy as np
import cv2
import pytest

from bubble_chamber.track_mask_review import propose_tile_segments,convert_review_to_sparse_mask,review_html,SCHEMA


def _sample():
 image=np.full((192,192),220,np.uint8)
 cv2.line(image,(12,75),(179,75),35,2)
 source={'label':'start','parent':'photo.bmp','image_sha256':'f'*64,
         'roi_original_xyxy':[250,100,442,292],
         'tile_sha256':'a'*64,'center_tile_xy':[96,75],
         'source_index':0}
 return image,source


def test_no_pseudolabel_is_training_truth_before_human_review():
 image,source=_sample()
 p=propose_tile_segments(image,source)
 assert p['schema']==SCHEMA
 assert p['automatic_labels_count']==0
 assert p['candidates']
 empty={'schema':SCHEMA,'proposal_digest':p['proposal_digest'],'judgments':[]}
 target,stats=convert_review_to_sparse_mask(p,empty)
 assert np.all(target==255)
 assert stats['positive_centerline_pixels']==0
 assert stats['trainable_bce'] is False


def test_explicit_human_track_and_artifact_roundtrip():
 image,source=_sample()
 p=propose_tile_segments(image,source)
 labels=[{'candidate_id':p['candidates'][0]['id'],'decision':'track_centerline'}]
 review={'schema':SCHEMA,'proposal_digest':p['proposal_digest'],'judgments':labels}
 mask,stats=convert_review_to_sparse_mask(p,review)
 assert int((mask==1).sum())>0
 assert stats['negative_certified_artifact_pixels']==0
 assert stats['trainable_bce'] is False
 assert np.any(mask==255)
 bad=copy.deepcopy(review);bad['proposal_digest']='bad'
 with pytest.raises(ValueError,match='digest mismatch'):
  convert_review_to_sparse_mask(p,bad)
 bad=copy.deepcopy(review);bad['judgments']*=2
 with pytest.raises(ValueError,match='duplicate'):
  convert_review_to_sparse_mask(p,bad)


def test_reject_proposal_mutation_and_bad_class():
 image,source=_sample();p=propose_tile_segments(image,source)
 invalid=copy.deepcopy(p);invalid['candidates'][0]['centerline_local_xy_px'][0][0]+=1
 review={'schema':SCHEMA,'proposal_digest':p['proposal_digest'],'judgments':[]}
 with pytest.raises(ValueError,match='modified'):
  convert_review_to_sparse_mask(invalid,review)
 review['judgments']=[{'candidate_id':p['candidates'][0]['id'],'decision':'different_track'}]
 with pytest.raises(ValueError,match='invalid decision'):
  convert_review_to_sparse_mask(p,review)


def test_html_binds_exact_image_and_no_network():
 image,source=_sample();p=propose_tile_segments(image,source)
 html=review_html([p],[image])
 assert 'data:image/png;base64,' in html
 assert 'URL.createObjectURL' in html
 assert '255 is UNLABELLED' not in html
 with pytest.raises(ValueError,match='differs'):
  review_html([p],[255-image])
