from pathlib import Path
import cv2
import numpy as np
import pytest
from types import SimpleNamespace
from scripts.research.bubble_chamber import s85_private_review_gallery as review


def curve(y):
    return np.column_stack((np.linspace(31,151,32),np.full(32,y)))


def test_private_overlay_is_2d_and_proposals_are_unknown():
    image=np.full((192,192),204,np.uint8)
    line,extra=curve(75),curve(98)
    edge=SimpleNamespace(i=0,j=1,end_i=-1,end_j=0)
    pred={'polylines':[line,extra],'chosen':[edge]}
    result=review.render_overlay(image,pred,title='S85 TEST SYNTHETIC ONLY',source_count=1)
    assert result.shape==(576,576,3) and result.dtype==np.uint8
    assert np.unique(result.reshape(-1,3),axis=0).shape[0]>20
    assert not np.all(result[250:400]==result[251:401])
    with pytest.raises(ValueError):review.render_overlay(image.astype(float),pred,title='fake',source_count=1)
    with pytest.raises(ValueError):review.render_overlay(image,pred,title='fake',source_count=3)


def test_private_static_html_contains_no_urls_or_truth_claims():
    p=review.offline_html([{'photo_group':'005','variants':[
       {'image':'images/proposal_0001.jpg','display':'S72','lines':1,'extra':0,'joins':0},
       {'image':'images/proposal_0002.jpg','display':'S85','lines':2,'extra':1,'joins':1}]}])
    assert p.count('<article')==2 and p.count('class="scene"')==1
    assert 'UNKNOWN' in p and 'экспертная GT отсутствует' in p
    assert 'https://' not in p and 'http://' not in p
    assert 'src="images/proposal_0001.jpg"' in p
    assert 'data-photo="005"' in p
    assert '3D' in p and 'физическ' in p.lower()


def test_private_viewer_cannot_be_written_inside_public_research_repo(tmp_path):
    with pytest.raises(ValueError,match='MUST NOT'):
        review.build_gallery(tmp_path/'v',tmp_path/'p',tmp_path/'w',tmp_path/'c',
                             review.ROOT/'docs/bubble_chamber/PUBLIC')
    with pytest.raises(RuntimeError,match='requires authenticated'):
        review.build_gallery(tmp_path/'v',tmp_path/'p',tmp_path/'w',tmp_path/'c',
                             tmp_path/'private-preview')
