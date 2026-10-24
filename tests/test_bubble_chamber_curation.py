from copy import deepcopy
import numpy as np
import pytest

from bubble_chamber.curation import curate_tracks, ReviewConflictError
from bubble_chamber.review import pair_key, result_digest, SCHEMA


def _segment(i, x0, x1):
    pts=np.column_stack((np.linspace(x0,x1,14),np.full(14,30.0)))
    return {'segment_id':i,'measurement_xy_px':pts.tolist()}


def _fixture():
    ps=[_segment(0,10,30),_segment(1,44,65),_segment(2,80,105)]
    joins=[dict(segment_A=0,endpoint_A='B',segment_B=1,endpoint_B='A'),
           dict(segment_A=1,endpoint_A='B',segment_B=2,endpoint_B='A'),
           dict(segment_A=0,endpoint_A='B',segment_B=2,endpoint_B='A')]
    result={'segments':ps,'join_proposals':joins,'pipeline':'p2','source_sha256_grayscale_pixels':'synthetic_hash'}
    def bundle(labels):
        return {'schema':SCHEMA,'source_sha256_grayscale_pixels':'synthetic_hash',
                'pipeline':'p2','result_sha256':result_digest(result),'annotator':'synthetic_tester',
                'labels':[{'pair_key':pair_key(joins[i]),'label':value,'comment':''} for i,value in labels]}
    return result,joins,bundle


def test_expert_approved_chain_preserves_unobserved_gaps():
    result, joins, bundle=_fixture()
    out=curate_tracks(result,bundle([(0,'same_track'),(1,'same_track')]))
    assert out['track_group_count']==1
    assert out['approved_gap_count']==2
    track=out['tracks'][0]
    assert track['source_fragment_ids']==[0,1,2]
    assert len(track['accepted_gap_bridges'])==2
    assert [x['unobserved_gap_px'] for x in track['accepted_gap_bridges']]==[14.0,15.0]
    assert out['no_inferred_pixels_across_gaps'] is True
    assert track['quadratic_fit_of_observed_points']['residual_rms_px']<0.0001
    assert out['auto_merged_fragments']==0


def test_different_track_constraint_cannot_be_overridden_by_indirect_chain():
    result, _, bundle=_fixture()
    with pytest.raises(ReviewConflictError,match='contradicts'):
        curate_tracks(result,bundle([(0,'same_track'),(1,'same_track'),(2,'different_track')]))


def test_same_endpoint_cannot_connect_two_continuations():
    result, _, bundle=_fixture()
    with pytest.raises(ReviewConflictError,match='same endpoint'):
        curate_tracks(result,bundle([(0,'same_track'),(2,'same_track')]))


def test_unreviewed_fragments_remain_distinct_and_no_fake_3d():
    result, _, bundle=_fixture()
    out=curate_tracks(result,bundle([]))
    assert out['track_group_count']==3
    assert out['approved_gap_count']==0
    assert out['physical_result_status'].startswith('UNAVAILABLE')


def test_stale_result_digest_is_rejected():
    result, _, bundle=_fixture()
    b=bundle([(0,'same_track')]);b['result_sha256']='invalid'
    with pytest.raises(ValueError,match='result content mismatch'):
        curate_tracks(result,b)


def test_reversed_attachment_yields_correct_traversal():
    result,joins,bundle=_fixture()
    # Source segment0 is oriented backwards; connect its A to B of segment1.
    result['join_proposals']=[dict(segment_A=0,endpoint_A='A',segment_B=1,endpoint_B='B')]
    b={'schema':SCHEMA,'source_sha256_grayscale_pixels':result['source_sha256_grayscale_pixels'],
       'pipeline':result['pipeline'],'result_sha256':result_digest(result),'annotator':'human',
       'labels':[{'pair_key':pair_key(result['join_proposals'][0]),'label':'same_track','comment':''}]}
    out=curate_tracks(result,b)
    assert out['track_group_count']==2
    p=out['tracks'][0]['pieces']
    assert [i['orientation'] for i in p]==['reversed','reversed']
