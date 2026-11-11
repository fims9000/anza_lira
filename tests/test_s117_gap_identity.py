"""S117 identity association tests; artificial photos are NOT real-data accuracy."""
import numpy as np
import pytest
from bubble_chamber.s117_gap_identity import GapConfig, associate_long_gaps


def line(name, x0, x1, y=50):
    return {'candidate_id': name,
            'observed_pixel_path_xy': [[x, y] for x in range(x0, x1 + 1)]}


def photo(response_value=0):
    gray = np.full((150, 550), 190, np.uint8)
    response = np.full(gray.shape, response_value, np.float32)
    return gray, response


def result(curves, response_value=0, **kw):
    gray, response = photo(response_value)
    return associate_long_gaps(curves, gray, response=response, **kw)


def test_long_dotted_same_line_has_one_group_without_invented_observations():
    one = line('one', 20, 80)
    two = line('two', 330, 410)
    out = result([one, two], 15)
    assert out['report']['accepted_provisional_links'] == 1, out['report']
    assert out['report']['output_group_count'] == 1
    assert out['links'][0]['gap_px'] == 250
    assert out['links'][0]['observed_connector_pixels'] == 0
    assert out['groups'][0]['confirmed_particle_id'] is None


def test_blank_gap_remains_review_only_and_separate_groups():
    out = result([line('A', 20, 80), line('B', 330, 410)], 0)
    assert out['report']['accepted_provisional_links'] == 0
    assert out['report']['output_group_count'] == 2
    assert out['report']['review_links'] >= 1


def test_ambiguous_parallel_endpoints_abstain():
    out = result([line('a', 20, 80, 50), line('b', 240, 300, 50),
                  line('c', 240, 300, 52)], 15)
    assert out['report']['accepted_provisional_links'] == 0


def test_weak_alignment_not_auto_grouped_even_with_bright_response():
    a = line('a', 20, 80, 40)
    b = {'candidate_id': 'b', 'observed_pixel_path_xy': [[220, y] for y in range(40, 101)]}
    out = result([a, b], 20)
    assert out['report']['accepted_provisional_links'] == 0


def test_reordered_input_has_stable_group_ids_and_link_ids():
    curves = [line('x', 20, 80), line('y', 330, 410)]
    a = result(curves, 15)
    b = result(list(reversed(curves)), 15)
    assert [g['group_id'] for g in a['groups']] == [g['group_id'] for g in b['groups']]
    assert {x['candidate_id'] for x in a['links']} == {x['candidate_id'] for x in b['links']}


def test_no_geometric_join_without_photo_or_with_blocker():
    curves = [line('x', 20, 80), line('y', 330, 410)]
    gray, response = photo(20)
    blocked = np.zeros(gray.shape, bool)
    blocked[50, 150:200] = True
    out = associate_long_gaps(curves, gray, response=response, blocked=blocked)
    assert out['report']['accepted_provisional_links'] == 0


def test_forced_expert_link_preserves_provenance():
    curves = [line('x', 20, 80), line('y', 330, 410)]
    baseline = result(curves, 0)
    key = baseline['links'][0]['candidate_id']
    out = result(curves, 0, decisions={key: 'same'})
    assert out['report']['expert_accepted_links'] == 1
    assert out['report']['added_observed_pixel_count'] == 0
    assert out['links'][0]['decision'] == 'EXPERT_CONFIRMED_SAME_LINK'


def test_reject_bad_ids_and_bad_response():
    gray, response = photo()
    with pytest.raises(ValueError, match='duplicate'):
        associate_long_gaps([line('x', 20, 80), line('x', 330, 410)], gray, response=response)
    with pytest.raises(ValueError, match='Directional response'):
        associate_long_gaps([line('x', 20, 80)], gray, response=np.zeros((5, 5)))


def test_s109_input_schema_supported():
    curves = [{'curve_id':'a','sampled_xy':[[x,50] for x in range(20,81)],'source_anchored':True},
              {'curve_id':'b','sampled_xy':[[x,50] for x in range(330,411)],'source_anchored':True}]
    out=result(curves,15)
    assert out['report']['accepted_provisional_links']==1


def test_inference_never_reads_manual_gt_labels():
    curves=[line('a',20,80),line('b',330,410)]
    curves[0]['manual_track_id']='ground_truth_x'
    curves[1]['manual_track_id']='ground_truth_y'
    a=result(curves,15)
    for c in curves:
        c['manual_track_id']='ground_truth_z'
    b=result(curves,15)
    assert a==b


def test_nearly_parallel_offset_cannot_auto_jump_even_if_cosine_is_high():
    # A 25 px side shift over 350 px produces cosine ~0.997, but is NOT a
    # proof of one trajectory: it could jump to a neighbouring parallel line.
    out = result([line('a', 20, 80, 50), line('b', 430, 490, 75)], 20)
    assert out['report']['accepted_provisional_links'] == 0
    assert any(e['decision'] == 'REVIEW_ONLY_PARALLEL_OFFSET_OR_CURVATURE' for e in out['links'])


def test_unobserved_s109_endpoint_cannot_auto_link():
    a = {'curve_id': 'a', 'sampled_xy': [[x,50] for x in range(20,81)],
         'source_anchored': True, 'sampled_source_supported':[True]*51+[False]*10}
    b = {'curve_id':'b','sampled_xy':[[x,50] for x in range(330,411)],
         'source_anchored':True,'sampled_source_supported':[False]*10+[True]*71}
    out = result([a,b],20)
    assert out['report']['accepted_provisional_links'] == 0


def test_multiscale_tangent_vetoes_junction_switch_without_deleting_source():
    # The last 18 px is straight and aligns perfectly to B, yet A changes
    # from vertical to horizontal within 70 px: a source-line junction risk.
    a={'candidate_id':'a','observed_pixel_path_xy':(
        [[80,y] for y in range(20,91)] + [[x,90] for x in range(81,106)])}
    b=line('b',125,210,y=90)
    out=result([a,b],20)
    assert out['report']['accepted_provisional_links']==0
    assert out['report']['output_group_count']==2
    assert any(e['decision']=='REVIEW_ONLY_MULTISCALE_DIRECTION_CHANGE' for e in out['links'])
    assert all(e['observed_connector_pixels']==0 for e in out['links'])
