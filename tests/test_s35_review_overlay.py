"""S35 UI overlay must not certify candidate pairs or modify S31 source state."""
import json
import pytest
from bubble_chamber.s35_review_overlay import overlay_s35_review


def minimal():
    rows=[];pairs=[]
    for i in range(28):
        rid=f'relation_{i:02d}';aid=f'draft_{i:03d}a';bid=f'draft_{i:03d}b'
        pairs.append({'relation_id':rid,'candidate_a':aid,'candidate_b':bid})
        rows.append({'relation_id':rid,'crop_id':'H50Z_030_01_005_fake','photo_id':'005',
                     'candidate_a':aid,'candidate_b':bid,'status':'UNKNOWN_REVIEW_REQUIRED',
                     'physics_auto_join_allowed':False,'human_expert_label':None,
                     'local_vs_ANZA_score_disagreement':0.1,
                     'measured_endpoint_gap_px':9.5,
                     'outgoing_tangent_a':.95,'outgoing_tangent_b':.98,
                     'provisional_scores_NOT_PHYSICAL_PROBABILITIES':{
                     'geometry_UNVERIFIED_SYNTHETIC_SCORER':.45,
                     'geometry+local_UNVERIFIED_SYNTHETIC_SCORER':.60,
                     'geometry+AZUNet_UNVERIFIED_SYNTHETIC_SCORER':.55}})
    pack={'schema':'s31_machine_first_h50z_draft_v1','physical_joins_authorized':False,
          'photos':[{'crop_id':'H50Z_030_01_005_fake','photo_id':'005','relation_drafts':pairs}]}
    html='<html><body><section id="s31-panel"></section><script id="s31-draft-payload" type="application/json">'+json.dumps(pack)+'</script></body></html>'
    x={'schema':'s35_private_real_h50z_pair_review_evidence_v1',
       'physics_auto_join_allowed':False,'human_confirmed_same_different_pairs':0,
       'rows':rows}
    return html,x


def test_overlay_keeps_original_export_and_source_untouched():
    html,x=minimal();new=overlay_s35_review(html,x)
    assert 's35-next-uncertain' in new and 'НЕ являются вероятностями' in new
    assert html.split('<script id="s31-draft-payload"')[1].split('</script>')[0] in new
    assert 'physics_auto_join_authorized' not in new.split('s35-unverified-source')[1]
    with pytest.raises(ValueError):overlay_s35_review(new,x)


def test_rejects_faked_expert_decision_and_wrong_source_identity():
    html,x=minimal();x['rows'][0]['human_expert_label']='same'
    with pytest.raises(ValueError,match='Unsafe'):overlay_s35_review(html,x)
    html,x=minimal();x['rows'][0]['crop_id']='other'
    with pytest.raises(ValueError,match='Foreign'):overlay_s35_review(html,x)
    html,x=minimal();x['rows'].pop()
    with pytest.raises(ValueError,match='Missing'):overlay_s35_review(html,x)
