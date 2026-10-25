import json
import pytest
from bubble_chamber.s33_evidence_review import inject_evidence


def fixture():
    drafts=[{'draft_id':f'd{i}','source_method':'classical'} for i in range(133)]
    machine={'schema':'s31_machine_first_h50z_draft_v1','physical_joins_authorized':False,
             'photos':[{'crop_id':'crop01','photo_id':'005','drafts':drafts}]}
    s='<html><body><section id="s31-panel"></section><script id="s31-draft-payload" type="application/json">'+json.dumps(machine)+'</script></body></html>'
    ev={'schema':'s33_uncalibrated_azunet_candidate_evidence_v1',
        'physics_auto_join_authorized':False,'merged_reviewer_drafts':133,
        'true_expert_H50Z_track_labels_added':0,'rows':[{'crop_id':'crop01','source_photo':'005',
         'draft_id':f'd{i}','source_method':'classical',
         'AZUNet_quantile':{'status':'UNVERIFIED_MODEL_EVIDENCE','median_center_minus_flank':.1},
         'AZUNet_local':{'status':'UNVERIFIED_MODEL_EVIDENCE','median_center_minus_flank':.07},
         'cross_normalization_disagreement_UNCALIBRATED':.03,
         'expert_approved':False,'physics_auto_merge_allowed':False} for i in range(133)]}
    return s,ev


def test_overlay_strictly_preserves_original_manual_review_document():
    html,e=fixture();merged=inject_evidence(html,e)
    assert 's33-evidence-panel' in merged
    assert 's33-evidence-source' in merged
    assert 'не являются вероятностями' in merged
    assert html.split('<script id="s31-draft-payload"')[1].split('</script>')[0] in merged
    with pytest.raises(ValueError,match='already injected'):
        inject_evidence(merged,e)


def test_reject_promoted_or_foreign_machine_evidence():
    h,e=fixture();e['rows'][0]['expert_approved']=True
    with pytest.raises(ValueError,match='cannot approve'):inject_evidence(h,e)
    h,e=fixture();e['rows'][0]['crop_id']='wrong'
    with pytest.raises(ValueError,match='Foreign'):inject_evidence(h,e)
    h,e=fixture();e['rows'].pop()
    with pytest.raises(ValueError,match='repeated'):inject_evidence(h,e)
