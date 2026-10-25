import pytest,json
from bubble_chamber.s27_review_priority import rank_questions,embed_priorities

def sample_rows():
    result=[]
    for photo in ('005','006'):
        for i in range(3):
            id=f'H50Z_030_01_{photo}_mined{i:02d}'
            result.append({'source':{'crop_id':id,'photo_id':photo,'review_stratum':('dense','quiet','fiducial_control')[i],
                                      'source_crop_sha256':'a'*64},
                           'mixed_model_candidates_NO_TRUTH':{'unverified_neural_disagreement_fraction':.01*(i+1),
                                                            'unverified_consensus_fraction':.2,'unverified_consensus_sato_overlap_fraction':.01*(i+1)},
                           'original_score_to_overlap':{name:{'candidate_difference':{'candidate_pixel_fraction_changed':.01*i},
                                'mean_boundary_jump_reduction_fraction':.4+.1*i} for name in ('unet','azunet','guided_az')}})
    return result

def test_balanced_two_photo_queue_without_truth():
    rows=sample_rows();a=rank_questions(rows)
    assert len(a['questions'])==6
    assert len(set(q['crop_id'] for q in a['questions']))==6
    assert a['no_auto_join_or_physical_output'] is True
    assert {q['photo'] for q in a['questions'][:3]}=={'005','006'}
    assert all(q['human_verified_line_pixels']==0 for q in a['questions'])
    assert all(q['automatic_physics_merges']==0 for q in a['questions'])
    assert {'dense','quiet','fiducial_control'}<=set(q['stratum'] for q in a['questions'])

def test_inject_preserves_S26_export_contract_and_checksum():
    rows=sample_rows();ranking=rank_questions(rows)
    data={'schema':'s26_review_visual_payload_v1','photos':[{'crop_id':r['source']['crop_id'],
                    'sha256':'a'*64,'image_modes':{'original':'data:image/png;base64,AAAA'}} for r in rows]}
    html='<html><body><div class="tools"><label>Фрагмент</label></div><script type="application/json" id="payload">'+json.dumps(data)+'</script></body></html>'
    out=embed_priorities(html,ranking)
    assert 's27-review-next' in out
    assert 's26_review_visual_payload_v1' in out
    assert out.count('id="payload"')==1
    ranking['questions'][0]['source_crop_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA'):embed_priorities(html,ranking)
