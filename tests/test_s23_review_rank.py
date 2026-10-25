import json
from copy import deepcopy
import numpy as np
import pytest
from bubble_chamber.s23_review_rank import observed_pair_rows,rank_unlabeled_real_pairs,insert_priority_in_workbench

class FakeModel:
    def predict_proba(self,x):
        p=np.clip(.5+.12*x[:,0]-.05*x[:,15],.1,.9)
        return np.stack([1-p,p],axis=1)

def sample():
    cand=[]
    for i,dy in enumerate((5,15,25)):
        cand.append({'id':str(i),'centerline_local_xy_px':[[10+j,dy+j//5] for j in range(10)]})
    tile={'tile_id':'H50Z_030_01_005_start1_tile','source':{'parent':'H50Z_030_01_005.bmp'},
          'parent_photo':'H50Z_030_01_005.bmp','candidates':cand}
    pack={'schema':'s19_real_photo_review_package_v1','package_sha256':'f'*64,'tiles':[tile]}
    priority={'schema':'s20_real_photo_review_priority_v1','source_package_sha256':'f'*64,
          'pair_questions':[{'tile_id':tile['tile_id'],'parent_photo':tile['parent_photo'],
          'candidate_a':'0','candidate_b':'1','source_endpoints':[1,0],'score_unvalidated':.3}],
          'candidate_questions':[], 'expert_approved_connections':0,'expert_approved_lines':0}
    return pack,priority

def test_real_rank_is_only_review_not_labels():
    pack,p=sample();out=rank_unlabeled_real_pairs(pack,p,FakeModel(),1000)
    assert out['s23_protocol']['physical_joins_authorized'] is False
    assert out['s23_protocol']['expert_pair_labels_created']==0
    assert out['s23_protocol']['review_pair_count']==len(p['pair_questions'])
    assert out['pair_questions'][0]['s23_domain_flag']=='FILM_AND_ANNOTATION_DOMAIN_UNVERIFIED'
    assert p['pair_questions'][0].get('s23_noncertified_rank_score') is None


def test_incompatible_review_package_rejected():
    pack,p=sample();pack['package_sha256']='0'*64
    with pytest.raises(ValueError):observed_pair_rows(pack,p)
    pack,p=sample();p['pair_questions'][0]['candidate_b']='missing'
    with pytest.raises(ValueError):observed_pair_rows(pack,p)


def test_source_html_tamper_rejected_and_state_preserved():
    pack,p=sample();out=rank_unlabeled_real_pairs(pack,p,FakeModel(),1000)
    base=json.dumps(p)
    html='<html><script id="s20-priority-payload" type="application/json">'+base+'</script>Подсказки только по геометрии, не готовые ответы.</html>'
    updated=insert_priority_in_workbench(html,p,out)
    assert 'ДРУГОЙ плёнки' in updated
    assert out['source_package_sha256']==p['source_package_sha256']
    with pytest.raises(ValueError):insert_priority_in_workbench(html,dict(p,source_package_sha256='wrong'),out)
