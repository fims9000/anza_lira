import pytest
import numpy as np
from scripts.research.bubble_chamber.s10_collect_fixed import paired_frame_bootstrap,dice_from_counts


def _rows():
    return [{'model_seed':s,'data_seed':2,'test_groups':['f1','f2','f3'],
             'conditions':{'stress':{'original_azunet':[[8,1,2],[9,0,1],[7,2,3]],
                                     'plain_unet':[[7,2,3],[9,0,1],[6,3,4]]}}}
            for s in [1,2,3]]


def test_paired_fixed_film_and_seed_bootstrap():
    x=paired_frame_bootstrap(_rows(),'stress',seed=4,repetitions=50)
    assert x['n_unique_test_frames']==3
    assert x['delta_pooled_micro_dice']>0
    assert x['ci95_by_test_source_frame'][0]>0
    assert x['conditional_on_these_model_seeds'] is True


def test_refuse_different_test_groups_or_masks():
    rows=_rows(); rows[1]['test_groups'][0]='leaked'
    with pytest.raises(ValueError,match='IDs differ'):
        paired_frame_bootstrap(rows,'stress')
    rows=_rows();rows[2]['conditions']['stress']['original_azunet'][0][2]+=1
    rows[2]['conditions']['stress']['plain_unet'][0][2]+=1
    with pytest.raises(ValueError,match='Test labels'):
        paired_frame_bootstrap(rows,'stress')


def test_counts_saturate_empty_truth_without_nan():
    assert dice_from_counts(np.zeros((2,3),dtype=np.int64))==0
