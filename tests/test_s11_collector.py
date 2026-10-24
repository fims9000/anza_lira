from copy import deepcopy
import numpy as np
import pytest
from scripts.research.bubble_chamber.s11_collect_multiseed import aggregate


def fake(seed,offset=0):
    frames={f'070_{i:03d}':[10+i+offset,1,3] for i in range(5)}
    base={'cldice_per_image':[.4]*5,'per_frame_counts':frames}
    variant={'cldice_per_image':[.7]*5,'per_frame_counts':{f:[v[0]+2,v[1],v[2]] for f,v in frames.items()}}
    return {'config':{'seed':seed,'data_seed':1},
            'source_meta':{'split_fingerprints':{'train':{'sha256':'a'},'test':{'sha256':'b'}}},
            'results':{'az_base':{'evaluation':{'clean':base,'stress':base}},
                       'az_cldice':{'evaluation':{'clean':variant,'stress':variant}}}}


def test_collector_positive_delta_and_seed_frame_guard():
    reports=[fake(20),fake(21)]
    a=aggregate(reports,draws=200)
    k=a['comparisons']['clean_az_cldice_minus_az_base']
    assert k['delta_micro_dice']>0
    assert k['delta_macro_cldice']>.25
    assert k['frames']==5 and k['model_seeds']==2
    assert k['delta_micro_dice_ci95'][0]>=0
    with pytest.raises(ValueError,match='seeds must be different'):
        aggregate([fake(20),fake(20)])
    p=fake(21);p['config']['data_seed']=2
    with pytest.raises(ValueError,match='Data seed changed'):
        aggregate([fake(20),p])
    p=fake(21);p['source_meta']['split_fingerprints']['train']['sha256']='c'
    with pytest.raises(ValueError,match='Renderer/source-frame drift'):
        aggregate([fake(20),p])
