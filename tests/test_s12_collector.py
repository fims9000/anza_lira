import json
import pytest
import numpy as np
from scripts.research.bubble_chamber.s12_collect_multiseed import pooled,collect,dice


def _source(seed=1,ids=None):
    ids=ids or [f'070_{i:03d}' for i in range(12)]
    base={f:[5,2,3] for f in ids}
    better={f:[7,2,1] for f in ids}
    return {
        'baseline': {'test': {
            'stress': {'per_frame_counts':base},
            'clean': {'per_frame_counts':base},
        }},
        'better': {'test': {
            'stress': {'per_frame_counts':better},
            'clean': {'per_frame_counts':better},
        }},
    }


def test_pooled_frame_dice_deterministic_and_truth_independent():
    runs={1:_source(),2:_source(),3:_source()}
    a=pooled(runs,'baseline','better',condition='stress',draws=250)
    assert a==pooled(runs,'baseline','better',condition='stress',draws=250)
    assert a['pooled_dice_difference']>0
    assert a['frame_cluster_CI95'][0]>0
    assert a['test_frames']==12 and a['model_seeds']==3


def test_mismatched_source_frames_rejected():
    runs={1:_source(),2:_source(ids=['different']*12)}
    with pytest.raises(ValueError):pooled(runs,'baseline','better',condition='stress')


def test_aggregate_validates_fixed_data(tmp_path):
    def result(seed,tag):
        report={'schema':'S12_DIRECT_GT_TANGENT_TO_AUTHENTIC_AZCONV_V1',
                'config':{'seed':seed},'data_fingerprints':{'train':tag},
                'source_git_blobs':{'azconv':'same'},'results':{}}
        for arm in ('az_base','az_direct05','az_direct_cldice05','unet_base','unet_cldice05'):
            report['results'][arm]={'test':{}}
            for cond in ('clean','stress'):
                report['results'][arm]['test'][cond]={
                 'source_frames':[f'070_{i:03d}' for i in range(12)],
                 'per_frame_counts':{f'070_{i:03d}':[5,2,3] for i in range(12)},
                 'counts':{'dice':.6},'macro_cldice':.5}
        return report
    p=[]
    for k in range(3):
        x=tmp_path/f'{k}.json';x.write_text(json.dumps(result(k,'same')));p.append(x)
    out=tmp_path/'metrics.json'
    x=collect(p,out)
    assert x['condition_means']['clean']['az_base']['mean_dice_descriptive']==.6
    assert out.exists()
    p[2].write_text(json.dumps(result(2,'changed')))
    with pytest.raises(ValueError,match='Data or train/test'):
        collect(p,out)
