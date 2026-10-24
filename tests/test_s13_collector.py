import json
import pytest
from pathlib import Path
from scripts.research.bubble_chamber.s13_collect import collect

def fake(seed,source='same'):
    f=[f'070_{x:03d}' for x in range(8)]
    arms={}
    for arm,tps in [('unet_base',5),('conv_guided',6),('az_guided',7),('az_pure',4)]:
        arms[arm]={'test':{}}
        for cond in ('clean','stress'):
            arms[arm]['test'][cond]={'counts':{'dice':tps/10},'macro_cldice':tps/10,
              'per_frame_counts':{x:[tps,2,10-tps] for x in f},'source_frames':f}
    return {'schema':'S13_ORIGINAL_AZ_GUIDED_UNET_AND_CONTROLS_V1','config':{'seed':seed,'data_seed':10},
      'exact_source_blob_sha':{'az':'same'},'data_fingerprints':{'train':source},
      'source':{'synthetic':True},'results':arms}

def test_identical_frames_and_paired_delta(tmp_path):
    paths=[]
    for k in [1,2,3]:
        p=tmp_path/f'{k}.json';p.write_text(json.dumps(fake(k)));paths.append(p)
    r=collect(paths,tmp_path/'summary.json',draws=200)
    v=r['comparisons']['stress:az_guided-minus-conv_guided']
    assert v['pooled_dice_difference']>0 and v['frame_cluster_CI95'][0]>0
    assert v['test_frames']==8 and v['model_seeds']==3
    assert (tmp_path/'summary.json').exists()
    paths[-1].write_text(json.dumps(fake(3,source='modified')))
    with pytest.raises(ValueError,match='changed'):collect(paths,tmp_path/'summary.json')
