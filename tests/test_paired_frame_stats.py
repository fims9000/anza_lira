from pathlib import Path
import csv
import pytest
from bubble_chamber.paired_frame_stats import paired_frame_delta


def _csv(path,paired=True):
    fieldnames=['frame_id','view','scene_id','left_id','right_id','candidate_rank','model','truth_same_track','accepted_offline']
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fieldnames);w.writeheader()
        for k in range(24):
            for model in ('A','B'):
                if model=='B' and not paired and k==0:continue
                w.writerow({'frame_id':f'070_{k:03d}', 'view':1,'scene_id':k,'left_id':'e1t01','right_id':'e1t02','candidate_rank':0,
                            'model':model,'truth_same_track':k%3==0,'accepted_offline':model=='B' and k%3==0})


def test_bootstrap_determinism_and_direction(tmp_path):
    p=tmp_path/'pred.csv';_csv(p)
    r=paired_frame_delta(p,'A','B',repetitions=50)
    assert r==paired_frame_delta(p,'A','B',repetitions=50)
    assert r['test_frames']==24
    assert r['B_minus_A_recall']==1.0
    assert r['test_tp_A']==0 and r['test_tp_B']==8


def test_unpaired_candidates_rejected(tmp_path):
    p=tmp_path/'pred.csv';_csv(p,False)
    with pytest.raises(ValueError,match='not paired'):
        paired_frame_delta(p,'A','B')
