import numpy as np
import pytest
from bubble_chamber.large_relation_study import frame_partition, _bootstrap_frame_ci, LargeRelationProtocol


def test_frame_split_never_leaks_views_or_gaps():
    ids=[f'070_{i:03d}' for i in range(80)]
    a=frame_partition(ids,20261024)
    b=frame_partition(ids[::-1],20261024)
    assert a==b
    assert set(a['train']).isdisjoint(a['test'])
    assert set(a['train']).isdisjoint(a['val'])
    assert set(a['val']).isdisjoint(a['test'])
    assert sum(map(len,a.values()))==80


def test_frame_partition_rejects_tiny_dataset():
    with pytest.raises(ValueError,match='>=20'):
        frame_partition(['070_008','070_009'],1)


def test_cluster_bootstrap_over_frames_deterministic():
    rows=[{'frame_id':f'070_{i:03d}','truth_same_track':bool(i%2)} for i in range(30)]
    accepted=np.asarray([i%3==0 for i in range(30)],dtype=bool)
    a=_bootstrap_frame_ci(rows,accepted,repeats=40,seed=13)
    assert a==_bootstrap_frame_ci(rows,accepted,repeats=40,seed=13)
    assert 0<=a['fpr_ci95'][0]<=a['fpr_ci95'][1]<=1


def test_protocol_is_immutable():
    p=LargeRelationProtocol()
    with pytest.raises((AttributeError,TypeError)):
        p.seed=9
