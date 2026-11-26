from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/research/ccta_graph_lira_safe_repair'))
from ct28_endpoint_jitter import jitter_batch
from ct28_runtime_gap_supervision import gap_pair_labels, window_labels


def fixture(ambiguous=False):
    reference=np.zeros((30,16,16),dtype=bool)
    reference[2:24,5,5]=True
    reference[3:11,10,5]=True
    broken=reference.copy()
    broken[11:14,5,5]=False
    gaps={'gap_radius_mm':1.5,'centers':[{'center_voxel':[12,5,5],'removed_voxels':3}]}
    endpoints=[NS(endpoint_id=k,voxel=v) for k,v in [('a',(10,5,5)),('b',(14,5,5)),('c',(10,10,5)),('d',(9,5,5))]]
    cand=lambda *ids:NS(kind='PAIR',endpoint_ids=ids,candidate_id='+'.join(ids))
    scenes=[NS(scene_id='positive',candidates=[cand('a','b'),cand('a','c')]),
            NS(scene_id='negative',candidates=[cand('a','c')])]
    if ambiguous:
        scenes[0].candidates.append(cand('b','d'))
    return reference,broken,np.eye(4),gaps,endpoints,scenes


def test_unique_gap_labels_and_negative_window():
    *args,scenes=fixture()
    gaps=gap_pair_labels(*args,scenes)
    assert gaps[0]['status']=='unique' and gaps[0]['candidate_pairs']==[['a','b']]
    assert window_labels(scenes,gaps)==({'positive':'PAIR','negative':'NONE'}, {})


def test_ambiguous_gap_not_used_as_positive_or_negative():
    *args,scenes=fixture(True)
    gaps=gap_pair_labels(*args,scenes)
    assert gaps[0]['status']=='ambiguous_pair'
    labels,skipped=window_labels(scenes,gaps)
    assert labels=={'negative':'NONE'} and 'positive' in skipped


def test_reject_changed_damage_receipt():
    r,b,a,g,e,s=fixture()
    g['centers'][0]['removed_voxels']=2
    with pytest.raises(ValueError,match='voxel count'):
        gap_pair_labels(r,b,a,g,e,s)


def test_jitter_preserves_alignment_and_inputs():
    class Fixed:
        def random(self): return 0.
        def normal(self,loc,scale,size): return np.array([[2.,0.,0.],[0.,0.,0.]])
    g=np.indices((12,12,12),dtype=np.float32)
    x=np.zeros((1,4,12,12,12),dtype=np.float32)
    x[0,0]=g[0]/12
    y=np.zeros((1,1,12,12,12),dtype=np.float32);y[0,0,5,5,5]=1
    for i,p in enumerate(([4.,5.,5.],[7.,5.,5.]),1):
        x[0,i]=np.exp(-np.sum((g-np.array(p)[:,None,None,None])**2,axis=0)/4.5)
    before=x.copy(); target=y.copy()
    a,b,n=jitter_batch(x,y,degree=2,voxel_mm=1.,endpoint_sigma_mm=1.5,noise_mm=.8,rng=Fixed())
    assert n==1 and np.unravel_index(a[0,1].argmax(),a.shape[2:])==(5,5,5)
    assert np.unravel_index(a[0,2].argmax(),a.shape[2:])==(6,5,5)
    assert b[0,0,4,5,5]==1 and a[0,0,4,5,5]==pytest.approx(x[0,0,5,5,5])
    assert not a[0,3].any() and np.array_equal(before,x) and np.array_equal(target,y)


def test_junction_jitter_seed_and_binary_target():
    grid=np.indices((12,12,12),dtype=np.float32)
    x=np.zeros((8,4,12,12,12),dtype=np.float32)
    y=np.zeros((8,1,12,12,12),dtype=np.float32);y[:,:,4:8,5,5]=1
    for i,p in enumerate(([4.,5.,5.],[7.,5.,5.],[5.,7.,5.]),1):
        x[:,i]=np.exp(-np.sum((grid-np.array(p)[:,None,None,None])**2,axis=0)/4.5)
    kw=dict(degree=3,voxel_mm=1.,endpoint_sigma_mm=1.5,noise_mm=.8)
    a,b,n=jitter_batch(x,y,rng=np.random.default_rng(2),**kw)
    c,d,m=jitter_batch(x,y,rng=np.random.default_rng(2),**kw)
    assert 0<n<=8 and n==m and np.array_equal(a,c) and np.array_equal(b,d)
    assert set(np.unique(b))<={0.,1.} and np.all(a[:,3].max(axis=(1,2,3))>0)
