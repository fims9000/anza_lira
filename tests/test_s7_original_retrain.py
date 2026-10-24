"""S7 unit-level contracts; exact source models are exercised in full repo/CI."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/research/bubble_chamber/s7_original_retrain.py'
spec=importlib.util.spec_from_file_location('s7_original_retrain', SCRIPT)
s7=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=s7
spec.loader.exec_module(s7)


def test_git_blob_integrity_gate(tmp_path, monkeypatch):
    expected={}
    for name in ('models/azconv.py','models/segmentation.py'):
        path=tmp_path/name;path.parent.mkdir(exist_ok=True)
        path.write_text('original content',encoding='utf-8')
        expected[name]=s7.blob_sha(path.read_bytes())
    monkeypatch.setattr(s7,'SOURCE_BLOBS',expected)
    assert s7.verify_sources(tmp_path)==expected
    (tmp_path/'models/azconv.py').write_text('changed content',encoding='utf-8')
    with pytest.raises(ValueError,match='source mismatch'):
        s7.verify_sources(tmp_path)
    (tmp_path/'models/azconv.py').unlink()
    with pytest.raises(FileNotFoundError):
        s7.verify_sources(tmp_path)


def test_procedural_deterministic_and_independent():
    a=s7.procedural_examples(3,32,seed=40,split='train')
    b=s7.procedural_examples(3,32,seed=40,split='train')
    c=s7.procedural_examples(3,32,seed=40,split='test')
    for (x,y,k),(p,q,l) in zip(a,b):
        assert k==l and np.array_equal(x,p) and np.array_equal(y,q)
        assert x.shape==(32,32) and y.shape==(32,32)
        assert set(np.unique(y)) <= {0,1}
    assert any(not np.array_equal(a[i][0],c[i][0]) for i in range(3))
    assert set(x[2] for x in a).isdisjoint(x[2] for x in c)


def test_metric_and_fixed_stress_mask():
    y=np.asarray([[1,0],[0,1]],np.uint8)
    p=np.asarray([[1,1],[0,0]],np.uint8)
    m=s7.metric_from_counts(s7.metrics_counts(p,y))
    assert m['tp']==1 and m['fp']==1 and m['fn']==1
    assert m['dice']==.5
    cfg=s7.RunConfig(size=32)
    x=torch.ones(2,1,32,32)*.3
    ids=torch.tensor([3,8])
    out=s7.apply_stress(x,ids,cfg)
    assert out.shape==x.shape
    assert torch.allclose(out,s7.apply_stress(x,ids,cfg))
    assert torch.all(out>=0) and torch.all(out<=1)
    assert not torch.allclose(out,x)


def test_frame_group_bootstrap_no_pixel_independence():
    y=np.zeros((4,2,2),np.uint8);y[:,0,0]=1
    good=y.copy();bad=np.zeros_like(y)
    a=s7.paired_bootstrap((good,y),(bad,y),['frame1','frame1','frame2','frame2'],seed=10,repeats=40)
    assert a['bootstrap_unit']=='source_frame'
    assert a['unique_groups']==2
    assert a['mean_delta_dice']>.9
    with pytest.raises(ValueError):
        s7.paired_bootstrap((good,y),(bad,y[:3]),['frame1']*4,seed=10)


def test_training_creates_restartable_checkpoints(tmp_path):
    cfg=s7.RunConfig(size=32,epochs=1,batch_size=2,train_limit=4,val_limit=2,test_limit=3,cpu_threads=2)
    sets={name:s7.procedural_examples(getattr(cfg,f'{name}_limit'),32,seed=cfg.seed,split=name)
          for name in ('train','val','test')}
    m=torch.nn.Conv2d(1,1,3,padding=1)
    result,clean,stress=s7.train_one('toy_smoke',m,sets,cfg,tmp_path)
    assert (tmp_path/'toy_smoke/last.pt').exists()
    assert (tmp_path/'toy_smoke/best.pt').exists()
    assert (tmp_path/'toy_smoke/best_final.pt').exists()
    obj=torch.load(tmp_path/'toy_smoke/best_final.pt',weights_only=False)
    assert obj['synthetic_labels_only'] is True
    assert result['trained_epochs']==1
    assert 'validation_selected_threshold' in obj
    assert clean[0].shape==clean[1].shape==(3,32,32)
    assert stress[0].shape==stress[1].shape==(3,32,32)


def test_no_original_source_refuses_training(tmp_path, monkeypatch):
    monkeypatch.setattr(s7,'REPO_ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):
        s7.run(s7.RunConfig(size=32,epochs=1),measurements=None,out_dir=tmp_path/'out')
