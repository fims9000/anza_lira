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


def test_split_fingerprints_detect_label_changes():
    cfg = s7.RunConfig(size=32, train_limit=4, val_limit=3, test_limit=3)
    sets = {split:s7.procedural_examples(getattr(cfg,f'{split}_limit'),32,seed=cfg.seed,split=split)
            for split in ('train','val','test')}
    d1=s7.split_fingerprints(sets)
    d2=s7.split_fingerprints(sets)
    assert d1==d2
    changed={k:list(v) for k,v in sets.items()}
    x,y,group=changed['train'][0]
    y2=y.copy(); y2[0,0]=1-y2[0,0]
    changed['train'][0]=(x,y2,group)
    assert s7.split_fingerprints(changed)['train']['sha256']!=d1['train']['sha256']
    assert s7.split_fingerprints(changed)['val']['sha256']==d1['val']['sha256']


def test_resume_bitwise_matches_fresh_two_epoch_run(tmp_path):
    import dataclasses
    cfg=s7.RunConfig(size=32,epochs=2,batch_size=2,train_limit=8,val_limit=3,
                     test_limit=4,steps_per_epoch=3,cpu_threads=2)
    sets={k:s7.procedural_examples(getattr(cfg,f'{k}_limit'),32,seed=cfg.seed,split=k)
          for k in ('train','val','test')}
    def model():
        torch.manual_seed(11)
        return torch.nn.Conv2d(1,1,3,padding=1)
    fresh,_,_=s7.train_one('toy',model(),sets,cfg,tmp_path/'fresh')
    one=dataclasses.replace(cfg,epochs=1)
    s7.train_one('toy',model(),sets,one,tmp_path/'resumed')
    resumed,_,_=s7.train_one('toy',model(),sets,cfg,tmp_path/'resumed',resume=True)
    assert resumed['resumed_from_epoch']==1
    assert fresh['history']==resumed['history']
    for f in ('last.pt','best.pt'):
        a=torch.load(tmp_path/'fresh'/'toy'/f,weights_only=False)
        b=torch.load(tmp_path/'resumed'/'toy'/f,weights_only=False)
        assert a['epoch']==b['epoch']
        assert a['data_fingerprints']==b['data_fingerprints']
        for k in a['state_dict']:
            assert torch.equal(a['state_dict'][k],b['state_dict'][k])


def test_resume_refuses_changed_dataset_and_learning_rate(tmp_path):
    import dataclasses
    cfg=s7.RunConfig(size=32,epochs=1,batch_size=2,train_limit=4,val_limit=2,test_limit=3)
    sets={k:s7.procedural_examples(getattr(cfg,f'{k}_limit'),32,seed=cfg.seed,split=k)
          for k in ('train','val','test')}
    model=lambda:torch.nn.Conv2d(1,1,3,padding=1)
    s7.train_one('toy',model(),sets,cfg,tmp_path)
    cfg_extended=dataclasses.replace(cfg,epochs=2)
    corrupted={k:list(v) for k,v in sets.items()}
    x,y,g=corrupted['test'][0];corrupted['test'][0]=(x,y,g+'_unexpected_source')
    with pytest.raises(ValueError,match='data changed'):
        s7.train_one('toy',model(),corrupted,cfg_extended,tmp_path,resume=True)
    with pytest.raises(ValueError,match='configuration changed'):
        s7.train_one('toy',model(),sets,dataclasses.replace(cfg_extended,learning_rate=.002),tmp_path,resume=True)
    with pytest.raises(FileNotFoundError,match='cannot resume'):
        s7.train_one('not_created',model(),sets,cfg_extended,tmp_path,resume=True)


def test_full_runner_with_injected_toy_exercises_all_outputs(tmp_path,monkeypatch):
    """Execution-contract smoke only, NOT a test of original AZConv network."""
    def toy_models(cfg):
        torch.manual_seed(cfg.seed)
        a=torch.nn.Conv2d(1,1,3,padding=1)
        torch.manual_seed(cfg.seed)
        b=torch.nn.Conv2d(1,1,3,padding=1)
        return {'plain_unet':a,'original_azunet':b}
    monkeypatch.setattr(s7,'verify_sources',lambda *args: {})
    monkeypatch.setattr(s7,'build_models',toy_models)
    cfg=s7.RunConfig(size=32,epochs=1,train_limit=4,val_limit=3,test_limit=4,batch_size=2)
    report=s7.run(cfg,measurements=None,out_dir=tmp_path/'smoke')
    assert report['frame_holdout'] is True
    assert report['source_image_pixels_are_real'] is False
    assert report['film_holdout'] is None
    assert report['resume_requested'] is False
    assert (tmp_path/'smoke'/'original_azunet'/'best_final.pt').is_file()
    assert (tmp_path/'smoke'/'plain_unet'/'best_final.pt').is_file()
    assert report['paired']['stress']['unique_groups']==4
    assert report['results']['plain_unet']['data_fingerprints']==report['results']['original_azunet']['data_fingerprints']
