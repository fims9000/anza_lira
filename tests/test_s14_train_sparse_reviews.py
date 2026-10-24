from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import cv2
import pytest
import torch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s14_train_sparse_anza.py'
spec=importlib.util.spec_from_file_location('s14train',SCRIPT)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def mini_manifest(tmp_path):
    rows=[]
    for split,count in [('train',2),('val',2),('unlabeled',2)]:
        for i in range(count):
            image=np.full((32,32),200,dtype=np.uint8)
            image[8:24,9+i]=35
            image[5:25,22]=130
            name=f'{split}_{i}.png'
            cv2.imwrite(str(tmp_path/name),image)
            item={'image':name,'image_sha256':hashlib.sha256((tmp_path/name).read_bytes()).hexdigest(),
                  'film_id':f'f_{split}','run_id':f'r_{split}','frame_id':f'{split}_{i}','split':split}
            if split!='unlabeled':
                labels=np.full(image.shape,255,dtype=np.uint8)
                labels[8:24,9+i]=1;labels[5:25,22]=0
                mask=f'{split}_{i}_mask.png';cv2.imwrite(str(tmp_path/mask),labels)
                item.update(mask=mask,mask_sha256=hashlib.sha256((tmp_path/mask).read_bytes()).hexdigest())
            rows.append(item)
    f=tmp_path/'manifest.json'
    f.write_text(json.dumps({'schema':'s14_sparse_image_reviews_v1','entries':rows}))
    return f


def test_manifest_validates_sha_and_keeps_unknown(tmp_path):
    f=mini_manifest(tmp_path)
    groups,stats=m.read_reviewed_manifest(f)
    assert stats['split_examples']=={'train':2,'val':2,'unlabeled':2}
    assert (groups['train'][0][1]==255).sum()>0
    assert np.all(groups['unlabeled'][0][1]==255)
    manifest=json.loads(f.read_text());manifest['entries'][0]['image_sha256']='invalid'
    f.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='SHA mismatch'):m.read_reviewed_manifest(f)


def test_leakage_rejected(tmp_path):
    f=mini_manifest(tmp_path)
    manifest=json.loads(f.read_text())
    manifest['entries'][2]['frame_id']=manifest['entries'][0]['frame_id']
    manifest['entries'][2]['run_id']=manifest['entries'][0]['run_id']
    f.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='frame leakage'):m.read_reviewed_manifest(f)


def test_az_training_on_few_synthetic_scribbles(tmp_path):
    f=mini_manifest(tmp_path)
    groups,stats=m.read_reviewed_manifest(f)
    out=tmp_path/'az_out'
    result=m.fit_sparse_reviews(groups,stats,out_dir=out,epochs=1,widths=(4,4,4,4),rules=2)
    assert result['architecture']=='azunet'
    assert not result['real_unattended_accuracy_established']
    assert np.isfinite(result['history'][0]['known_validation_bce'])
    assert (out/'best.pt').is_file() and (out/'last.pt').is_file()
    payload=torch.load(out/'best.pt',weights_only=False)
    assert payload['source_git_blobs']==m.MODEL_SOURCE_BLOBS
    assert payload['dataset_fingerprint']==stats['source_manifest_sha256']
