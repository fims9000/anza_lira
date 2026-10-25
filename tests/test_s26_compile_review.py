import json,hashlib
from pathlib import Path
import cv2,numpy as np,pytest
from scripts.research.bubble_chamber.s26_compile_human_review import compile_review,stroke_pixels

def toy_package(tmp_path):
    source=tmp_path/'src';source.mkdir()
    records=[]
    for photo,count in [('005',6),('006',3)]:
        for j in range(count):
            id=f'H50Z_030_01_{photo}_start{j:02}'
            path=source/f'{id}.png'
            cv2.imwrite(str(path),np.full((512,512),220,dtype='uint8'))
            records.append({'crop_id':id,'source_crop_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'width':512,'height':512,'strokes':[]})
    document={'schema':'s26_real_track_manual_review_v1','images':records,'generated_model_labels':0,
              'unknown_pixels_not_background':True}
    f=tmp_path/'reviews.json';f.write_text(json.dumps(document))
    return source,f,document

def test_empty_real_review_is_blocked_and_all_pixels_unknown(tmp_path):
    src,p,doc=toy_package(tmp_path)
    result=compile_review(p,src,tmp_path/'out')
    assert not result['trainable_under_original_sparse_pipeline']
    assert result['positive_reviewed_pixel_total']==result['negative_reviewed_pixel_total']==0
    assert len(result['records'])==9 and all(x['unknown_pixels']==512*512 for x in result['records'])
    manifest=json.loads((tmp_path/'out/human_only_manifest.json').read_text())
    assert set(r['split'] for r in manifest['entries'])=={'unlabeled'}

def test_sparse_human_labels_are_not_filled_between_lines(tmp_path):
    src,p,d=toy_package(tmp_path)
    for a in [d['images'][0],d['images'][6]]:
        a['strokes']=[{'label':'track','radius_px':1,'points_xy':[[8,10],[20,20]]},
                      {'label':'artifact','radius_px':1,'points_xy':[[200,200],[211,210]]}]
    p.write_text(json.dumps(d))
    report=compile_review(p,src,tmp_path/'out')
    assert report['trainable_under_original_sparse_pipeline']
    im=cv2.imread(str(tmp_path/'out/masks'/f'{d["images"][0]["crop_id"]}_manual_01255.png'),cv2.IMREAD_UNCHANGED)
    assert set(np.unique(im))=={0,1,255}
    assert np.sum(im==255)>.99*im.size
    assert im[10,8]==1 and im[200,200]==0 and im[300,300]==255

def test_tampered_source_digest_fails_closed(tmp_path):
    src,p,d=toy_package(tmp_path);d['images'][0]['source_crop_sha256']='0'*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='digest'):compile_review(p,src,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_conflicting_manual_pixels_rejected(tmp_path):
    src,p,d=toy_package(tmp_path)
    d['images'][0]['strokes']=[{'label':'track','radius_px':2,'points_xy':[[10,10]]},
                             {'label':'artifact','radius_px':2,'points_xy':[[10,10]]}]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='conflicting'):compile_review(p,src,tmp_path/'out')

def test_injection_without_human_proof_rejected(tmp_path):
    src,p,d=toy_package(tmp_path);d['generated_model_labels']=37
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='do not promote'):compile_review(p,src,tmp_path/'out')

def test_all_invalid_hand_draw_strokes_rejected():
    with pytest.raises(ValueError):stroke_pixels({'label':'track','radius_px':3,'points_xy':[[-1,1]]})
    with pytest.raises(ValueError):stroke_pixels({'label':'fake','radius_px':3,'points_xy':[[1,1]]})
