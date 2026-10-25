import numpy as np, pytest
from scripts.research.bubble_chamber.s27_overlap_photo_audit import validate_photo_rows,model_thresholds,summary

def test_locked_photo_validation_and_strata(tmp_path):
    import cv2,hashlib
    a=np.full((512,512),220,'uint8')
    for k in ('005_start00','006_mined00'):
        path=tmp_path/f'H50Z_030_01_{k}.png';assert cv2.imwrite(str(path),a)
    rows=[{'crop_id':'H50Z_030_01_'+k,'source_photo':k[:3], 'stratum':'start',
           'source_crop_sha256':hashlib.sha256((tmp_path/f'H50Z_030_01_{k}.png').read_bytes()).hexdigest()} for k in ('005_start00','006_mined00')]
    assert len(validate_photo_rows(rows,tmp_path))==2
    rows[0]['source_crop_sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):validate_photo_rows(rows,tmp_path)


def test_val_source_is_only_photo005():
    r={'input':{'train_and_validation_photo':'005','heldout_test_photo':'006'},'methods':{k:{'validation':{'threshold':x}} for k,x in [('unet_local',.6),('azunet_local',.7),('sato',.3)]}}
    g={'val_005_calibration':{'threshold':.65}}
    assert model_thresholds(r,g)=={'unet':.6,'azunet':.7,'guided_az':.65,'sato':.3}
    r['input']['train_and_validation_photo']='006'
    with pytest.raises(ValueError,match='photo005'):model_thresholds(r,g)
