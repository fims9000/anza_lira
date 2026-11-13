import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.cracks_real_auto_run_v23 import zip_index,_train,BLUE,ORANGE

class RealAutoTrainingTests(unittest.TestCase):
    def test_train_crowd_calibration_without_expert(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            images=root/'images.zip'; masks=root/'annotations.zip'
            ids=[340,342,344,346,348,350,352,311,313,315,317]
            with zipfile.ZipFile(images,'w') as zi,zipfile.ZipFile(masks,'w') as za:
                for sid in ids:
                    rng=np.random.default_rng(sid)
                    im=np.uint8(np.clip(rng.normal(100,8,(90,200)),0,255))
                    im[:,98:100]=240
                    buffer=io.BytesIO();Image.fromarray(im).save(buffer,format='PNG')
                    zi.writestr('images/section_%03d.png'%sid,buffer.getvalue())
                    lab=np.zeros((90,200,3),np.uint8);lab[:]=ORANGE
                    lab[:,97:102]=BLUE
                    buffer=io.BytesIO();Image.fromarray(lab).save(buffer,format='PNG')
                    for author in ('novice01','novice02','practitioner1','practitioner2'):
                        za.writestr(f'Fault segmentations/{author}/section_{sid:03d}.png',buffer.getvalue())
            with zipfile.ZipFile(images) as zi,zipfile.ZipFile(masks) as za:
                idx,ann=zip_index(zi,za)
                model,tau,summary=_train(zi,za,idx,ann,out=root)
            self.assertTrue((root/'crowd_only_anisotropic_hgb.joblib').is_file())
            self.assertGreaterEqual(len(summary['train_sections']),5)
            self.assertGreaterEqual(len(summary['calibration_sections']),3)
            self.assertIn(tau,summary['calibration_curves'][0].values() if False else
                         [r['threshold'] for r in summary['calibration_curves']])
            self.assertGreater(summary['N_pos'],0)

if __name__=='__main__':unittest.main()
