import unittest
import numpy as np
from scripts.cracks_real_neighbor_path_v17 import (
    curves,candidate_scores,choose,gradient,OFFSETS,distance_to_expert,audit)

class TestRealNeighborPathV17(unittest.TestCase):
    def test_identical_visible_endpoints(self):
        cs=curves((32,20),(32,55),(28,36,21,54))
        self.assertEqual(len(cs),len(OFFSETS))
        self.assertEqual(len({p.shape[1] for p in cs}),1)
        self.assertTrue(np.allclose(cs[2][0],32))
    def test_no_hidden_truth_required_by_policy(self):
        paths=curves((30,20),(30,60),(28,34,22,59))
        image=np.zeros((80,90),np.float32)
        image[30,20:61]=4
        scores=candidate_scores(paths,[image,image,image,np.zeros_like(image),np.zeros_like(image)])
        self.assertEqual(choose(scores,'central'),OFFSETS.index(0.))
        self.assertEqual(choose(scores,'central_adjacent'),OFFSETS.index(0.))
        truth=np.zeros_like(image,dtype=bool)
        truth[30,20:61]=True
        self.assertGreaterEqual(distance_to_expert(paths[2],truth)[0],0)
    def test_neighbor_order_permutation(self):
        scores=np.arange(25,dtype=float).reshape(5,5)
        self.assertEqual(choose(scores,'adjacent'),choose(scores[:,[0,2,1,3,4]],'adjacent'))
    def test_gradient_image_feature(self):
        image=np.zeros((40,60),np.float32);image[:,30:]=1
        g=gradient(image)
        self.assertEqual(g.shape,image.shape)
        self.assertGreater(g[:,29].mean(),g[:,5].mean())
    def test_local_zip_reproducibility_and_flagging(self):
        import io,tempfile,zipfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);az=root/'images.zip';lz=root/'labels.zip'
            image=np.ones((128,220),dtype=np.uint8)*128
            for x in range(20,200,11):image[:,x:x+2]=200
            buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG')
            with zipfile.ZipFile(az,'w') as z:
                for i in range(1,241):z.writestr(f'images/section_{i:03}.png',buf.getvalue())
            mask=np.ones((128,220,3),dtype=np.uint8)*255
            mask[60,20:200]=[31,119,180]
            buf=io.BytesIO();Image.fromarray(mask).save(buf,format='PNG')
            with zipfile.ZipFile(lz,'w') as z:z.writestr('expert/section_060.png',buf.getvalue())
            r=audit(az,lz,root/'audit.json',root/'cases.csv',400)
            self.assertEqual(r['N_expert_sections'],1)
            self.assertGreater(r['N_gaps'],0)
            self.assertFalse(r['is_full_ANZA_SBPP_LIRA_E2E'])
            self.assertFalse(r['uses_expert_hidden_gap_for_inference'])
            self.assertFalse(r['false_bridge_negatives_available'])
