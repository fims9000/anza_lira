import unittest
import numpy as np
import torch

from scripts.cracks_real_neural_v26 import (
    CompactDetector, normalize_image, sample_batch, masked_binary_loss,
    dense_probs, TRAIN_IDS, CAL_IDS, PATCH, BATCH,
)
from models.azconv import AZConv2d

class V26ActualAZConvTests(unittest.TestCase):
    def test_original_az_operator_is_really_used(self):
        az=CompactDetector("az")
        conv=CompactDetector("conv")
        self.assertIsInstance(az.stem,AZConv2d)
        self.assertIsInstance(conv.stem,torch.nn.Conv2d)
        for m in (az,conv):
            x=torch.randn(2,1,16,18)
            out=m(x)
            self.assertEqual(tuple(out.shape),(2,1,16,18))
            self.assertTrue(bool(torch.isfinite(out).all()))
            target=torch.zeros_like(out)
            known=torch.ones_like(out)
            target[:, :, 7:9, 9:11]=1.
            loss=masked_binary_loss(out,target,known)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertGreater(float(sum(p.grad.abs().sum() for p in m.parameters()
                if p.grad is not None)),0.)
    def test_identical_patch_sampling_seed_across_model_variants(self):
        h,w=100,200
        image=np.random.default_rng(7).random((h,w)).astype(np.float32)
        pos=np.zeros((h,w),bool);pos[30:35,40:50]=True
        neg=np.zeros((h,w),bool);neg[65:78,120:135]=True
        rec={'image':normalize_image(image),'positive':pos,'negative':neg,
             'pos_idx':np.argwhere(pos),'neg_idx':np.argwhere(neg)}
        a=sample_batch([rec],np.random.default_rng(991))
        b=sample_batch([rec],np.random.default_rng(991))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(a[0].shape,(BATCH,1,PATCH,PATCH))
        self.assertGreater(int((a[2]*(a[1]>.5)).sum()),0)
        self.assertGreater(int((a[2]*(a[1]<=.5)).sum()),0)
    def test_cal_and_train_ids_disjoint(self):
        self.assertFalse(set(TRAIN_IDS)&set(CAL_IDS))
        self.assertGreater(min(TRAIN_IDS),max(CAL_IDS))
    def test_image_dense_output_shape(self):
        m=CompactDetector("conv")
        p=dense_probs(m,np.random.default_rng(10).random((32,38)))
        self.assertEqual(p.shape,(32,38))
        self.assertGreaterEqual(p.min(),0.)
        self.assertLessEqual(p.max(),1.)
    def test_reject_single_class_loss(self):
        z=torch.zeros((1,1,8,9))
        with self.assertRaisesRegex(ValueError,"Batch lacks explicit class"):
            masked_binary_loss(z,z,torch.ones_like(z))

if __name__=="__main__":
    unittest.main()
