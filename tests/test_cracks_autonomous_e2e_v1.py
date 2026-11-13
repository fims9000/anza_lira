"""Data-isolation and autonomous geometric-route tests; never uses a label in policy."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.cracks_autonomous_e2e_v1 import (
    BLUE,ORANGE,features,endpoints_from_mask,candidates,complete,explicit_mask_metrics
)


class AutonomousCracksTests(unittest.TestCase):
    def test_full_geometry_from_predicted_mask_without_expert_points(self):
        image=np.zeros((85,125),np.uint8)
        image[40,:]=235
        mask=np.zeros(image.shape,bool)
        mask[40,15:45]=True
        mask[40,61:100]=True
        probability=np.full(image.shape,.3,np.float32)
        probability[mask]=.9
        ends=endpoints_from_mask(mask)
        self.assertEqual(len(ends),4)
        options,_=candidates(mask,probability,image)
        self.assertTrue(options)
        linked,links,counts=complete(mask,probability,image,decision_threshold=.4,max_links=1)
        self.assertTrue(links)
        self.assertTrue(linked[40,53])
        self.assertFalse(counts["no_link"])

    def test_no_link_high_threshold_is_safe(self):
        mask=np.zeros((85,125),bool)
        mask[40,15:45]=True
        mask[40,61:100]=True
        linked,links,info=complete(mask,np.zeros(mask.shape,np.float32),
                                   np.zeros(mask.shape,np.uint8),
                                   decision_threshold=2.)
        self.assertFalse(links)
        self.assertTrue(info["no_link"])
        np.testing.assert_array_equal(linked,mask)

    def test_white_and_green_never_count_as_explicit_negative(self):
        labels=np.zeros((5,8,3),np.uint8)+255
        labels[0,0]=BLUE
        labels[0,1]=ORANGE
        labels[0,2]=[44,160,44]
        pred=np.zeros((5,8),bool)
        pred[0,:3]=True
        metrics=explicit_mask_metrics(pred,labels)
        self.assertEqual(metrics["tp"],1)
        self.assertEqual(metrics["fp"],1)
        self.assertEqual(metrics["fn"],0)
        self.assertEqual(metrics["negative_pixels"],1)

    def test_features_have_14_finite_channels(self):
        x=features(np.ones((64,100),np.uint8)*90)
        self.assertEqual(x.shape,(64,100,14))
        self.assertTrue(np.isfinite(x).all())

if __name__=="__main__":unittest.main()
