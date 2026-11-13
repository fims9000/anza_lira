import unittest
import numpy as np
from scripts.cracks_real_risk_calibrated_v27 import (
    subset_mask,calibrate_region_link_threshold,THRESHOLDS
)


class RiskCalibratedV27Tests(unittest.TestCase):
    def test_subset_never_removes_existing_pixels(self):
        raw=np.zeros((24,30),bool)
        raw[8,1:9]=True
        selected=({"score":.50,"path":[(8,9),(8,10),(8,11)]},
                  {"score":.05,"path":[(12,10),(13,10)]})
        prediction={"raw_mask":raw,"selected":selected}
        linked,count=subset_mask(prediction,.30)
        self.assertEqual(count,1)
        self.assertTrue(linked[8,10])
        self.assertFalse(linked[12,10])
        self.assertTrue(np.all(~raw|linked))

    def test_calibration_only_consensus_explicit_labels(self):
        raw=np.zeros((25,40),bool)
        pos=np.zeros_like(raw)
        neg=np.zeros_like(raw)
        pos[10,10:15]=True
        neg[15,10:15]=True
        pred={"raw_mask":raw,"selected":[
            {"score":.50,"path":[(10,10),(10,11),(10,12)]},
            {"score":.10,"path":[(15,10),(15,11),(15,12)]}
        ]}
        result,allrows=calibrate_region_link_threshold([(pred,pos,neg)]*4)
        self.assertIsNotNone(result)
        self.assertEqual(result["links"],4)
        self.assertEqual(result["new_orange"],0)
        self.assertGreaterEqual(result["threshold"],.30)
        self.assertEqual(len(allrows),len(THRESHOLDS))

    def test_no_labels_yields_no_link_proposed_but_no_invalid_success(self):
        raw=np.zeros((20,20),bool)
        prediction={"raw_mask":raw,"selected":[]}
        zeros=np.zeros_like(raw)
        r,curves=calibrate_region_link_threshold([(prediction,zeros,zeros)]*4)
        self.assertIsNotNone(r)
        self.assertEqual(r["links"],0)
        self.assertEqual(r["new_blue"],0)
if __name__=="__main__":unittest.main()
