import unittest
import numpy as np
from scripts.cracks_real_oracle_gap_audit import paths_from_mask, split

class TestRealCracksMaskAudit(unittest.TestCase):
    def test_straight_branch_trace(self):
        mask=np.zeros((100,100),dtype=bool)
        mask[10:90,50]=True
        traces=list(paths_from_mask(mask))
        self.assertEqual(len(traces),1)
        self.assertEqual(len(traces[0]),80)

    def test_spatial_block_same_split(self):
        self.assertEqual(split(41),split(42))
        self.assertEqual(split(41)[1],8)
        self.assertEqual(split(59)[1],11)

    def test_no_branch_is_not_evidence_of_full_e2e(self):
        mask=np.zeros((64,64),dtype=bool)
        mask[8:20,25]=True
        self.assertEqual(list(paths_from_mask(mask)),[])

if __name__=="__main__":
    unittest.main()
