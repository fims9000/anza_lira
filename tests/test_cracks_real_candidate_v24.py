import unittest
import numpy as np
from scripts.cracks_real_auto_core_v23 import extract_traces,run_pipeline
from scripts.cracks_real_candidate_v24 import (
  extended_terminal_candidates,virtual_landing_candidates,run_expanded)

class V24CandidateTests(unittest.TestCase):
    def test_relaxed_superset_and_no_ground_truth(self):
        p=np.zeros((100,180),np.float32)
        p[42,10:44]=.91;p[42,90:140]=.92
        _,traces,ports=extract_traces(p,threshold=.7)
        self.assertEqual(len(ports),4)
        self.assertEqual(len(run_pipeline(p,threshold=.7,strategy='context')['candidates']),0)
        ext=extended_terminal_candidates(ports)
        self.assertTrue(any(e[2]>36 for e in ext))
        r=run_expanded(p,.7,mode='terminal_extended')
        self.assertEqual(r['terminal_port_count'],4)
        self.assertGreater(len(r['candidates']),0)
    def test_virtual_candidate_reaches_interior(self):
        p=np.zeros((110,180),np.float32)
        p[30,15:49]=.9
        p[45,10:165]=.92
        _,traces,term=extract_traces(p,threshold=.7)
        self.assertEqual(len(term),4)
        ports,edges=virtual_landing_candidates(traces,term,p)
        self.assertGreater(len(ports),len(term))
        self.assertTrue(any(i<len(term)<=j for i,j,_,_ in edges))
    def test_must_reject_far_and_no_link(self):
        p=np.zeros((100,160),np.float32)
        p[30,15:48]=.90
        r=run_expanded(p,.75,mode='virtual_landing')
        self.assertEqual(r['none_count'],r['terminal_port_count'])
        self.assertEqual(len(r['selected']),0)
    def test_addition_only_masks(self):
        p=np.zeros((100,160),np.float32)
        p[45,12:55]=.9;p[45,66:115]=.9
        for mode in ('terminal_extended','virtual_landing'):
            a=run_expanded(p,.7,mode=mode)
            self.assertTrue(np.all(~a['raw_mask']|a['completed_mask']))
            self.assertGreaterEqual(a['none_count'],0)

if __name__=='__main__':unittest.main()
