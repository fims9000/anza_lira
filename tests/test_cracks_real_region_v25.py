import unittest
import numpy as np
from scipy import ndimage as ndi
from scripts.cracks_real_region_v25 import (
  component_extremity_ports,filter_distinct_components,run_regions,region_candidates)

class RegionV25Tests(unittest.TestCase):
    def test_region_endpoints_derived_from_dense_mask(self):
        p=np.zeros((100,180),np.float32)
        p[45,10:50]=.9;p[45,64:125]=.95
        mask=p>=.6
        lbl,ports=component_extremity_ports(p,mask)
        self.assertEqual(len(ports),4)
        self.assertEqual(len(set(x.trace_id for x in ports)),2)
        for x in ports:self.assertTrue(lbl[x.point]>0)
        r=run_regions(p,.6,mode='region_only')
        self.assertGreater(len(r['candidates']),0)
        self.assertTrue(np.all(~r['raw_mask']|r['completed_mask']))
    def test_no_false_link_with_one_connected_component(self):
        p=np.zeros((100,150),np.float32)
        p[37,15:120]=.95
        for mode in ('region_only','region_hybrid'):
            r=run_regions(p,.6,mode=mode)
            self.assertEqual(len(r['selected']),0)
            self.assertTrue(np.array_equal(r['raw_mask'],r['completed_mask']))
    def test_no_connected_component_self_edge(self):
        p=np.zeros((100,160),np.float32)
        p[41,10:45]=.95;p[41,61:90]=.95
        m,t,tr,lbl,ports,candidates=region_candidates(p,.7,mode='region_hybrid')
        self.assertTrue(all(lbl[ports[i].point]!=lbl[ports[j].point]
                            for i,j,_,_ in candidates))
    def test_ambiguous_competitor_is_allowed_to_reject(self):
        p=np.zeros((110,180),np.float32)
        p[25,15:45]=.9;p[80,120:150]=.92
        for mode in ('region_only','region_hybrid'):
            r=run_regions(p,.7,mode=mode)
            self.assertEqual(len(r['selected']),0)

if __name__=='__main__':unittest.main()
