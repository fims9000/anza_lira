import unittest
import numpy as np

from scripts.cracks_multiseed_v28 import (
    SEEDS, METHODS, fit_seed, section_bootstrap, summarize
)

class V28ProtocolTests(unittest.TestCase):
    def test_fixed_three_seeds_and_arm_names(self):
        self.assertEqual(SEEDS,(11,23,37))
        self.assertEqual(METHODS,("conv","az"))

    def test_section_cluster_bootstrap_zero_delta(self):
        rows=[]
        for sid in range(40):
            for method in METHODS:
                rows.append({"section":sid, "seed":11, "method":method,
                             "policy":"region_hybrid", "eligible":1, "connected":sid%2,
                             "blue_recalled":5, "n_blue":10,
                             "orange_false":2, "n_orange":100,"links":2})
        result=section_bootstrap(rows,11,repeats=100)
        self.assertEqual(result["difference"],0.)
        self.assertEqual(result["descriptive_interval95"],[0.,0.])

    def test_counted_per_section_rejects_incomplete(self):
        rows=[{"section":1,"seed":11,"method":"az","policy":"raw"}]
        with self.assertRaises(AssertionError):
            summarize(rows,"az",11,"raw")

if __name__=="__main__":
    unittest.main()
