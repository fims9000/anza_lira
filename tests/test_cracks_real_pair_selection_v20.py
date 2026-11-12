import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cracks_real_pair_selection_v20 import (
    conditional_case, candidate_features, decide, grouped_ci, experiment, METHODS
)


class TestV20ControlledPairing(unittest.TestCase):
    def test_correct_geometry_and_permutation(self):
        grads=[np.zeros((100,150),dtype=np.float32) for _ in range(5)]
        source=(30,30)
        src_tan=(0,6)
        targets=((30,43),(55,36))
        tangents=((0,6),(6,0))
        feature=candidate_features(source,src_tan,targets,tangents,grads)
        self.assertEqual(decide(feature,"nearest"),0)
        self.assertEqual(decide(feature,"heading"),0)
        self.assertEqual(decide(feature,"geometry"),0)
        self.assertEqual(decide(feature[::-1],"geometry"),1)
        self.assertTrue(all(decide(feature,m)==0 for m in METHODS))
    def test_visible_only_extraction(self):
        source=[(30,i) for i in range(15,117)]
        decoy=[(41,i) for i in range(15,117)]
        data=conditional_case(source,decoy,49,49)
        self.assertIsNotNone(data)
        e0,tan,targets,tans,span=data
        self.assertGreaterEqual(span[0],15)
        self.assertEqual(span[1]-span[0],12)
        self.assertNotEqual(tuple(targets[0]),tuple(targets[1]))
    def test_section_clustered_bootstrap(self):
        rows=[
            {"section":1,"geometry_adjacent_correct":1,"geometry_correct":0},
            {"section":1,"geometry_adjacent_correct":1,"geometry_correct":1},
            {"section":5,"geometry_adjacent_correct":0,"geometry_correct":1},
        ]
        ci=grouped_ci(rows,"geometry_adjacent","geometry",reps=200)
        self.assertAlmostEqual(ci["difference_accuracy"],0.0)
        self.assertEqual(ci["N_sections"],2)
    def test_invalid_unpinned_candidate_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            bad=root/"v19.csv"
            bad.write_text("malformed,labels\n",encoding="utf8")
            with self.assertRaisesRegex(RuntimeError,"V19 preselected candidate source hash"):
                experiment(root/"missing_images.zip",root/"missing_gt.zip",
                           bad,root/"out.json",root/"out.csv")


if __name__=="__main__":
    unittest.main()
