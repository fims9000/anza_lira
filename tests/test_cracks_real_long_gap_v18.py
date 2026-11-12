import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cracks_real_long_gap_v18 import (
    GAPS, OFFSETS, clustered_interval, make_curves, evaluate
)


class V18LongGapTests(unittest.TestCase):
    def test_long_straight_path_has_identical_hermite_baseline(self):
        path=[(i,50) for i in range(14,114)]
        for gap in GAPS:
            data=make_curves(path,gap)
            self.assertIsNotNone(data)
            straight,candidates,hidden,span,_=data
            self.assertEqual(len(hidden),gap)
            self.assertEqual(len(candidates),len(OFFSETS))
            self.assertLess(np.max(np.abs(straight-candidates[2])),1.e-8)
            self.assertEqual(span[1]-span[0],gap)

    def test_zero_insufficient_visible_support(self):
        path=[(i,30) for i in range(12,80)]
        self.assertIsNone(make_curves(path,48))
        self.assertIsNotNone(make_curves(path,24))

    def test_grouped_pair_bootstrap(self):
        rows=[
            {"section":3,"adjacent_mean_distance":1.,"straight_mean_distance":2.},
            {"section":3,"adjacent_mean_distance":1.5,"straight_mean_distance":2.5},
            {"section":7,"adjacent_mean_distance":2.,"straight_mean_distance":3.},
        ]
        result=clustered_interval(rows,"adjacent","straight",reps=100)
        self.assertAlmostEqual(result["delta"],-1.)
        self.assertEqual(result["N_sections"],2)
        self.assertAlmostEqual(result["ci95"][0],-1.)
        self.assertAlmostEqual(result["ci95"][1],-1.)

    def test_fixture_archive_and_oracle_flags(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            images=root/"images.zip"
            masks=root/"masks.zip"
            im=np.zeros((140,180),dtype=np.uint8)+127
            im[:,90:92]=230
            out=io.BytesIO();Image.fromarray(im).save(out,format="PNG")
            with zipfile.ZipFile(images,"w") as z:
                for sec in range(1,240):
                    z.writestr("images/section_%03d.png"%sec,out.getvalue())
            lab=np.zeros((140,180,3),dtype=np.uint8)+255
            lab[20:120,91]=[31,119,180]
            out=io.BytesIO();Image.fromarray(lab).save(out,format="PNG")
            with zipfile.ZipFile(masks,"w") as z:
                z.writestr("expert/section_060.png",out.getvalue())
            result=evaluate(images,masks,root/"v18.json",root/"cases.csv",400)
            self.assertEqual(result["status"],"EXECUTED_DESCRIPTIVE")
            self.assertFalse(result["full_ANZA_LIRA_e2e"])
            self.assertFalse(result["valid_negative_instance_pairs"])
            self.assertFalse(result["read_hidden_expert_mask_for_inference"])
            self.assertEqual(result["data_sections_with_expert_masks"],1)
            self.assertEqual(result["all"]["N"],len(GAPS))
            self.assertEqual(result["all"]["N_nontrivial_straight"],0)
            rows=list(csv.DictReader((root/"cases.csv").open()))
            self.assertEqual(len(rows),len(GAPS))


if __name__=="__main__":
    unittest.main()
