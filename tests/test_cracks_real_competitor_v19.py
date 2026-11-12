import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cracks_real_competitor_audit_v19 import (
    closest_interior_pair, audit, THRESHOLDS
)

class CompetitorV19Tests(unittest.TestCase):
    def test_near_distinct_tracks(self):
        a=[(y,40) for y in range(20,105)]
        b=[(y,50) for y in range(20,105)]
        d,ia,ib=closest_interior_pair(a,b)
        self.assertAlmostEqual(d,10.)
        self.assertGreaterEqual(ia,15)
        self.assertGreaterEqual(ib,15)
    def test_no_pair_without_visible_support(self):
        a=[(y,40) for y in range(20,50)]
        b=[(y,50) for y in range(20,105)]
        self.assertIsNone(closest_interior_pair(a,b))
    def test_real_archive_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);zippath=root/"annotation.zip"
            arr=np.zeros((130,170,3),dtype=np.uint8)+255
            arr[20:110,40]=[31,119,180]
            arr[20:110,50]=[31,119,180]
            out=io.BytesIO();Image.fromarray(arr).save(out,format="PNG")
            with zipfile.ZipFile(zippath,"w") as z:
                z.writestr("labels/expert/section_060.png",out.getvalue())
            report=audit(zippath,root/"v19.json",root/"cases.csv",400)
            self.assertEqual(report["denominator_expert_sections"],1)
            self.assertEqual(report["denominator_distinct_pairs"],1)
            self.assertEqual(report["threshold_counts"]["12"]["pairs"],1)
            self.assertEqual(report["threshold_counts"]["6"]["pairs"],0)
            self.assertEqual(report["potential_ambiguities_le20"],1)
            self.assertFalse(report["valid_geological_negative_pairs"])
            self.assertFalse(report["full_ANZA_LIRA_e2e"])

if __name__=="__main__":
    unittest.main()
