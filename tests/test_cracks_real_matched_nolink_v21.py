import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cracks_real_matched_nolink_v21 import (
    construct_one, build_case, scores_for_candidates, costs,
    fit_cal_threshold, section_bootstrap, experiment, METHODS, V19_CSV_SHA256
)


class V21MatchedCompetitorsTests(unittest.TestCase):
    def setUp(self):
        # Two parallel tracks: same heading/tangent and near-equal distance
        # when the source gap is long enough.
        self.source=[(20,x) for x in range(110)]
        self.decoy=[(30,x) for x in range(110)]
        self.images=[np.zeros((80,140),np.float32) for _ in range(5)]

    def test_first_geometry_matched_pair_exists(self):
        case=construct_one(self.source,self.decoy,55)
        self.assertIsNotNone(case)
        self.assertIn(case["gap"],(36,48,24))
        self.assertGreaterEqual(case["false_match"]["log_distance_ratio"],0.)
        self.assertLessEqual(case["false_match"]["cosine_difference"],.06)
        self.assertLessEqual(case["false_match"]["tangent_difference"],.10)
        self.assertGreaterEqual(np.linalg.norm(case["false_1"]-case["true"]),5.)

    def test_geometry_has_no_hidden_label_input(self):
        case=construct_one(self.source,self.decoy,55)
        true=case["true"];fake=case["false_1"]
        feat=scores_for_candidates(case["e0"],case["t0"],
            [true,fake],[case["true_tan"],case["false_tan_1"]],
            self.images,case["gap"])
        self.assertEqual(len(feat),2)
        for method in METHODS:
            a=costs(feat,method)
            b=costs(feat[::-1],method)
            self.assertAlmostEqual(a[0],b[1])
            self.assertAlmostEqual(a[1],b[0])

    def test_positive_and_no_link_differ_only_in_candidates(self):
        case=construct_one(self.source,self.decoy,55)
        pos=build_case(7,0,1,0,0,1,case,self.images,"positive")
        self.assertIn(pos["correct_index"],(0,1))
        neg=build_case(7,0,1,0,0,1,case,self.images,"no_link")
        self.assertIsNotNone(neg)
        self.assertEqual(neg["correct_index"],-1)
        self.assertEqual(pos["source_endpoint"],neg["source_endpoint"])
        self.assertEqual(pos["gap"],neg["gap"])

    def test_calibration_never_uses_odd_section(self):
        # On even CAL, null score 0.9 forces tight threshold; odd synthetic
        # null deliberately has -100 and must NOT change CAL threshold.
        rows=[
            {"section":2,"kind":"positive","geometry_best_cost":0.2,
             "geometry_correct":1},
            {"section":2,"kind":"no_link","geometry_best_cost":0.9,
             "geometry_correct":-1},
            {"section":3,"kind":"no_link","geometry_best_cost":-100.,
             "geometry_correct":-1},
        ]
        report=fit_cal_threshold(rows,"geometry")
        self.assertEqual(report["N_null"],1)
        self.assertEqual(report["CAL_correct_accept"],1)
        self.assertLess(report["threshold"],0.9)
        self.assertGreater(report["threshold"],0.19)

    def test_bootstrap_groups_by_section(self):
        rows=[
            {"section":3,"adjacent_correct":1,"geometry_correct":0},
            {"section":3,"adjacent_correct":0,"geometry_correct":0},
            {"section":7,"adjacent_correct":0,"geometry_correct":1}
        ]
        r=section_bootstrap(rows,"adjacent","geometry",reps=50)
        self.assertAlmostEqual(r["delta_accuracy"],0.)
        self.assertEqual(r["N_sections"],2)

    def test_reject_unpinned_v19_source(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);source=p/"wrong.csv";source.write_text("invalid")
            with self.assertRaisesRegex(RuntimeError,"SHA256 mismatch"):
                experiment(p/"a.zip",p/"b.zip",source,p/"x.json",p/"x.csv")


if __name__=="__main__":
    unittest.main()
