import unittest

import numpy as np

from scripts.tver_synthetic_risk_v29 import (
    FAMILIES,SHAPE,THRESHOLD,
    generate_group,raw_decision,evaluate,_filtered_mask,_mask_connected,
)


class V29SyntheticLinkTests(unittest.TestCase):
    def test_counterfactual_truth_differs_but_observation_exactly_same(self):
        bank=generate_group("CAL",0)
        self.assertEqual(tuple(x.condition for x in bank),FAMILIES)
        pos,near,negative,distinct=bank
        self.assertTrue(pos.physically_connected)
        self.assertFalse(negative.physically_connected)
        self.assertTrue(np.array_equal(pos.probability,negative.probability))
        self.assertFalse(np.array_equal(pos.probability,distinct.probability))
        self.assertEqual(pos.probability.shape,SHAPE)
        self.assertEqual(pos.left_anchor,negative.left_anchor)
        self.assertEqual(pos.right_anchor,negative.right_anchor)
        self.assertLess(float(np.min(pos.probability)),THRESHOLD)
        self.assertGreater(float(np.max(pos.probability)),THRESHOLD)

    def test_all_synthetic_gap_length_bands_represented_without_leakage(self):
        bands={generate_group("CAL",j)[0].gap_band for j in range(3)}
        self.assertEqual(bands,{"short_12px","mid_22px","long_32px"})
        for j in range(3):
            a,b=generate_group("CAL",j),generate_group("TEST",j)
            self.assertFalse(np.array_equal(a[0].probability,b[0].probability))
            self.assertTrue(np.array_equal(a[0].probability,
                                           generate_group("CAL",j)[0].probability))

    def test_original_repo_policies_see_no_ground_truth_data(self):
        easy,near,indist,negative=generate_group("CAL",9)
        for method in ("V23_nearest","V23_geometry",
                       "V23_context","V25_region_hybrid"):
            p=raw_decision(easy,method)
            q=raw_decision(indist,method)
            self.assertTrue(np.array_equal(p["completed_mask"],q["completed_mask"]))
            self.assertTrue(np.array_equal(p["raw_mask"],q["raw_mask"]))
            self.assertEqual(len(p["selected"]),len(q["selected"]))
            self.assertIn("ports",p)
            self.assertIn("candidates",p)
            self.assertGreaterEqual(evaluate(easy,method,p)["correct_connected"],0)

    def test_no_link_is_zero_false_bridge_and_does_not_derive_latent_endpoints(self):
        for condition in generate_group("CAL",23):
            p=raw_decision(condition,"NO_LINK")
            e=evaluate(condition,"NO_LINK",p)
            self.assertEqual(e["newly_selected_links"],0)
            self.assertEqual(e["false_bridge"],0)
            self.assertEqual(e["correct_connected"],0)

    def test_reject_only_monotonic_and_no_image_modification(self):
        scene=generate_group("CAL",5)[0]
        for method in ("V23_nearest","V23_geometry",
                       "V23_context","V25_region_hybrid"):
            p=raw_decision(scene,method)
            raw=np.asarray(p["raw_mask"],bool)
            allaccepted=_filtered_mask(p,-1e6)
            noaccepted=_filtered_mask(p,1e6)
            self.assertTrue(np.array_equal(noaccepted,raw))
            self.assertTrue(np.all(raw<=allaccepted))
            a=evaluate(scene,method,p,-1e6)
            b=evaluate(scene,method,p,1e6)
            self.assertGreaterEqual(a["newly_selected_links"],
                                    b["newly_selected_links"])

    def test_right_anchor_never_invisible_in_original_synthetic_mask(self):
        for j in range(10):
            for scene in generate_group("CAL",j):
                base=scene.probability>=THRESHOLD
                self.assertTrue(base[scene.left_anchor])
                self.assertTrue(base[scene.right_anchor])
                self.assertFalse(_mask_connected(
                    base,scene.left_anchor,scene.right_anchor))


if __name__=="__main__":
    unittest.main()
