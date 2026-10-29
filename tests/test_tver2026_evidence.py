"""Frozen research evidence must not be silently reinterpreted."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.tver2026.check_evidence import check_evidence
from tools.tver2026.check_background_rng import audit

class FrozenEvidence(unittest.TestCase):
    def test_source_anchors_and_ratios(self):
        self.assertEqual(check_evidence()["status"], "FROZEN_SOURCE_ANCHORS_AND_ARITHMETIC_PASS")
    def test_p0_oracle_ceiling_not_equated_to_actual(self):
        a = check_evidence()
        self.assertGreater(a["p0_fixed_decision_candidate_oracle_ceiling"], a["p0_relation_recovery"])
        self.assertLess(a["p0_fixed_decision_candidate_oracle_ceiling"], .50)
    def test_generator_random_draw_order(self):
        self.assertEqual(audit(128)["matched_background_frequencies"], 0)

if __name__ == "__main__": unittest.main()
