import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cracks_real_mirrored_v22 import (
    reflect_about_axis, reflect_vector, tied_geometry,
    image_scores, choose, group_ci, evaluate, canonical_v21_digest,
)

class RealMirroredV22Tests(unittest.TestCase):
    def test_exact_geometry_equal_by_reflection(self):
        source=np.asarray([20.,30.])
        tangent=np.asarray([4.,14.])
        real=np.asarray([30.,70.])
        true_tangent=np.asarray([3.,12.])
        false=reflect_about_axis(real,source,tangent)
        fake_tangent=reflect_vector(true_tangent,tangent)
        tie=tied_geometry(source,tangent,real,true_tangent,
                          false,fake_tangent)
        self.assertLess(float(tie.max()),1e-10)
        self.assertGreater(np.linalg.norm(real-false),1.)
        original=reflect_about_axis(false,source,tangent)
        self.assertLess(np.linalg.norm(original-real),1e-10)

    def test_looks_only_at_image_values(self):
        a=np.zeros((100,110),np.float32)
        b=np.zeros_like(a)
        a[30:33,:]=3.
        source=np.array([20.,20.])
        t=np.linspace(.08,.92,48)
        paths=[
            source[:,None]*(1-t)[None,:]+np.asarray([31.,71.])[:,None]*t[None,:],
            source[:,None]*(1-t)[None,:]+np.asarray([60.,71.])[:,None]*t[None,:]
        ]
        scores,signals,prior=image_scores(paths,[a,b,b,b,b],
                            np.array([[30.]*12,[5.]*12]))
        self.assertEqual(signals.shape,(2,5))
        self.assertIn(choose(scores["central"]),(0,1))
        self.assertEqual(choose(scores["geometry_tie"]),0)
        rev,_,_=image_scores(paths[::-1],[a,b,b,b,b],
                              np.array([[30.]*12,[5.]*12]))
        self.assertAlmostEqual(scores["central"][0],rev["central"][1])

    def test_clustered_result_not_zero_count(self):
        rows=[
            {"section":1,"central_correct":1,"far_correct":0},
            {"section":1,"central_correct":0,"far_correct":0},
            {"section":7,"central_correct":1,"far_correct":1},
        ]
        r=group_ci(rows,"central","far")
        self.assertAlmostEqual(r["delta"],1/3)
        self.assertEqual(r["N_sections"],2)

    def test_numeric_semantic_digest_preserves_labels_and_ignores_last_bits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def mkfile(name,cosine,label):
                p=root/name
                with p.open("w",encoding="utf8",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=[
                        "kind","section","correct_index",
                        "matched_cos_difference","candidate_features_json",
                        "source_endpoint"])
                    writer.writeheader()
                    writer.writerow({
                        "kind":"positive","section":"7",
                        "correct_index":str(label),
                        "matched_cos_difference":str(cosine),
                        "candidate_features_json":json.dumps([{
                            "length":37.0,"heading":0.95577900872195}]),
                        "source_endpoint":json.dumps([20.0,10.0])
                    })
                return p
            a=mkfile("a.csv",0.01578371406153478,1)
            b=mkfile("b.csv",0.01578371406153467,1)
            c=mkfile("c.csv",0.01578371406153467,0)
            self.assertEqual(canonical_v21_digest(a),canonical_v21_digest(b))
            self.assertNotEqual(canonical_v21_digest(a),canonical_v21_digest(c))

    def test_refuse_unpinned_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            a=p/"v19.csv";b=p/"v21.csv"
            a.write_text("bad",encoding="utf8")
            b.write_text("bad",encoding="utf8")
            with self.assertRaisesRegex(RuntimeError,"pinned V21"):
                evaluate(p/"none.zip",p/"none2.zip",a,b,p/"x.json",p/"x.csv")

if __name__=="__main__":
    unittest.main()
