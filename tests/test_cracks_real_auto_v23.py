import unittest
import numpy as np
from scipy import ndimage as ndi
from scripts.cracks_real_auto_core_v23 import (
    anisotropic_features, extract_traces, propose_edges, score_edges,
    accept_edges, run_pipeline, trace_path
)
from scripts.cracks_real_auto_run_v23 import (
    consensus_training_mask, evaluate_mask,topology_measure,BLUE,ORANGE,zip_index
)


class AutonomousV23Tests(unittest.TestCase):
    def test_feature_tensor_is_finite_and_local(self):
        img=np.zeros((64,100),np.float32)
        img[:,40:45]=.9
        feats=anisotropic_features(img)
        self.assertEqual(feats.shape,(64,100,10))
        self.assertTrue(np.isfinite(feats).all())
        self.assertTrue(np.allclose(anisotropic_features(img),feats))

    def test_automatic_ports_and_link_no_oracle(self):
        prob=np.zeros((100,150),np.float32)
        prob[48,18:51]=.94
        prob[48,60:112]=.93
        _,traces,ports=extract_traces(prob,threshold=.7)
        self.assertEqual(len(traces),2)
        self.assertEqual(len(ports),4)
        edges=propose_edges(ports,max_gap=35)
        self.assertTrue(any(9<=e[2]<=11 for e in edges))
        self.assertTrue(len(accept_edges(score_edges(edges,ports,prob,strategy='geometry'),
                         ports,strategy='geometry'))>=1)
        result=run_pipeline(prob,threshold=.7,strategy='nearest')
        self.assertTrue(result['completed_mask'][48,54])
        self.assertGreater(result['completed_mask'].sum(),result['raw_mask'].sum())
        self.assertGreaterEqual(result['none_count'],0)
        # No model class access to expert truth or manually provided termini.

    def test_no_link_is_possible(self):
        prob=np.zeros((70,140),np.float32)
        prob[30,10:50]=.95
        result=run_pipeline(prob,threshold=.65,strategy='context')
        self.assertEqual(len(result['selected']),0)
        self.assertTrue(np.array_equal(result['raw_mask'],result['completed_mask']))
        self.assertEqual(result['none_count'],len(result['ports']))

    def test_path_dijkstra_is_contiguous_and_bounded(self):
        prob=np.ones((90,130),np.float32)*.5
        prob[44,20:100]=.9
        route=trace_path(prob,(44,21),(44,99))
        self.assertGreater(len(route),0)
        self.assertEqual(route[0],(44,21))
        self.assertEqual(route[-1],(44,99))
        self.assertTrue(all(max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1
                            for a,b in zip(route,route[1:])))

    def test_oracle_only_evaluation_does_not_change_inference(self):
        prob=np.zeros((100,145),np.float32)
        prob[52,20:62]=.9
        prob[52,69:120]=.9
        before=run_pipeline(prob,threshold=.65,strategy='nearest')['completed_mask'].copy()
        truth=np.zeros((100,145,3),np.uint8)
        truth[:]=ORANGE
        truth[52,20:120]=BLUE
        metric=evaluate_mask(before,truth)
        self.assertIn('expert_blue_recalled_tol2',metric)
        truth[:]=ORANGE
        truth[52,31:55]=BLUE
        self.assertTrue(np.array_equal(before,
                          run_pipeline(prob,threshold=.65,strategy='nearest')['completed_mask']))

    def test_connectivity_metric_monotone_under_added_prediction(self):
        gold=np.zeros((80,130),dtype=bool)
        gold[38,25:112]=True
        predicted=np.zeros_like(gold)
        predicted[36,25:112]=True
        before=topology_measure(predicted,gold)
        self.assertGreater(before['connected_gold_anchor_pairs'],0)
        # Adding a disconnected closer pixel cannot invalidate an existing link.
        grown=predicted.copy()
        grown[39,41]=True
        after=topology_measure(grown,gold)
        self.assertGreaterEqual(after['connected_gold_anchor_pairs'],
                                before['connected_gold_anchor_pairs'])

    def test_candidate_proposal_is_symmetric_and_finite(self):
        prob=np.zeros((80,120),np.float32)
        prob[28,10:39]=.92
        prob[28,48:83]=.93
        _,_,ports=extract_traces(prob,threshold=.6)
        candidates=propose_edges(ports,max_gap=40)
        self.assertTrue(all(i<j and d>=3 and np.isfinite(g)
                    for i,j,d,g in candidates))

if __name__=='__main__':unittest.main()
