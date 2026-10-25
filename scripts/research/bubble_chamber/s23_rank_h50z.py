#!/usr/bin/env python3
"""S23 optional expert review order for REAL H50Z; never label or join.

Uses learned geometric relation model trained on artificial visualMetro gaps.
The scores are out-of-domain ranking heuristics, not calibrated predictions.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from hashlib import sha256
import joblib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s19_real_review import load_package
from bubble_chamber.s21_selective_graph import acquire_human_pairs,_support_cut
from bubble_chamber.s23_review_rank import rank_unlabeled_real_pairs,insert_priority_in_workbench
from scripts.research.bubble_chamber.s21_run_safe_graph import row_split,dataset_view
from scripts.research.bubble_chamber.s22_gap_curriculum import fit_classifier


def build(args):
    manifest=audited_large_manifest(args.measurements)
    pools,split,availability=row_split(manifest,20261029)
    train=dataset_view([r for gap in (1,3,5) for r in pools[gap]['train']])
    ids,trail=acquire_human_pairs(train['rows'],train['context'],train['base'],640,seed=20261029,warm_start=60,batch_size=40,max_per_frame=4)
    model=fit_classifier(train['rows'],train['context'],ids,20261029)
    support=_support_cut(train['base'],.995)
    pkg=load_package(args.proposals,args.tiles)
    original=json.loads(args.s20_priority.read_text(encoding='utf-8'))
    updated=rank_unlabeled_real_pairs(pkg,original,model,support)
    original_html=args.workbench.read_text(encoding='utf-8')
    new_html=insert_priority_in_workbench(original_html,original,updated)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'S23_expert_questions_rank_only.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
    (args.out_dir/'S23_optional_workbench.html').write_text(new_html,encoding='utf-8')
    joblib.dump(model,args.out_dir/'S23_graph_ranker_geometry_only.joblib')
    sourcehash=sha256(args.proposals.read_bytes()).hexdigest()
    receipt={'schema':'s23_local_review_rank_receipt','source_visualmetro_measurements_sha256':manifest['provenance_sha256'],
             'expert_training_labels_simulated_from_saved_track_ids':640,
             'actual_real_photo_expert_labels':0,
             'actual_real_photo_expert_pair_labels':0,
             'additional_validation_labels_for_model_calibration':0,
             'original_photo_proposal_json_sha256':sourcehash,
             's19_source_package_sha256':pkg['package_sha256'],
             'model_is_trained_on_real_photo_pixels':False,
             'original_workbench_sha256':sha256(args.workbench.read_bytes()).hexdigest(),
             'output_workbench_sha256':sha256(new_html.encode()).hexdigest(),
             'output_kind':'ALTERNATE_HUMAN_REVIEW_PRIORITY_ONLY',
             'output_reviewer_approvals':0,'output_physical_join_permission':False,
             'review_pairs':len(updated['pair_questions']),
             'real_pairs_outside_observed_scale_support':updated['s23_protocol']['out_of_observed_gap_scale_count'],
             'real_pairs_insufficient_geometry':updated['s23_protocol']['insufficient_geometry_count'],
             'test_film_independent':False,'source_data_protection':'original BMP and review files never pushed to public Git'}
    (args.out_dir/'S23_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt


def main():
    p=argparse.ArgumentParser()
    for name in ('measurements','proposals','tiles','s20_priority','workbench','out_dir'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=p.parse_args();build(args)

if __name__=='__main__':main()
