#!/usr/bin/env python3
"""Build offline H50Z expert annotator and compile real expert decisions if supplied.

This entrypoint never labels real-image pixels automatically, even if Frangi or
ANZA proposed a line. Before actual reviews all 7 tiles are UNKNOWN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s19_real_review import (
    load_package, make_offline_review_html, empty_review, compile_review,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals',type=Path,required=True,help='S15 proposals.json generated from the real BMPs')
    parser.add_argument('--images',type=Path,required=True,help='Directory containing the original 512x512 grayscale PNG crops')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--reviews-json',type=Path,help='ONLY actual exported human reviewer decisions (never invent these)')
    args=parser.parse_args()
    package=load_package(args.proposals,args.images)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'S19_review_workbench.html').write_text(make_offline_review_html(package,args.images),encoding='utf-8')
    (out/'review_package.json').write_text(json.dumps(package,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'blank_all_unknown.json').write_text(json.dumps(empty_review(package),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.reviews_json:
        human=json.loads(args.reviews_json.read_text(encoding='utf-8'))
        report=compile_review(package,human,out_dir=out/'human_compiled',image_root=args.images)
    else:
        report=compile_review(package,empty_review(package))
    (out/'S19_readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'workbench':str(out/'S19_review_workbench.html'),'real_crops':len(package['tiles']),
                      'human_verified_positive_pixels':report['confirmed_positive_pixels'],
                      'human_verified_negative_pixels':report['confirmed_negative_pixels'],
                      'human_verified_pairs':report['reviewed_relation_labels'],
                      'can_train_sparse_real_image_model':report['trainable_real_image_baseline'],
                      'certified_real_track_merges':0},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
