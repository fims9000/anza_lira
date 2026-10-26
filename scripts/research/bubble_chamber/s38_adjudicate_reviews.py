#!/usr/bin/env python3
"""PRIVATE A/B comparison with public-safe aggregate-only report.

No native physical track ground truth, model training targets or automatic
accepted joins are created; conflicts need explicit human resolution.
"""
from __future__ import annotations
import argparse,json,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bubble_chamber.s38_review_protocol import compare_sessions


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--private-source-zip',required=True,type=Path)
    a.add_argument('--review-a',required=True,type=Path)
    a.add_argument('--review-b',required=True,type=Path)
    a.add_argument('--private-adjudication-json',required=True,type=Path)
    a.add_argument('--public-aggregate-json',required=True,type=Path)
    opts=a.parse_args()
    with zipfile.ZipFile(opts.private_source_zip) as z:
        template=json.loads(z.read('annotations/S31_EMPTY_CORRECTIONS_TEMPLATE.json'))
        drafts=json.loads(z.read('annotations/S31_MACHINE_DRAFTS.json'))
        evidence=json.loads(z.read('review/S35_28_REAL_H50Z_PAIR_QUESTIONS_PRIVATE.json'))
    review_a=json.loads(opts.review_a.read_text(encoding='utf8'))
    review_b=json.loads(opts.review_b.read_text(encoding='utf8'))
    private,public=compare_sessions(template,drafts,evidence,review_a,review_b)
    opts.private_adjudication_json.parent.mkdir(parents=True,exist_ok=True)
    opts.public_aggregate_json.parent.mkdir(parents=True,exist_ok=True)
    opts.private_adjudication_json.write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    opts.public_aggregate_json.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(public,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
