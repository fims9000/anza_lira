#!/usr/bin/env python3
"""Build a source-locked native H50Z review pack or evaluate human strokes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.s107_review import build_pack, evaluate, validate_annotation, verify_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='cmd', required=True)
    build = commands.add_parser('build')
    build.add_argument('--source-dir', type=Path, required=True)
    build.add_argument('--s105-root', type=Path, default=None,
                       help='Optional stored S105 previews; unused for blind ROI selection')
    build.add_argument('--out-dir', type=Path, required=True)
    build.add_argument('--seed', type=int, default=20261110)
    build.add_argument('--no-blind', action='store_true')
    verify = commands.add_parser('verify')
    verify.add_argument('--pack-dir', type=Path, required=True)
    score = commands.add_parser('evaluate')
    score.add_argument('--pack-dir', type=Path, required=True)
    score.add_argument('--review-json', type=Path, required=True)
    score.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == 'build':
        m = build_pack(args.source_dir, args.s105_root, args.out_dir, seed=args.seed,
                       blind_per_image=0 if args.no_blind else 4)
        template = ROOT / 'assets' / 's107_review.html'
        if not template.is_file():
            raise FileNotFoundError(template)
        shutil.copyfile(template, args.out_dir / 'index.html')
        html = (args.out_dir / 'index.html').read_text(encoding='utf-8')
        html = html.replace('__REVIEW_MANIFEST_JSON__',
                            json.dumps(m, ensure_ascii=False).replace('</', '<\\/'))
        (args.out_dir / 'index.html').write_text(html, encoding='utf-8')
        print(json.dumps({'status':'REVIEW_PACK_CREATED','tile_count':len(m['tiles']),
                          'manifest_sha256':m['manifest_sha256'],
                          'open':str(args.out_dir/'index.html')},ensure_ascii=False,indent=2))
    elif args.cmd == 'verify':
        m = verify_pack(args.pack_dir)
        print(json.dumps({'status':'VERIFIED','tile_count':len(m['tiles']),
                          'manifest_sha256':m['manifest_sha256']},indent=2))
    else:
        m = verify_pack(args.pack_dir)
        export = json.loads(args.review_json.read_text(encoding='utf-8'))
        validate_annotation(export, m)
        r = evaluate(export, args.pack_dir, args.out_dir)
        print(json.dumps({'status':'PARTIAL_REFERENCE_EVALUATION',
                          'positive_coverage':r['s105_centerline_coverage_on_explicitly_marked_strokes'],
                          'artifact_overlap':r['s105_candidate_overlap_on_explicitly_marked_artifacts'],
                          'global_precision_recall_claimed':False,
                          'output':str(args.out_dir/'report.json')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
