"""S90 private, offline, blinded physicist image-appearance review (not GT)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bubble_chamber.s90_physicist_review import ReviewConfig, build_review_pack, verify_reviews


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='generate PRIVATE blinded expert review pack')
    build.add_argument('--s86-dir', type=Path, required=True)
    build.add_argument('--s89-dir', type=Path, required=True)
    build.add_argument('--bmp', type=Path, required=True)
    build.add_argument('--private-out-dir', type=Path, required=True)
    build.add_argument('--max-cases', type=int, default=24)
    build.add_argument('--max-per-tile', type=int, default=5)
    build.add_argument('--crop-size', type=int, default=192)
    verify = commands.add_parser('verify', help='validate exported reviewer JSON; NO GT promotion')
    verify.add_argument('--pack-dir', type=Path, required=True)
    verify.add_argument('--responses', type=Path, nargs='+', required=True)
    verify.add_argument('--private-out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        out = build_review_pack(args.s86_dir, args.s89_dir, args.bmp,
                                args.private_out_dir, ReviewConfig(args.max_cases,
                                                                  args.max_per_tile,
                                                                  args.crop_size))
    else:
        out = verify_reviews(args.pack_dir, args.responses, args.private_out_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
