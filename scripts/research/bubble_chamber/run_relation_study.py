#!/usr/bin/env python3
"""Run a locked synthetic-only two-way relation comparison, no real labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.relation_study import Protocol, run_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-scenes', type=int, default=160)
    p.add_argument('--val-scenes', type=int, default=70)
    p.add_argument('--test-scenes', type=int, default=70)
    args = p.parse_args()
    if min(args.train_scenes, args.val_scenes, args.test_scenes) < 4:
        p.error('each split needs >=4 independent scenes')
    result = run_study(Protocol(train_scenes=args.train_scenes,val_scenes=args.val_scenes,test_scenes=args.test_scenes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False) + '\n',encoding='utf-8')
    print(json.dumps(result['results'],indent=2))


if __name__ == '__main__':
    main()
