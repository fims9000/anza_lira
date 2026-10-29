#!/usr/bin/env python3
"""Recompute S89 generated-only hard-negative stress with frozen S88 gates."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s89_generated_stress import write_benchmark


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-per-variant',type=int,default=4)
    args=parser.parse_args(argv)
    result=write_benchmark(args.output,args.n_per_variant)
    print(result['schema'],result['protocol'])


if __name__=='__main__':
    main()
