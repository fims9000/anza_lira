#!/usr/bin/env python3
"""Run only frozen Graph-Cut V2 benchmark or candidate phases."""

from __future__ import annotations

import argparse
import json

from lira_graph_cut_v2.figures import generate
from lira_graph_cut_v2.runner import build_benchmark, freeze, run_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "benchmark", "candidate", "figures"), required=True)
    args = parser.parse_args()
    result = {"freeze": freeze, "benchmark": build_benchmark, "candidate": run_candidate, "figures": generate}[args.phase]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
