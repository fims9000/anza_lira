#!/usr/bin/env python3
"""PRIVATE S91: audit S90 reviewer agreement, build blinded follow-up queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.s91_review_consensus import ConsensusConfig, analyze_reviews


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', required=True, type=Path, help='original PRIVATE S90 pack')
    parser.add_argument('--verified-dir', required=True, type=Path, help='S90 verified responses')
    parser.add_argument('--s89-dir', required=True, type=Path, help='unaltered PRIVATE S89 audit')
    parser.add_argument('--private-out-dir', required=True, type=Path)
    parser.add_argument('--max-followup-cases', type=int, default=24)
    parser.add_argument('--no-followup', action='store_true')
    args = parser.parse_args(argv)
    result = analyze_reviews(args.pack_dir, args.verified_dir, args.s89_dir,
                             args.private_out_dir,
                             ConsensusConfig(args.max_followup_cases,
                                             not args.no_followup))
    print(json.dumps({key: result[key] for key in (
        'status', 'selected_candidate_count', 'unique_reviewer_count',
        'cases_reviewed_by_at_least_two', 'unresolved_selected_case_count',
        'followup_queue_selected_count', 'native_physics_track_GT_count',
        'no_representative_physical_precision_or_recall')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
