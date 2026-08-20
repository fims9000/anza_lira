#!/usr/bin/env python3
"""Run only V1.1 SS1.5 pre-training audits; never start B0-B3 training."""

from __future__ import annotations

import json

from structural_stability_v1_1.runner import run_pretraining_hardening


if __name__ == "__main__":
    print(json.dumps(run_pretraining_hardening(), indent=2, sort_keys=True))
