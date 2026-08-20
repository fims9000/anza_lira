#!/usr/bin/env python3
"""Run the frozen zero-training ANZA-S Phase A2 causal audit."""

from __future__ import annotations

import json

from anza_s.a2.run import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
