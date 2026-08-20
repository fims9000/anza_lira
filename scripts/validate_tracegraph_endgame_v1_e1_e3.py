#!/usr/bin/env python3
"""Validate frozen TRACEGRAPH ENDGAME V1 E1--E3 artifacts."""

from __future__ import annotations

import json

from anza_tracegraph.endgame_v1.validators.e3 import validate


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
