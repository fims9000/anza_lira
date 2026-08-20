#!/usr/bin/env python3
"""Validate the frozen ANZA-KIR seed-41 result."""

from __future__ import annotations

import json

from anza_kir.validator import validate


if __name__ == "__main__": print(json.dumps(validate(), indent=2, sort_keys=True))
