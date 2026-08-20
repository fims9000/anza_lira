#!/usr/bin/env python3
from __future__ import annotations

import json

from anza_tracegraph.ports_v3.validator import validate


if __name__ == "__main__": print(json.dumps(validate(), indent=2, sort_keys=True))
