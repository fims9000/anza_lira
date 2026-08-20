#!/usr/bin/env python3
import json
from pathlib import Path

from anza_tracegraph.validator import validate
from anza_tracegraph.runner import RESULT

if __name__ == "__main__":
    result = validate()
    (RESULT / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
