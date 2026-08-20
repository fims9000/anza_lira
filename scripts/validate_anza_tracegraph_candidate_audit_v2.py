#!/usr/bin/env python3
import json

from anza_tracegraph.candidate_audit_v2 import RESULT
from anza_tracegraph.candidate_audit_v2_validator import validate

if __name__ == "__main__":
    result = validate(); (RESULT / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))

