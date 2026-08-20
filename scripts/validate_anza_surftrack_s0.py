#!/usr/bin/env python3
from anza_surftrack.validator import validate_s0

if __name__ == "__main__":
    result = validate_s0(); print(f"ANZA SURFTRACK S0 VALIDATION: {result['status']}"); print(f"RESEARCH STATUS: {result.get('research_status')}")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
