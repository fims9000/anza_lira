#!/usr/bin/env python3
from __future__ import annotations

from anza_leads.rc1_validator import validate_rc1


if __name__ == "__main__":
    result = validate_rc1(); print(f"ANZA LEADS RC1 VALIDATION: {result['status']}"); print(f"RESEARCH STATUS: {result.get('research_status')}")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
