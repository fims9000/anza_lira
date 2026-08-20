from __future__ import annotations

import subprocess
import sys


def test_context_cli_status_is_read_only_and_lists_frozen_matrix() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_context_repair.py", "status"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "protocol_hash=bc197e6e9517532e" in completed.stdout
    for candidate in ("B0", "B1", "B2", "B3"):
        assert candidate in completed.stdout
