from __future__ import annotations

import subprocess


def test_dry_run_prints_exact_bounded_matrix(tmp_path) -> None:
    completed = subprocess.run(
        [
            "/home/lebedeffson/Code/venv/bin/python",
            "scripts/run_method_repair.py",
            "dry-run",
            "--root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert [line.split()[0] for line in lines[:5]] == ["A0", "A1", "A2", "A3", "A4"]
    assert lines[-1].endswith("expert=LOCKED old_test=LOCKED new_test=LOCKED")
