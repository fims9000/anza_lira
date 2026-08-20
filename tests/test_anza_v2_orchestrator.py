import sys

import pytest

from scripts.anza_v2_study import _run_logged_phase


def test_logged_phase_records_output_and_fails_closed(tmp_path) -> None:
    _run_logged_phase("ok", [sys.executable, "-c", "print('phase=test status=PASS')"], tmp_path)
    assert "status=PASS" in (tmp_path / "ok.log").read_text()
    with pytest.raises(RuntimeError, match="failed"):
        _run_logged_phase("bad", [sys.executable, "-c", "raise SystemExit(3)"], tmp_path)
