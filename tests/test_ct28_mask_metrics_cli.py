from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_mask_metrics_cli import run


def test_real_file_interface_and_test_firewall(tmp_path):
    before = np.zeros((12,12,12), dtype=bool)
    before[2,5,5] = before[9,5,5] = True
    gt = before.copy()
    gt[2:10,5,5] = True
    files = [tmp_path / f"{name}.npy" for name in ["before","after","reference","affine"]]
    for path, value in zip(files, (before, gt, gt, np.eye(4))):
        np.save(path, value)
    result = run(patient=957, before=files[0], after=files[1], reference=files[2], affine=files[3], output=tmp_path / "report.json")
    assert result["dice_after"] == 1.0
    assert result["dice_before"] < 1.0
    with pytest.raises(ValueError, match="Held-out"):
        run(patient=954, before=files[0], after=files[1], reference=files[2], affine=files[3], output=tmp_path / "heldout.json")
