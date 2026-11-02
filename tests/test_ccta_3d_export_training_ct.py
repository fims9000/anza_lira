"""Never export a held-out CCTA through the train-only archive helper."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_export_training_ct import export_training_ct


def test_export_blocks_test_and_validation_before_archive_access(tmp_path):
    for patient in (954, 957, 961, 972, 980):
        with pytest.raises(ValueError, match="training"):
            export_training_ct(patient, tmp_path / "nonexistent_archive",
                               tmp_path / "ct.nii.gz", tmp_path / "ct.json")
