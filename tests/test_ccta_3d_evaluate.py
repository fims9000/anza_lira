"""Synthetic NIfTI post-hoc evaluation: never trains or guides the solver."""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_evaluate import evaluate_files


def _write(path, content, affine):
    nib.save(nib.Nifti1Image(content.astype(np.uint8), affine), str(path))


def test_posthoc_nifti_evaluation_and_broken_groundtruth_isolation(tmp_path):
    ref = np.zeros((30,30,30), dtype=bool)
    ref[4:20, 11, 11] = True
    before = ref.copy()
    before[10:13, 11, 11] = False
    affine = np.diag([.7, .8, 1.5, 1.])
    originals = [tmp_path / n for n in ("before.nii.gz","after.nii.gz","ref.nii.gz")]
    for path, array in zip(originals, (before, ref, ref)):
        _write(path, array, affine)
    metrics = evaluate_files(patient_id="953", before=originals[0],
                             after=originals[1], reference=originals[2],
                             output=tmp_path/"metrics.json")
    assert metrics["dice_after"] == 1
    assert metrics["cldice_after"] == 1
    assert metrics["dice_delta"] > 0
    assert metrics["truth_used_for_inference"] is False
    assert metrics["added_voxels"] == 3
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_files(patient_id="953", before=originals[0],
                       after=originals[1], reference=originals[2],
                       output=tmp_path/"metrics.json")


def test_heldout_and_registration_guard(tmp_path):
    ref = np.zeros((8,8,8), dtype=bool)
    ref[2:4,2,2] = True
    inputs = [tmp_path/f"{x}.nii.gz" for x in ("b","a","r")]
    for name in inputs:
        _write(name, ref, np.eye(4))
    with pytest.raises(ValueError, match="Held-out"):
        evaluate_files(patient_id="954", before=inputs[0], after=inputs[1],
                       reference=inputs[2], output=tmp_path/"result.json")
    affine = np.eye(4)
    affine[0,3] = 5
    _write(inputs[-1], ref, affine)
    with pytest.raises(ValueError, match="affine mismatch"):
        evaluate_files(patient_id="953", before=inputs[0], after=inputs[1],
                       reference=inputs[2], output=tmp_path/"result.json")
