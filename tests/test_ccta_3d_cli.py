"""NIfTI CLI smoke tests for genuine input/output interface; all data synthetic."""
from pathlib import Path
import json
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_cli import run_cli
from ccta_3d_runtime import HELD_OUT_CT28_IDS


def files(tmp_path):
    ct = np.full((40, 55, 32), 240.0, dtype=np.float32)
    mask = np.zeros(ct.shape, dtype=np.uint8)
    for y in (8, 14, 25, 36, 46):
        mask[10:20, y, 16] = 1
    affine = np.diag([.7, .7, 1.4, 1])
    affine[:3, 3] = [5, 8, -2]
    ct_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), ct_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return ct_path, mask_path


def kwargs(tmp_path, ct_path, mask_path, **changes):
    arguments = dict(
        ct_path=ct_path, mask_path=mask_path, patient_id="external_synthetic",
        mode="candidates", model_pack=None,
        output_mask=tmp_path / "repaired.nii.gz",
        report_path=tmp_path / "decisions.json",
        radius_mm=0, max_new_voxels=50,
        min_component_voxels=5,
        trace_mm=3, max_endpoints=256, max_scenes=128,
        threshold=.5,
    )
    return {**arguments, **changes}


def test_candidate_cli_roundtrip_non_destructive(tmp_path):
    ct_path, mask_path = files(tmp_path)
    before = np.asanyarray(nib.load(mask_path).dataobj) > 0
    report = run_cli(**kwargs(
        tmp_path, ct_path, mask_path,
        output_additions=tmp_path / "new_voxels.nii.gz",
    ))
    assert report["status"] == "candidate_inspection_only"
    assert report["accepted_decisions"] == 0
    assert report["endpoint_count"] == 10
    assert report["scene_count"] > 0
    assert report["input_provenance"]["ct_sha256"]
    after_image = nib.load(tmp_path / "repaired.nii.gz")
    assert np.array_equal(np.asanyarray(after_image.dataobj) > 0, before)
    assert np.allclose(after_image.affine, nib.load(mask_path).affine)
    assert json.loads((tmp_path / "decisions.json").read_text())["added_voxels"] == 0
    assert nib.load(tmp_path / "new_voxels.nii.gz").get_fdata().sum() == 0
    with pytest.raises(FileExistsError, match="Immutable"):
        run_cli(**kwargs(tmp_path, ct_path, mask_path))


def test_invalid_mask_affine_is_rejected(tmp_path):
    ct_path, mask_path = files(tmp_path)
    mask = nib.load(mask_path)
    wrong_affine = mask.affine.copy()
    wrong_affine[0, 3] += 3
    nib.save(nib.Nifti1Image(np.asanyarray(mask.dataobj), wrong_affine), mask_path)
    with pytest.raises(ValueError, match="affine mismatch"):
        run_cli(**kwargs(tmp_path, ct_path, mask_path))


def test_cli_requires_model_pack_and_blocks_test_before_opening_data(tmp_path):
    ct_path, mask_path = files(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        run_cli(**kwargs(tmp_path, ct_path, mask_path, mode="k10"))
    for patient in HELD_OUT_CT28_IDS:
        with pytest.raises(ValueError, match="Held-out"):
            run_cli(**kwargs(tmp_path, ct_path, mask_path, patient_id=patient))
