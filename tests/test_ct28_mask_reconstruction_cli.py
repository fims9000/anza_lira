from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_mask_reconstruction_cli import run_from_files, load_route_record


def sample(tmp_path, patient=957):
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = True
    mask_path = tmp_path / "broken.npy"
    affine_path = tmp_path / "affine.npy"
    np.save(mask_path, mask)
    np.save(affine_path, np.eye(4))
    routes = tmp_path / "path.jsonl"
    routes.write_text(json.dumps({
        "scene_id": "s1", "scan_id": patient, "routes": [
            {"kind": "PAIR", "status": "ok", "paths_voxel": [
                [(x, 5, 5) for x in range(3, 10)]
            ]}
        ]
    }) + "\n")
    return routes, mask_path, affine_path


def test_cli_reconstructs_from_actual_path_bundle_shape(tmp_path):
    routes, mask, affine = sample(tmp_path)
    out_mask = tmp_path / "repaired.npy"
    report_path = tmp_path / "metrics.json"
    report = run_from_files(routes=routes, scene_id="s1", mask_file=mask,
                            affine_file=affine, output_mask=out_mask,
                            output_report=report_path, radius_mm=0,
                            max_new_voxels=20)
    assert report["accepted"]
    assert report["added_voxels"] == 5
    assert out_mask.exists() and report_path.exists()
    assert np.load(out_mask).sum() == 7
    with pytest.raises(FileExistsError):
        run_from_files(routes=routes, scene_id="s1", mask_file=mask,
                       affine_file=affine, output_mask=out_mask,
                       output_report=report_path, radius_mm=0,
                       max_new_voxels=20)


def test_cli_explicitly_blocks_heldout_test(tmp_path):
    route, *_ = sample(tmp_path, patient=954)
    with pytest.raises(ValueError, match="Held-out"):
        load_route_record(route, "s1")
