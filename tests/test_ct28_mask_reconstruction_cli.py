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


def test_no_truncation_of_noninteger_voxels_or_overwrite_of_inputs(tmp_path):
    routes, mask, affine = sample(tmp_path)
    from ct28_mask_reconstruction_cli import proposals_for_record
    source = json.loads(routes.read_text())
    source["routes"][0]["paths_voxel"][0][0][0] = 3.5
    with pytest.raises(ValueError, match="integer voxel"):
        from ct28_mask_reconstruction import apply_approved_paths
        apply_approved_paths(np.load(mask), np.load(affine), proposals_for_record(source),
                             radius_mm=0, max_new_voxels=20)
    with pytest.raises(ValueError, match="Cannot replace"):
        run_from_files(routes=routes, scene_id="s1", mask_file=mask,
                       affine_file=affine, output_mask=tmp_path / "out.npy",
                       output_report=mask, radius_mm=0, max_new_voxels=20)


def test_scene_mismatch_and_unsafe_route_rejected(tmp_path):
    routes, *_ = sample(tmp_path)
    obj = json.loads(routes.read_text())
    obj["scene_id"] = "CT28S:test:pair_only:957:0"
    routes.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="frozen train/validation"):
        load_route_record(routes, obj["scene_id"])
    obj["scene_id"] = "s1"
    obj["routes"][0]["extra_endpoint_hit"] = True
    routes.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="unsafe route"):
        load_route_record(routes, "s1")
