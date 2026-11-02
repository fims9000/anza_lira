"""Predeclare immutable five-patient val pipeline without accessing CCTA scans."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_val_freeze import (
    build_validation_freeze,verify_validation_freeze,sha,
    DEFAULT_SETTINGS
)


def test_five_val_protocol_can_be_frozen_and_reverified(tmp_path):
    model=tmp_path/"synthetic_K10_model_pack"
    model.write_bytes(b"synthetic model; not executable on real CT")
    path=tmp_path/"path_protocol.json"
    path.write_text("{}")
    output=tmp_path/"protocol.json"
    verified=build_validation_freeze(
        output=output,model=model,path_protocol=path,
        required_model_sha=sha(model),
    )
    assert verified["patients"]==[957,961,965,966,974]
    assert verified["patient_count"]==5
    assert verified["settings"]==DEFAULT_SETTINGS
    assert verified["held_out_test_accessed"] is False
    assert len(verified["expected_validation_CT_SHA256"])==5
    assert verify_validation_freeze(
        output,model=model,path_protocol=path,
        required_model_sha=sha(model),
    )==verified
    with pytest.raises(FileExistsError):
        build_validation_freeze(
            output=output,model=model,path_protocol=path,
            required_model_sha=sha(model),
        )


def test_freeze_rejects_model_and_configuration_changes(tmp_path):
    model=tmp_path/"model"
    model.write_bytes(b"frozen synthetic")
    protocol=tmp_path/"frozen_path.json"
    protocol.write_text("{}")
    dest=tmp_path/"frozen_val.json"
    checksum=sha(model)
    build_validation_freeze(output=dest,model=model,path_protocol=protocol,
                            required_model_sha=checksum)
    changed=json.loads(dest.read_text())
    changed["settings"]["gap_radius_mm"]=2.9
    dest.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="immutable freeze"):
        verify_validation_freeze(dest,model=model,path_protocol=protocol,
                                 required_model_sha=checksum)
    dest.write_text(json.dumps({
        **changed,"settings":DEFAULT_SETTINGS
    }))
    model.write_bytes(b"model was changed after freezing")
    with pytest.raises(ValueError,match="SHA"):
        verify_validation_freeze(dest,model=model,path_protocol=protocol,
                                 required_model_sha=checksum)
