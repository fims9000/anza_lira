from __future__ import annotations

import json

from method_repair.audit import profile_frozen_models, run_forensic_audit


def test_forensic_audit_is_idempotent_and_fail_closed(tmp_path) -> None:
    output = tmp_path / "baseline.json"
    first = run_forensic_audit(output)
    first_bytes = output.read_bytes()
    second = run_forensic_audit(output)

    assert output.read_bytes() == first_bytes
    assert first == second == json.loads(output.read_text())
    assert first["expert_data_accessed"] is False
    assert first["training_started"] is False
    assert first["legacy_code_modified"] is False
    assert first["cracks_white_semantics"]["status"] == "NOT_ESTABLISHED"
    assert first["implementation_facts"]["H5_declared_controls"] == {
        "junction_score_returned_as_diagnostic": True,
        "junction_score_changes_forward": False,
        "cone_flag_changes_forward": False,
    }


def test_frozen_deadline_artifacts_have_expected_hashes(tmp_path) -> None:
    payload = run_forensic_audit(tmp_path / "baseline.json")
    assert payload["frozen_deadline_sha256"] == {
        "deadline_zip": "1f3e99bfedfddc33160db84c49b0954e77fea6086bd99a8fdafbbbf822e6eebe",
        "main_cracks": "fe6365b7c53922f2a56de4eaafa03a04145ff525a2ad031b574ce1ab8c8f2fbf",
        "thesis_numbers": "a107848d457a41735fb473be48d70b0348f2d27421be2be4f58e32a8a61f61b1",
    }


def test_runtime_profile_rejects_invalid_budget() -> None:
    try:
        profile_frozen_models(image_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid profiling budget was accepted")
