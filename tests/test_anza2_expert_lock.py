import json
from pathlib import Path


def test_phase0_data_contract_never_reads_expert_content() -> None:
    contract = json.loads(Path("results/anza2/phase0/data_contract.json").read_text())
    lock = contract["expert_lock"]
    assert lock["directory_exists"] is True
    assert lock["content_hashes_computed"] is False
    assert lock["pixels_read"] is False
    assert lock["scores_read"] is False
    assert lock["expert_data_accessed"] is False
