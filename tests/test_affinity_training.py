from dataclasses import replace

import torch

from affinity_repair.matrix import affinity_matrix
from affinity_repair.training import build_candidate, load_checkpoint, run_candidate


def test_each_candidate_one_epoch_smoke_and_resume(tmp_path):
    c1_checkpoint = None
    for spec in affinity_matrix():
        result = run_candidate(
            spec,
            tmp_path,
            device="cpu",
            stage1_epochs=1,
            stage2_epochs=1,
            train_samples=4,
            validation_samples=2,
            image_size=32,
            widths=(4, 6, 8, 10),
            clean_checkpoint=c1_checkpoint,
        )
        assert result["status"] == "COMPLETE"
        assert result["checkpoint_reload"] == "PASS"
        if spec.candidate_id == "C1":
            c1_checkpoint = tmp_path / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
    repeated = run_candidate(
        affinity_matrix()[-1], tmp_path, device="cpu", stage1_epochs=1, stage2_epochs=1,
        train_samples=4, validation_samples=2, image_size=32, widths=(4, 6, 8, 10),
        clean_checkpoint=c1_checkpoint,
    )
    assert repeated["action"] == "SKIP"


def test_checkpoint_hash_change_fails_closed(tmp_path):
    spec = affinity_matrix()[0]
    run_candidate(
        spec, tmp_path, device="cpu", stage2_epochs=1, train_samples=2,
        validation_samples=1, image_size=32, widths=(4, 6, 8, 10),
    )
    checkpoint = tmp_path / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
    changed = replace(spec, seed=41)
    model = build_candidate(changed, widths=(4, 6, 8, 10))
    try:
        load_checkpoint(checkpoint, spec=changed, model=model)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed config reused checkpoint")
