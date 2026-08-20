import numpy as np

from anza2.phase3d.structural_sampler import MANDATORY_STRATA, balanced_curriculum_indices, strata_inventory
from anza2.phase3d.visible_mode_targets import split_visible_and_latent_targets
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def _fake_rows():
    return [
        {"split": split, "case": case, "index": offset + index}
        for split, offset in (("train", 0), ("validation", 1000), ("confirm", 2000))
        for index, case in enumerate(MANDATORY_STRATA.values())
    ]


def test_exact_strata_quota_and_no_contiguous_prefix_sampling():
    schedule = balanced_curriculum_indices(_fake_rows(), quota=4, seed=7)
    counts = {name: sum(row["stratum"] == name for row in schedule) for name in MANDATORY_STRATA}
    assert counts == {name: 4 for name in MANDATORY_STRATA}
    assert [row["index"] for row in schedule] != list(range(len(schedule)))


def test_split_inventories_are_disjoint_by_index_namespace():
    rows = _fake_rows()
    train = {row["index"] for row in rows if row["split"] == "train"}
    validation = {row["index"] for row in rows if row["split"] == "validation"}
    confirm = {row["index"] for row in rows if row["split"] == "confirm"}
    assert not (train & validation or train & confirm or validation & confirm)
    assert all(strata_inventory(rows, split="train").values())


def test_visible_targets_exclude_privileged_positive_gap_axes():
    sample = generate_sample_v4("train", 0, image_size=64)
    targets = split_visible_and_latent_targets(sample)
    gap = np.asarray(sample["positive_gap_mask"], dtype=bool)
    assert targets["latent_continuation_theta_valid"][:, gap].any()
    assert not targets["visible_theta_valid"][:, gap].any()
    assert not (targets["visible_theta_valid"] & targets["latent_continuation_theta_valid"]).any()


def test_no_cracks_or_expert_inputs_exist_in_phase3d_data_module():
    import inspect
    import anza2.phase3d.case_manifest as module
    source = inspect.getsource(module).lower()
    assert "cracks" not in source and "expert" not in source
