from cracks_v2.split import build_grouped_oof_split


def test_grouped_oof_is_section_disjoint_and_covers_every_outer_once() -> None:
    sections = [value for value in range(1, 401) if value not in {9, 49, 73, 185, 249, 336, 385}]
    split = build_grouped_oof_split(sections)
    assert split["fold_count"] == 5
    assert split["outer_exactly_once"] is True
    assert split["outer_union_count"] == len(sections)
    assert split["expert_data_accessed"] is False
    for fold in split["folds"]:
        train = set(fold["train_section_ids"])
        dev = set(fold["dev_section_ids"])
        outer = set(fold["outer_section_ids"])
        assert not (train & dev or train & outer or dev & outer)
        assert all(abs(a - b) > 2 for a in train for b in outer)
