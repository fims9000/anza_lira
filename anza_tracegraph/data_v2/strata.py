"""Frozen TRACEGRAPH_RELATION_V2 strata and disjoint streams."""

POSITIVE_STRATA = (
    "straight_gap",
    "curved_gap",
    "s_curve_gap",
    "long_gap",
    "x_crossing_correct",
    "acute_crossing_correct",
    "t_junction_continue",
    "y_junction_continue",
    "weak_branch_continue",
    "close_parallel_continue",
    "low_contrast_continue",
    "partial_occlusion_continue",
    "multiple_plausible_correct",
    "cluttered_corridor_continue",
)

NEGATIVE_STRATA = (
    "none_isolated_end",
    "parallel_wrong_only",
    "x_wrong_only",
    "t_wrong_only",
    "y_wrong_only",
    "independent_collinear_fault",
)

STRATA = POSITIVE_STRATA + NEGATIVE_STRATA
SPLIT_SIZES = {"calibration": 3840, "development": 3840, "confirm": 3840}
SPLIT_SEEDS = {"calibration": 5_201_000_000, "development": 5_211_000_000, "confirm": 5_221_000_000}

MAIN_SAFETY_STRATA = (
    "x_crossing_correct",
    "acute_crossing_correct",
    "t_junction_continue",
    "y_junction_continue",
    "long_gap",
    "close_parallel_continue",
    "partial_occlusion_continue",
)
