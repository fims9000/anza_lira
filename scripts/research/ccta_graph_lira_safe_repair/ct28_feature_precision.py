"""Stable, importable local-feature precision transform for CT28 HGB."""

from __future__ import annotations

import pandas as pd


def round_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.round(6)
