"""
Data-driven forecaster, trained on the OMNI2 reanalysis.

This is the "GraphCast/AIFS role" in the pipeline: a model that learns the
statistical relationship between the current + recent state and the state
72h later, purely from data, rather than physically simulating solar wind
propagation. Starting point is a histogram-based gradient-boosted tree
ensemble (fast to train, robust to the noisy/gappy nature of this data,
handles missing values natively — important since the live ACE feed can't
populate every feature the reanalysis can, e.g. flow pressure has to be
derived rather than measured directly). The same feature frame could later
feed a sequence model (temporal CNN/transformer) if more lead-time/skill is
needed — noted in ROADMAP.md.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "gbr_speed72h.joblib"


def train(X: pd.DataFrame, y: pd.Series, **kwargs) -> HistGradientBoostingRegressor:
    params = dict(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=0,
    )
    params.update(kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return model


def save(model: HistGradientBoostingRegressor, feature_columns: list[str]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_PATH)


def load() -> tuple[HistGradientBoostingRegressor, list[str]]:
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_columns"]


def exists() -> bool:
    return MODEL_PATH.exists()
