"""
Baselines that any ML model must beat to be worth deploying — exactly how
NWP verification always reports skill relative to persistence and climatology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(speed_now: pd.Series) -> pd.Series:
    """Naive: assume speed stays constant over the forecast horizon."""
    return speed_now


def recurrence27d_forecast(speed_27d_ago: pd.Series) -> pd.Series:
    """27.27-day synodic recurrence: assume the value one solar rotation ago repeats.
    A much stronger baseline than persistence at a 72h horizon for corotating streams."""
    return speed_27d_ago


def mae(y_true, y_pred) -> float:
    return float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
