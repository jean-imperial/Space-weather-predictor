"""
Feature engineering for the 72h-ahead solar wind speed forecast.

Built on the OMNI2 reanalysis (src.data.omni). Mirrors how a data-driven NWP
model (GraphCast-style) is trained: features summarise the recent state
(lags, rolling stats) plus a "climatology"-like recurrence signal — here the
27.27-day solar (synodic Carrington) rotation period, since corotating
interaction regions and coronal-hole high-speed streams are the dominant
source of 1-3 day-ahead predictability in solar wind speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_HOURS = 72
SYNODIC_ROTATION_HOURS = 27.2753 * 24  # Carrington/synodic solar rotation, ~654.6h

LAGS_HOURS = [0, 1, 3, 6, 12, 24, 48]
ROLL_WINDOWS_HOURS = [24, 72]

FEATURE_COLUMNS = [
    "speed",
    "density",
    "temperature",
    "b_avg",
    "bz_gsm",
    "flow_pressure",
]


def add_lag_features(df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in LAGS_HOURS:
            out[f"{col}_lag{lag}h"] = out[col].shift(lag)
        for win in ROLL_WINDOWS_HOURS:
            out[f"{col}_roll{win}h_mean"] = out[col].rolling(win, min_periods=win // 2).mean()
            out[f"{col}_roll{win}h_std"] = out[col].rolling(win, min_periods=win // 2).std()
    return out


def add_recurrence_feature(df: pd.DataFrame, column: str = "speed") -> pd.DataFrame:
    """Value one synodic solar rotation ago — the solar-wind analogue of a persistence/
    climatology baseline feature (corotating streams recur every ~27 days)."""
    out = df.copy()
    shift_steps = int(round(SYNODIC_ROTATION_HOURS))
    out[f"{column}_recur27d"] = out[column].shift(shift_steps)
    return out


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year encoding as a weak proxy for solar-cycle / seasonal HCS-tilt effects."""
    out = df.copy()
    doy = out.index.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def build_supervised_frame(
    omni_df: pd.DataFrame, horizon_hours: int = HORIZON_HOURS
) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the OMNI2 hourly reanalysis into an (X, y) supervised learning frame.

    y = speed at t + horizon_hours (the operational forecast target).
    X = engineered features available at time t.
    """
    df = add_lag_features(omni_df)
    df = add_recurrence_feature(df)
    df = add_cyclic_time_features(df)
    target = df["speed"].shift(-horizon_hours).rename("target_speed")

    feature_cols = [c for c in df.columns if c not in omni_df.columns or c in ("speed",)]
    # keep engineered columns + the raw current-time driver columns
    feature_cols = sorted(set(feature_cols) | {"speed", "density", "temperature", "b_avg", "bz_gsm"})
    X = df[feature_cols]

    frame = pd.concat([X, target], axis=1).dropna()
    return frame[feature_cols], frame["target_speed"]


def build_inference_features(
    recent_df: pd.DataFrame,
    feature_columns: list[str],
    recurrence_value: float | None = None,
    cadence_minutes: int = 1,
) -> pd.DataFrame:
    """Build a single-row feature vector for operational inference from a recent (short) window
    of live ACE data (1-minute cadence). Because the live feed only spans ~2h (see src.data.ace),
    lag/rolling features that reach further back than the available window fall back to the
    oldest/latest value we do have — a documented limitation, see ROADMAP.md.
    """
    df = recent_df.copy()
    latest = df.iloc[-1]
    steps_per_hour = 60 // cadence_minutes

    row = {}
    for col in ("speed", "density", "temperature", "b_avg", "bz_gsm", "flow_pressure"):
        if col not in df.columns:
            row[col] = np.nan
            continue
        series = df[col].dropna()
        row[col] = latest.get(col, np.nan)

        for lag in LAGS_HOURS:
            key = f"{col}_lag{lag}h"
            if key not in feature_columns:
                continue
            steps_back = lag * steps_per_hour
            idx = len(series) - 1 - steps_back
            row[key] = series.iloc[max(idx, 0)] if len(series) else np.nan

        for win in ROLL_WINDOWS_HOURS:
            mean_key = f"{col}_roll{win}h_mean"
            std_key = f"{col}_roll{win}h_std"
            # available window is much shorter than `win` hours in practice;
            # use whatever history we have as the best available estimate.
            if mean_key in feature_columns:
                row[mean_key] = series.mean() if len(series) else np.nan
            if std_key in feature_columns:
                row[std_key] = series.std() if len(series) > 1 else np.nan

    row["speed_recur27d"] = recurrence_value if recurrence_value is not None else np.nan
    ts = df.index[-1]
    doy = pd.Timestamp(ts).dayofyear
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out = pd.DataFrame([row])
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[feature_columns]
