import numpy as np
import pandas as pd

from src.features import (
    add_cyclic_time_features,
    add_lag_features,
    add_recurrence_feature,
    build_supervised_frame,
)


def _toy_omni_frame(n_hours=2000):
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "speed": 400 + 50 * np.sin(np.arange(n_hours) / 24) + rng.normal(0, 5, n_hours),
            "density": rng.uniform(1, 10, n_hours),
            "temperature": rng.uniform(1e4, 2e5, n_hours),
            "b_avg": rng.uniform(1, 10, n_hours),
            "bz_gsm": rng.normal(0, 3, n_hours),
            "flow_pressure": rng.uniform(0.5, 3, n_hours),
        },
        index=idx,
    )


def test_add_lag_features_shifts_correctly():
    df = _toy_omni_frame(100)
    out = add_lag_features(df, columns=["speed"])
    assert "speed_lag1h" in out.columns
    # lag1h at row i should equal the raw value at row i-1
    assert np.isclose(out["speed_lag1h"].iloc[10], df["speed"].iloc[9])


def test_add_recurrence_feature_shifts_27_days():
    df = _toy_omni_frame(2000)
    out = add_recurrence_feature(df)
    shift_steps = int(round(27.2753 * 24))
    assert np.isclose(out["speed_recur27d"].iloc[shift_steps + 5], df["speed"].iloc[5])


def test_cyclic_time_features_bounded():
    df = _toy_omni_frame(50)
    out = add_cyclic_time_features(df)
    assert out["doy_sin"].between(-1, 1).all()
    assert out["doy_cos"].between(-1, 1).all()


def test_build_supervised_frame_target_is_future_speed():
    df = _toy_omni_frame(2000)
    X, y = build_supervised_frame(df, horizon_hours=72)
    assert len(X) == len(y)
    assert len(X) > 0
    # no leakage: target column itself must not appear as a feature
    assert "target_speed" not in X.columns
    # spot check: target at some row equals the raw speed 72h later
    ts = X.index[100]
    expected = df.loc[ts + pd.Timedelta(hours=72), "speed"]
    assert np.isclose(y.loc[ts], expected)
