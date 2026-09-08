"""
Train the 72h-ahead solar wind speed model on the OMNI2 reanalysis and
report skill against the persistence and 27-day recurrence baselines —
the space-weather equivalent of verifying an ML weather model against
persistence/climatology.

Usage:
    python -m scripts.train
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import omni
from src.features import build_supervised_frame
from src.models import baselines, gbr

TRAIN_YEARS = list(range(2015, 2025))  # 2015-2024: training corpus
TEST_YEARS = [2025]  # 2025: held-out verification period, like an NWP skill season


def main():
    print("Loading OMNI2 reanalysis (training + test years)...")
    df = omni.load_years(TRAIN_YEARS + TEST_YEARS)
    print(f"  {len(df)} hourly records, {df.index.min()} .. {df.index.max()}")

    print("Building supervised (lag/rolling/recurrence) feature frame...")
    X, y = build_supervised_frame(df)
    split_date = pd.Timestamp(f"{TEST_YEARS[0]}-01-01")
    train_mask = X.index < split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"  train: {len(X_train)} rows, test: {len(X_test)} rows")

    print("Training gradient-boosted forecaster...")
    model = gbr.train(X_train, y_train)
    gbr.save(model, list(X.columns))

    pred_ml = model.predict(X_test)
    pred_persist = baselines.persistence_forecast(X_test["speed"])
    pred_recur = baselines.recurrence27d_forecast(X_test["speed_recur27d"])

    print("\n=== 72h-ahead solar wind speed forecast skill (2025 held-out test) ===")
    print(f"{'Model':<28}{'MAE (km/s)':>12}{'RMSE (km/s)':>13}")
    for name, pred in [
        ("Persistence", pred_persist),
        ("27-day recurrence", pred_recur),
        ("Gradient-boosted (this repo)", pred_ml),
    ]:
        print(
            f"{name:<28}{baselines.mae(y_test, pred):>12.1f}"
            f"{baselines.rmse(y_test, pred):>13.1f}"
        )

    print("\nComputing permutation importance on a test sample (this may take a moment)...")
    from sklearn.inspection import permutation_importance

    sample = X_test.sample(min(2000, len(X_test)), random_state=0)
    y_sample = y_test.loc[sample.index]
    result = permutation_importance(model, sample, y_sample, n_repeats=3, random_state=0, n_jobs=-1)
    top_features = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 features by permutation importance:")
    print(top_features.head(10).to_string())


if __name__ == "__main__":
    main()
