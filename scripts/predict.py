"""
Produce one operational-style forecast: solar wind speed at L1, 72h from now,
initialised from the live ACE feed — the "run the trained model from today's
analysis" step, mirroring how an operational NWP forecast is initialised from
the live analysis rather than from ERA5.

Usage:
    python -m scripts.predict
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import ace, omni
from src.features import HORIZON_HOURS, build_inference_features
from src.models import baselines, gbr


def main():
    if not gbr.exists():
        print("No trained model found — run `python -m scripts.train` first.")
        return

    model, feature_columns = gbr.load()

    print("Fetching live ACE real-time feed (NOAA SWPC)...")
    live = ace.fetch_current()
    history = ace.append_to_history(live)
    print(f"  latest sample: {live.index.max()} UTC, {len(live)} rows this fetch")
    print(f"  local rolling ACE archive now spans {len(history)} rows "
          f"({history.index.min()} .. {history.index.max()})")

    # rename ACE columns to match the OMNI feature naming used in training
    recent = live.rename(columns={"bt": "b_avg"})
    # ACE's real-time feed doesn't report flow pressure directly; derive it the same
    # way OMNI2 defines it (ignoring the small alpha-particle correction term, which
    # the real-time feed has no data for): P (nPa) = 1.6726e-6 * Np(cm^-3) * V(km/s)^2
    recent["flow_pressure"] = 1.6726e-6 * recent["density"] * recent["speed"] ** 2

    # 27-day recurrence feature: needs data from ~27 days ago. The live ACE archive
    # won't have that until it's been running a while (see ROADMAP.md); fall back to
    # the most recent definitive OMNI2 value as a stand-in, and flag it clearly.
    recurrence_value = None
    recurrence_source = "unavailable"
    target_recur_time = live.index.max() - pd.Timedelta(days=27, hours=6, minutes=36)
    if len(history) and (history.index.min() <= target_recur_time):
        nearest = history.index[history.index.get_indexer([target_recur_time], method="nearest")[0]]
        recurrence_value = history.loc[nearest, "speed"]
        recurrence_source = f"local ACE archive @ {nearest}"
    else:
        try:
            odf = omni.load_years([target_recur_time.year])
            naive_target = target_recur_time.tz_localize(None)
            nearest_idx = odf.index[odf.index.get_indexer([naive_target], method="nearest")[0]]
            val = odf.loc[nearest_idx, "speed"]
            if pd.notna(val):
                recurrence_value = float(val)
                recurrence_source = f"OMNI2 reanalysis @ {nearest_idx} (NOT live — placeholder, see ROADMAP.md)"
        except Exception as exc:
            recurrence_source = f"unavailable ({exc})"

    X_live = build_inference_features(recent, feature_columns, recurrence_value=recurrence_value)

    speed_now = float(live["speed"].dropna().iloc[-1])
    pred_ml = float(model.predict(X_live)[0])
    pred_persist = baselines.persistence_forecast(pd.Series([speed_now])).iloc[0]
    pred_recur = (
        baselines.recurrence27d_forecast(pd.Series([recurrence_value])).iloc[0]
        if recurrence_value is not None
        else None
    )

    now = live.index.max()
    target_time = now + pd.Timedelta(hours=HORIZON_HOURS)

    print("\n=== Example 72h-ahead solar wind speed forecast ===")
    print(f"Initialised from ACE at:  {now} UTC")
    print(f"Forecast valid for:       {target_time} UTC")
    print(f"Current speed (t=0):      {speed_now:.1f} km/s")
    print(f"27-day recurrence input:  "
          f"{f'{recurrence_value:.1f} km/s' if recurrence_value is not None else 'n/a'} "
          f"[source: {recurrence_source}]")
    print("-" * 55)
    print(f"Persistence baseline:     {pred_persist:.1f} km/s")
    if pred_recur is not None:
        print(f"27-day recurrence model:  {pred_recur:.1f} km/s")
    print(f"Trained model (this repo):{pred_ml:>7.1f} km/s   <-- example prediction")
    print("=" * 55)
    print(
        "\nCaveat: initialised from a live, unvalidated real-time feed and a "
        "short (~2h) observation window, using a model trained on the "
        "quality-controlled OMNI2 reanalysis — the same train/operational "
        "distribution gap every real-time ML forecast system has to manage. "
        "See ROADMAP.md for open items before this is submission-ready."
    )


if __name__ == "__main__":
    main()
