"""
OMNI2 hourly data — the "reanalysis" training corpus.

Why OMNI2 plays the role of ERA5 here
--------------------------------------
ECMWF/Google-style ML weather models (GraphCast, AIFS, ...) train on ERA5: a
long, multi-instrument, quality-controlled *reanalysis*, then run operationally
initialised from live analyses. There isn't a solar-wind "ERA5", but OMNI2 is
the closest equivalent used throughout the space-weather literature:

  * It merges multiple spacecraft (IMP-8, Wind, ACE, DSCOVR over the decades)
    into one continuous, cross-calibrated, quality-controlled hourly record
    propagated to Earth's bow shock nose.
  * It goes back to 1963, giving many solar cycles to learn from.
  * Like ERA5, it lags real time substantially (definitive OMNI2 typically
    trails by months — see `describe_latency`), so it is a *training* corpus,
    not something you can query for "now".

Source: NASA SPDF, https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
Format: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# word index (1-based, per omni2.text) -> (column name, fill value)
_COLUMNS = {
    1: ("year", None),
    2: ("doy", None),
    3: ("hour", None),
    9: ("b_avg", 999.9),
    17: ("bz_gsm", 999.9),
    16: ("by_gsm", 999.9),
    23: ("temperature", 9999999.0),
    24: ("density", 999.9),
    25: ("speed", 9999.0),
    29: ("flow_pressure", 99.99),
    39: ("kp", 99),
    40: ("sunspot_number", 999),
    41: ("dst", 99999),
    50: ("ap", 999),
    51: ("f107", 999.9),
}


def _cache_path(year: int) -> Path:
    return CACHE_DIR / f"omni2_{year}.dat"


def _download_year(year: int, force: bool = False) -> Path:
    path = _cache_path(year)
    if path.exists() and not force:
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/omni2_{year}.dat"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _parse_file(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            row = {name: parts[idx - 1] for idx, (name, _fill) in _COLUMNS.items()}
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    # replace documented fill values with NaN
    for idx, (name, fill) in _COLUMNS.items():
        if fill is not None:
            df.loc[df[name] >= fill * 0.99999, name] = np.nan

    timestamp = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(
        df["doy"] - 1, unit="D"
    ) + pd.to_timedelta(df["hour"], unit="h")
    df.insert(0, "timestamp", timestamp)
    df = df.drop(columns=["year", "doy", "hour"]).set_index("timestamp")
    return df


def load_years(years: list[int], force_download: bool = False) -> pd.DataFrame:
    """Fetch (with local caching) and parse OMNI2 hourly data for the given years."""
    frames = [_parse_file(_download_year(y, force=force_download)) for y in years]
    df = pd.concat(frames).sort_index()
    return df


def latest_definitive_timestamp(df: pd.DataFrame) -> pd.Timestamp:
    """Last timestamp with a non-NaN speed value — i.e. where the reanalysis record actually ends."""
    return df["speed"].dropna().index.max()


def describe_latency(df: pd.DataFrame) -> str:
    last = latest_definitive_timestamp(df)
    lag_days = (pd.Timestamp.utcnow().tz_localize(None) - last).days
    return (
        f"Latest definitive OMNI2 record: {last} UTC "
        f"(~{lag_days} days behind real time). "
        "This is the reanalysis-latency gap: fine for training, unusable for "
        "operational initial conditions — use src.data.ace for that."
    )


if __name__ == "__main__":
    data = load_years([2025, 2026])
    print(data.tail())
    print(describe_latency(data))
