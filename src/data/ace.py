"""
Live ACE real-time feed — the "initial conditions" source.

Continuing the NWP analogy: OMNI2 (src.data.omni) is our ERA5-like reanalysis
used to *train* the model. Operationally, an NWP model isn't run from ERA5 —
it's initialised from the live operational analysis. Here that role is played
by NOAA SWPC's real-time ACE (SWEPAM + MAG) text feeds.

Caveat (important, see ROADMAP.md): these feeds only expose a rolling ~2 hour
window at 1-minute cadence. There is no public endpoint for "the last 27 days
of real-time ACE" comparable to OMNI2's history, so a feature like "value 27
days ago" cannot be built from a single fetch. `append_to_history` persists
every fetch into a local rolling archive (data/raw/ace_live_history.csv) so
that, once this pipeline has been running operationally for a while, that
history becomes available — the same "spin-up" limitation that any freshly
deployed real-time archive has.

Source: https://services.swpc.noaa.gov/text/ace-swepam.txt
        https://services.swpc.noaa.gov/text/ace-magnetometer.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

SWEPAM_URL = "https://services.swpc.noaa.gov/text/ace-swepam.txt"
MAG_URL = "https://services.swpc.noaa.gov/text/ace-magnetometer.txt"

HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ace_live_history.csv"


def _parse_swepam(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if not line or line[0] not in "12":  # data lines start with a 4-digit year
            continue
        parts = line.split()
        yr, mo, da, hhmm = parts[0:4]
        status, density, speed, temperature = parts[6:10]
        ts = pd.Timestamp(f"{yr}-{mo}-{da} {hhmm[:2]}:{hhmm[2:]}", tz="UTC")
        rows.append(
            dict(
                timestamp=ts,
                status=int(status),
                density=float(density),
                speed=float(speed),
                temperature=float(temperature),
            )
        )
    df = pd.DataFrame(rows).set_index("timestamp")
    df = df.replace({"density": -9999.9, "speed": -9999.9, "temperature": -1.00e05}, pd.NA)
    return df


def _parse_magnetometer(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if not line or line[0] not in "12":
            continue
        parts = line.split()
        yr, mo, da, hhmm = parts[0:4]
        status = parts[6]
        bx, by, bz, bt, lat, lon = parts[7:13]
        ts = pd.Timestamp(f"{yr}-{mo}-{da} {hhmm[:2]}:{hhmm[2:]}", tz="UTC")
        rows.append(
            dict(
                timestamp=ts,
                status=int(status),
                bx_gsm=float(bx),
                by_gsm=float(by),
                bz_gsm=float(bz),
                bt=float(bt),
            )
        )
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def fetch_current(timeout: int = 20) -> pd.DataFrame:
    """Fetch the latest rolling window (~2h @ 1-min cadence) of ACE plasma + IMF data."""
    plasma = _parse_swepam(requests.get(SWEPAM_URL, timeout=timeout).text)
    mag = _parse_magnetometer(requests.get(MAG_URL, timeout=timeout).text)
    df = plasma.join(mag, how="outer", lsuffix="_plasma", rsuffix="_mag")
    df = df[(df.get("status_plasma", 0) == 0) | (df.get("status_mag", 0) == 0)]
    return df.sort_index()


def append_to_history(df: pd.DataFrame) -> pd.DataFrame:
    """Merge a fresh fetch into the local rolling history archive and return the full archive."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        existing = pd.read_csv(HISTORY_PATH, index_col=0, parse_dates=True)
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df.sort_index()
    combined.to_csv(HISTORY_PATH)
    return combined


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORY_PATH, index_col=0, parse_dates=True)


if __name__ == "__main__":
    live = fetch_current()
    print(live.tail())
    hist = append_to_history(live)
    span = (hist.index.max() - hist.index.min()) if len(hist) else None
    print(f"Local ACE history archive now spans: {span} ({len(hist)} rows)")
