"""
Solar Orbiter (SolO) in-situ data via the ESA Solar Orbiter Archive (SOAR).

Role in the pipeline — and an important finding from probing the archive
--------------------------------------------------------------------------
The pitch for using Solar Orbiter is that, when it is magnetically well
connected to Earth (roughly in quadrature or closer to the Sun-Earth line
than L1), its in-situ plasma/field measurements sample solar wind structures
*before* they corotate or propagate to L1 — extra lead time for a 72h
forecast, similar to how assimilating an extra satellite sounding improves
an NWP analysis.

However, probing SOAR (see ROADMAP.md for the session log) found:
  * L2 SWA-PAS (proton bulk parameters) and MAG data are available only up to
    ~6-7 months before "now" in this environment (e.g. on 2026-09-08, L2 data
    was only available up to ~2026-03). This is real archive/processing
    latency, not a bug — L2 SolO products go through a calibration pipeline
    before public release.
  * SOAR's TAP catalog lists low-latency ("-ll") descriptors, but only for
    the Energetic Particle Detector (EPD) suite (epd-epthet1-ll, epd-sis-ll,
    epd-step-ll, ...) — no low-latency MAG or SWA-PAS. Querying these via
    `sunpy.net.Fido` returned zero results for recent dates in this session;
    it's unclear whether that's an ingestion gap, a query/attrs mismatch, or
    genuinely no near-real-time archive access. This needs follow-up (see
    open questions) — possibly via ESA's dedicated space-weather low-latency
    delivery rather than the public science archive.

Conclusion: for this challenge (operational forecasts every 6h against live
ACE), Solar Orbiter L2 data CANNOT currently be used as a real-time input.
It is used here in an offline/research capacity only:
  1. As additional *training* signal — extending the reanalysis-style
     training corpus with an independent vantage point for periods when SolO
     was usefully positioned.
  2. As a testbed for a "connectivity-gated" feature (only meaningful when
     SolO-Earth magnetic connectivity is favourable) that could migrate to
     real-time use if/when a genuine low-latency plasma feed is confirmed.

Source: https://soar.esac.esa.int  (queried here via `sunpy.net.Fido`)
"""
from __future__ import annotations

from pathlib import Path

import cdflib
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "solar_orbiter"


def search_available(start: str, end: str, product: str = "SWA-PAS-GRND-MOM"):
    """List what SOAR actually has for a date range. Returns a sunpy UnifiedResponse."""
    from sunpy.net import Fido, attrs as a

    return Fido.search(a.Time(start, end), a.soar.Product(product))


def fetch_swa_pas(start: str, end: str) -> pd.DataFrame:
    """Download (if needed) and parse SWA-PAS ground-moment proton bulk parameters.

    NOTE: only useful for `end` at least ~6 months before today — see module
    docstring. This is a training/research-data loader, not an operational one.
    """
    from sunpy.net import Fido, attrs as a

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    query = Fido.search(a.Time(start, end), a.soar.Product("SWA-PAS-GRND-MOM"))
    files = Fido.fetch(query, path=str(CACHE_DIR / "{file}"))
    frames = [_parse_swa_cdf(f) for f in sorted(files)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def fetch_mag(start: str, end: str) -> pd.DataFrame:
    """Download (if needed) and parse MAG RTN-normal magnetic field data. Same latency caveat."""
    from sunpy.net import Fido, attrs as a

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    query = Fido.search(a.Time(start, end), a.soar.Product("MAG-RTN-NORMAL"))
    files = Fido.fetch(query, path=str(CACHE_DIR / "{file}"))
    frames = [_parse_mag_cdf(f) for f in sorted(files)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def _parse_swa_cdf(path: str) -> pd.DataFrame:
    cdf = cdflib.CDF(path)
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
    velocity_rtn = cdf.varget("V_RTN")  # (N, 3): Vr, Vt, Vn, km/s
    density = cdf.varget("N")  # cm^-3
    temperature = cdf.varget("T")  # eV
    speed = np.linalg.norm(velocity_rtn, axis=1)
    df = pd.DataFrame(
        {
            "speed": speed,
            "density": density,
            "temperature": temperature,
        },
        index=pd.DatetimeIndex(epoch, name="timestamp"),
    )
    return df[df["speed"] < 1e5]  # drop CDF fill values


def _parse_mag_cdf(path: str) -> pd.DataFrame:
    cdf = cdflib.CDF(path)
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget("EPOCH"))
    b_rtn = cdf.varget("B_RTN")  # (N, 3)
    bt = np.linalg.norm(b_rtn, axis=1)
    df = pd.DataFrame(
        {"br": b_rtn[:, 0], "bt_comp": b_rtn[:, 1], "bn": b_rtn[:, 2], "bt": bt},
        index=pd.DatetimeIndex(epoch, name="timestamp"),
    )
    return df[df["bt"] < 1e5]


if __name__ == "__main__":
    # Demonstration using a period SOAR actually has (see module docstring).
    result = search_available("2026-02-14", "2026-02-21")
    print(result)
