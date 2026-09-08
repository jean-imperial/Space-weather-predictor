# Space-weather-predictor

A start on **[FDL.ai BenchLab: Solar Wind Prediction 2026](https://fdl.ai/benchlab-solar-wind-prediction-2026)** —
forecast solar wind bulk speed at L1, 72 hours ahead, evaluated live against
real-time ACE data (Nov 2026 – Apr 2027).

See [ROADMAP.md](ROADMAP.md) for the full challenge summary, the design
approach, what's been validated so far, and — importantly — **the open
questions this needs your input on** before it's submission-ready.

## Approach, in one sentence

Treat this like a data-driven NWP model (GraphCast/AIFS-style): train on a
long, quality-controlled **reanalysis** (OMNI2 — the space-weather analogue
of ERA5), then run the trained model operationally, initialised from the
**live real-time ACE feed** (the analogue of an operational analysis), and
always report skill against the standard baselines (persistence, 27-day
solar-rotation recurrence).

```
OMNI2 (2015-2025, hourly)  ──train──▶  HistGradientBoostingRegressor
        "reanalysis"                        │
                                             ▼
Live ACE feed (NOAA SWPC)  ──initialise──▶  72h-ahead speed forecast
   "real-time analysis"
```

Solar Orbiter is integrated as an **offline/research** data source (see
[src/data/solar_orbiter.py](src/data/solar_orbiter.py)) — its L2 science
data currently has ~6+ months of archive latency, so it can't feed the live
operational pipeline yet. Details and options in ROADMAP.md.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train on the OMNI2 reanalysis, see skill vs. baselines
python -m scripts.train

# 2. Fetch live ACE data and produce one example forecast
python -m scripts.predict
```

## Layout

```
src/data/omni.py           OMNI2 hourly reanalysis loader (training corpus)
src/data/ace.py            Live ACE real-time feed + rolling local archive
src/data/solar_orbiter.py  Solar Orbiter (SOAR) loader — offline/research use
src/features.py            Lag/rolling/recurrence feature engineering
src/models/baselines.py    Persistence + 27-day recurrence baselines
src/models/gbr.py          Trained forecaster (HistGradientBoostingRegressor)
scripts/train.py           Train + evaluate against baselines
scripts/predict.py         Live example prediction
tests/                     Feature-engineering sanity tests
```

## Current example output

```
=== Example 72h-ahead solar wind speed forecast ===
Initialised from ACE at:  2026-09-08 15:37:00+00:00 UTC
Forecast valid for:       2026-09-11 15:37:00+00:00 UTC
Current speed (t=0):      487.8 km/s
27-day recurrence input:  440.0 km/s [source: OMNI2 reanalysis (placeholder, not live)]
-------------------------------------------------------
Persistence baseline:     487.8 km/s
27-day recurrence model:  440.0 km/s
Trained model (this repo):  447.7 km/s   <-- example prediction
```

Held-out 2025 skill (MAE, km/s): persistence 122.3, 27-day recurrence 123.9,
**this model 86.4**.
