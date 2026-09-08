# Roadmap & open questions

## The challenge (from https://fdl.ai/benchlab-solar-wind-prediction-2026)

- **Task**: forecast solar wind bulk speed upstream of Earth, 72h ahead,
  operationally — a new forecast every 6 hours.
- **Evaluation**: live, unseen data against the real ACE satellite (not a
  static held-out test set), over Nov 2026–Jan 2027 then Feb–Apr 2027.
- **Deliverable**: a Docker container with the packaged prediction code,
  MIT-licensed, teams retain IP.
- **Timeline**: briefing materials Sep–Oct 2026, pre-deployment window
  Nov 1–10, 2026, official launch Nov 11, 2026.
- **Teams**: 1–4 people, capped at 100 teams; no space-physics background
  required. $20,000 prize pool.

The linked page is a marketing/overview page — the **briefing pack** (behind
pre-registration) almost certainly has the precise metric, exact data feed
names/cadence, submission format, and container spec. That pack, not this
page, should be the source of truth once you have it — see open questions
below.

## What's built so far

1. **`src/data/omni.py`** — loads NASA's OMNI2 hourly dataset (1963–present,
   merged/cross-calibrated across decades of spacecraft). This plays the role
   ERA5 plays for GraphCast/AIFS: a long, consistent **reanalysis** to train
   on. Cached 2015–2026 locally in `data/raw/`.
2. **`src/data/ace.py`** — pulls NOAA SWPC's live ACE real-time text feeds
   (SWEPAM plasma + magnetometer) and appends every fetch to a local rolling
   archive, since the public feed itself only exposes a ~2h window. This is
   the "live analysis" used to initialise forecasts operationally.
3. **`src/data/solar_orbiter.py`** — pulls real Solar Orbiter L2 data
   (SWA-PAS proton moments, MAG) from ESA's SOAR archive via `sunpy`.
   Validated against real data on 2026-02-14 (see below).
4. **`src/features.py`, `src/models/`** — lag/rolling/recurrence feature
   engineering; a `HistGradientBoostingRegressor` trained on OMNI2
   (2015–2024) and evaluated on held-out 2025:

   | Model | MAE (km/s) | RMSE (km/s) |
   |---|---|---|
   | Persistence | 122.3 | 152.8 |
   | 27-day solar-rotation recurrence | 123.9 | 156.8 |
   | **This repo's model** | **86.4** | **118.4** |

5. **`scripts/predict.py`** — end-to-end: fetches live ACE data right now,
   builds features, runs the trained model, prints a 72h-ahead forecast
   alongside both baselines. This is real, live output, not a mock.

## Key finding: Solar Orbiter cannot (yet) feed the live pipeline

I probed ESA's SOAR archive directly rather than assuming:

- **L2 SWA-PAS (proton speed/density/temperature) and MAG data**: available
  only up to ~6–7 months before "now" (on 2026-09-08, data was available
  through ~2026-03). This is real processing/calibration latency in the
  science archive, not a bug on our end.
- SOAR's catalog *does* list low-latency (`-ll`) products, but only for the
  **Energetic Particle Detector (EPD)** suite (`epd-step-ll`, `epd-sis-ll`,
  ...) — not for MAG or SWA-PAS. Querying these for recent dates returned
  zero results in this session; unclear if that's an ingestion gap, a
  wrong query attribute, or genuinely no public near-real-time access.
- **Consequence**: Solar Orbiter is used here as an **offline research
  input** (extending the training corpus with a second vantage point,
  prototyping a "magnetic-connectivity-gated" feature) — not as a live
  operational input, unless one of the open questions below resolves that.

## Open questions — need your input

1. **Briefing pack access.** Do you have (or can you get) the actual
   briefing materials? They should specify the exact evaluation metric
   (MAE? RMSE? skill score vs. a named baseline? a threshold/event-based
   score, given the "wildcard bonuses for event-based performance"?), the
   exact ACE data product/cadence used for grading, and the container I/O
   contract. Everything above is a reasonable default, not a guess at their
   actual rubric.
2. **Solar Orbiter's real role.** Given the latency finding above, how do
   you want to use it?
   - (a) Training-corpus augmentation only (safe, already partly wired up),
   - (b) Chase down genuine low-latency EPD/plasma access (would need ESA
     contact or a different delivery channel — worth a day of investigation
     before committing to it as a scored input), or
   - (c) Drop it and focus entirely on ACE/OMNI, revisit if time allows.
3. **Team & repo mechanics.** Team size/members (challenge caps at 4)? Do
   you want this developed straight in this git repo, or per the parent
   playground's `from_claude` push workflow? Any co-authors whose commits
   should be attributed separately?
4. **Compute/deployment target.** The container runs unattended for months
   (Nov–Apr) generating a forecast every 6h. Where will it actually run
   (your own server, a cloud VM, FDL-provided infra)? Any constraints on
   image size, runtime, or internet access from inside the container (it
   needs to reach `services.swpc.noaa.gov` at minimum)?
5. **Model ambition.** Current model is a solid, fast, interpretable
   baseline-beater. Given "no space physics background required" but a
   $20k pool across 100 teams, do you want to invest in a more
   sophisticated sequence model (temporal transformer / LSTM over the OMNI2
   time series, closer to an actual GraphCast-style architecture), richer
   features (e.g. coronal hole area from SDO/AIA imagery, a classic solar
   wind speed predictor), or keep iterating on the current gradient-boosted
   approach?
6. **The 27-day recurrence feature at inference time.** Right now, live
   inference falls back to an OMNI2 reanalysis value (not live) for "value
   27 days ago" because the freshly-deployed ACE archive hasn't accumulated
   27 days yet, and OMNI2 real-time-ish updates lag by ~1-2 weeks (see
   `omni.describe_latency`). Fine as an interim measure — but by the
   Nov 1 pre-deployment window, this repo's own rolling ACE archive should
   have real history. Worth confirming that's an acceptable bridge.
7. **Validation strategy.** Held-out 2025 is a decent first pass, but solar
   wind speed is strongly solar-cycle dependent — do you want walk-forward
   validation across multiple years/solar cycle phases before trusting the
   skill numbers, and/or backtesting specifically against known past
   high-speed-stream and CME events?

## Suggested next steps (once the above is answered)

- Wire up the Docker container + a scheduler stub for the 6-hourly cadence.
- Walk-forward validation across 2018–2025 instead of a single test year.
- Decide + implement the Solar Orbiter path from question 2.
- If briefing-pack metric differs from MAE/RMSE, retrain/tune against it directly.
