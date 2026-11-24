# Groundwater Level Forecasting for Bangalore

Forecasting groundwater levels at 29 telemetric monitoring stations around Bangalore, **7 and 30 days ahead**, with a **calibrated 90% prediction interval** on every forecast.

Stations are linked in a distance-based graph so a shared GCN + GRU model can use information from neighbouring wells. Intervals come from **split conformal prediction**, which we compare against raw quantile regression.

> In our literature survey (24 papers), we found no prior work that combines station-graph forecasting with calibrated conformal intervals for groundwater level. Our survey was not exhaustive.

## Key results

All numbers are on the held-out test period (2025-07-01 to 2026-06-04).

**Point forecasts (pooled RMSE, metres)**

| Model | 7 days | 30 days |
|---|---|---|
| Persistence ("level stays the same") | 1.258 | 2.652 |
| Shared LSTM (levels relative to today) | 1.232 | 2.509 |
| GCN + GRU | 1.216 | 2.486 |
| Same model, graph switched off | 1.209 | 2.418 |

**90% prediction intervals**

| Method | 7-day coverage | 7-day width | 30-day coverage | 30-day width |
|---|---|---|---|---|
| Raw quantile regression | 0.857 | 1.77 m | 0.842 | 4.07 m |
| Conformal, one width for all stations | 0.908 | 2.85 m | 0.928 | 7.42 m |
| **Conformal, per station** | **0.931** | **2.52 m** | **0.955** | **6.73 m** |

What these results show:

- The learned models beat persistence by about 3% at 7 days and 6–9% at 30 days. The gains are small because weekly changes in level are tiny (median about 0.19 m).
- **Rainfall drives the gain.** Without rainfall features, the model does worse than persistence.
- **The graph gave no reliable accuracy gain.** Differences between random seeds were as large as the effect of the graph.
- **Conformal calibration restores coverage that quantile regression loses**, but the intervals are wider.

These results come from one time split and mostly one random seed. See [Limitations](#limitations).

## Data

- **Groundwater levels:** [India-WRIS](https://indiawris.gov.in/) telemetric station data (CGWB) for Bangalore Urban and Rural, with 4 readings per day. We started with 33 stations and kept 29 after cleaning. Levels are negative numbers giving depth below ground, so a value closer to 0 means the water table is higher.
- **Rainfall:** IMD 0.25° gridded daily rainfall via [`imdlib`](https://pypi.org/project/imdlib/). Each station is matched to its nearest grid cell. Rainfall for 2026 comes from provisional real-time IMD files.

Raw and cleaned data are not committed (see [.gitignore](.gitignore)). To rebuild them, put the per-station `.xlsx` files downloaded from India-WRIS in `data/raw/`.

## Method

1. **Cleaning** ([src/data.py](src/data.py), [src/config.py](src/config.py)):
   - average the readings to daily values
   - apply manual cutoffs and masks
   - remove readings above ground level
   - remove flat-line stretches (stuck sensors) and spikes
   - interpolate gaps of 3 days or less
   - add a `missing` flag for days without a real reading
2. **Features:** a 30-day input window with 7 features per day: level, rain, 7-day and 30-day rain sums, month as sin/cos, and the missing flag. Scalers are fitted per station on training days only.
3. **Time split** by target date:
   - Train: up to 2024-06-30, with the last 2 months used for validation
   - Calibration: 2024-07-01 to 2025-06-30
   - Test: 2025-07-01 onwards
4. **Models:** persistence baseline, shared LSTM, and GCN + GRU (k = 6 nearest-neighbour graph). We also ran ablations: no graph, no rain, a low-data setting and a second seed.
5. **Uncertainty:** split conformal prediction fitted on the calibration year, both global and per station. We compare it against quantile regression trained with pinball loss.

## Repository layout

```
src/            Python pipeline: cleaning, features, models, conformal, plots
results/        Metrics CSVs, prediction .npz files, diagnostic plots
app/            React + Leaflet map demo (backtest viewer)
data/           raw/ and clean/ (not committed)
```

| File | Purpose |
|---|---|
| `src/config.py` | Dropped stations, cutoffs, mask ranges, time split |
| `src/data.py`, `src/build_clean.py` | Load the Excel files and clean the series |
| `src/get_imd.py`, `src/fill_imd_realtime.py`, `src/build_rain.py` | Download rainfall and match it to stations |
| `src/build_panel.py` | Build features and scale them into `data/clean/panel.csv` |
| `src/windows.py`, `src/evaluate.py` | Sample windows, split masks, metrics |
| `src/baseline_persistence.py` | Persistence baseline |
| `src/lstm_baseline.py` | Shared LSTM |
| `src/graph.py`, `src/graph_data.py`, `src/gcn_gru.py` | Graph construction and the GCN + GRU model |
| `src/gcn_quantile.py` | Quantile regression comparison |
| `src/conformal.py`, `src/conformal_stations.py` | Conformal intervals (pooled and per station) |
| `src/compare_runs.py` | Compare ablations, seeds and low-data runs |
| `src/plot_*.py`, `src/check_*.py`, `src/test_*.py` | Plots, diagnostics and sanity checks |
| `src/export_demo.py` | Export results to `app/public/demo_data.json` |

## Setup

You need Python 3.10+ and, optionally, an NVIDIA GPU. The models are small: we trained them on an RTX 3050 6 GB, and a CPU also works.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

This installs PyTorch too (`torch 2.11.0+cu128`, the CUDA 12.8 build we used). It runs on the CPU when no NVIDIA GPU is found. On macOS, where CUDA builds don't exist, delete the two PyTorch lines from `requirements.txt` and run `pip install torch` instead. To check that the GPU is visible:

```bash
python src/check_gpu.py
```

## Running the pipeline

Run every command from the project root.

```bash
# 1. Clean the station data      -> data/clean/daily_clean.csv, stations.csv
python src/build_clean.py

# 2. Get rainfall                -> data/raw/imd/, data/clean/rain_daily.csv
python src/get_imd.py
python src/fill_imd_realtime.py
python src/build_rain.py

# 3. Build features              -> data/clean/panel.csv, scalers.json
python src/build_panel.py

# 4. Baselines
python src/baseline_persistence.py
python src/lstm_baseline.py

# 5. Graph model and ablations   -> results/<name>_metrics.csv, <name>_pred.npz
python src/gcn_gru.py graph
python src/gcn_gru.py nograph
python src/gcn_gru.py graph norain
python src/gcn_gru.py graph lowdata
python src/gcn_gru.py graph seed=1

# 6. Uncertainty
python src/gcn_quantile.py
python src/conformal.py gcn nograph
python src/conformal_stations.py gcn

# 7. Compare runs and plot
python src/compare_runs.py gcn gcn_norain gcn_s1 nograph nograph_s1 gcn_lowdata nograph_lowdata
python src/plot_intervals.py gcn
```

Each run sets its random seed and saves its best checkpoint (`results/*.pt`). Checkpoints are not committed.

## Demo app

The `app/` folder has an interactive map of the stations. You can colour the stations by skill versus persistence, by interval width or by test coverage. A dashed ring marks a station with coverage under 85%. Click a station to see a chart with:

- the observed readings (black dots)
- the forecast (blue line)
- the 90% interval (shaded band)

```bash
python src/export_demo.py      # regenerate app/public/demo_data.json (optional; a copy is committed)
cd app
npm install
npm run dev
```

**This is a backtest, not a live forecast.** Every forecast shown is for a date between July 2025 and June 2026 that the model never trained on. Each one uses only the 30 days before the forecast date.

## Limitations

- We used one time split and mostly one seed, and the differences between the neural models are small.
- The spike filter misses bursts of consecutive outliers.
- A few stations are under-covered, and very volatile stations get very wide intervals.
- We looked at test results while diagnosing problems, so the test numbers are not a pristine one-shot evaluation. We did not change the cleaning rules afterwards.
- CGWBNG0018 has no test data, and CGWBNG0026 has only 99 real training days.
- Rainfall for 2026 comes from provisional IMD files.
