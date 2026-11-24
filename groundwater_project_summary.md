# Groundwater Level Forecasting for Bangalore: Project Summary

Written so teammates can explain the project to our faculty and pick up context. Every number below comes from our own runs. If a number is not here, we did not measure it.

---

## 1. One-paragraph version

We forecast groundwater level at 29 monitoring stations around Bangalore, 7 and 30 days ahead, and attach a 90% prediction interval to each forecast. The data is telemetric station data from India-WRIS (CGWB) plus IMD rainfall. We compared a "no change" baseline (persistence), a shared LSTM, and a GCN + GRU that lets neighbouring stations share information. The learned models beat persistence modestly, and rainfall is what produces the gain. The graph did not give a reliable accuracy gain. The main contribution is the uncertainty part: **conformal prediction restores 90% coverage that raw quantile regression loses**, at the cost of wider intervals. In our survey we found none that combine station-graph forecasting with calibrated conformal intervals for this data (we do not claim nobody has done it).

---

## 2. The problem and the plan

- **Goal:** forecast water level at each station 7 and 30 days ahead, with a calibrated 90% interval.
- **Novelty claim (careful wording):** graph forecasting plus conformal intervals. Say "we found none in our survey", never "nobody has done this".
- **Phases:** A setup, B data, C weather, D baselines, E graph model (GCN + GRU), F conformal + quantile comparison, G evaluation and ablations, H demo and report.
- **Never cut:** the baselines and the conformal step.
- **Working rules:** one phase at a time, inspect real data before assuming, small readable functions, test on fake data first, never invent numbers, set seeds, save the best checkpoint.

---

## 3. Data

- 33 station files (.xlsx, 4 readings per day) from India-WRIS. After inspection we **dropped 4 stations** and kept **29**:
  - CGWBNG0020: unusable (swings between -73 and +25 m)
  - CGWBNG0033: smooth drift above ground level
  - CGWBNG0009: only about 13 months trustworthy
  - CGWBNG0030: only about 15 months trustworthy
- Manual fixes: CGWBNG0018 kept only up to 2025-06-30 (chaotic after), so it has **no test samples**. CGWBNG0032 has 2024-01-01 to 2024-07-10 masked (sensor floor plateau).
- Levels are negative numbers meaning depth below ground. A rise toward zero means the water table is rising. See the sign check in section 8.
- Daily series: mean of the day's readings.
- Data ends **2026-06-04**.

### Cleaning pipeline (in order)
1. Manual cutoff and mask ranges.
2. Range rule: level above 0 is invalid.
3. Flat rule: 10-day window with range at most 0.01 m (sensor resolution) means a stuck sensor.
4. Spike rule: centred 7-day median with a local noise scale (k = 5, minimum limit 0.3 m).
5. Interpolate only gaps of 3 days or less. Every day without a real trusted reading gets `missing = 1`.
6. The loss and all metrics ignore targets that are not real readings.

Honest note: the spike filter is centred (it looks at days on both sides), which bends the "past values only" idea. It is an offline data-quality step. It also misses bursts of several consecutive outliers (visible at CGWBNG0007).

---

## 4. Weather data

- IMD 0.25 degree gridded daily rainfall. Yearly archives for 2021 to 2025, plus provisional real-time daily files for 2026 (up to 2026-06-04).
- Each station is matched to its nearest grid cell (29 stations map to 9 distinct cells).
- Yearly mean rainfall (mm): 2021 1228, 2022 1636, 2023 703, 2024 731, 2025 897. The training years were on the drier side.

---

## 5. Setup of the experiments

- **Input window:** the last 30 days. **Targets:** level 7 and 30 days ahead.
- **Features (7):** level, rain, 7-day rain sum, 30-day rain sum, month (sin, cos), missing flag.
- **Scaling:** per station on real training days only. Rain is log1p then standardised on training data.
- **Time split by target date:**
  - Train: up to 2024-06-30 (last 2 months used as validation for early stopping)
  - Calibration: 2024-07-01 to 2025-06-30 (used only to set interval widths)
  - Test: 2025-07-01 onwards (7727 targets at 7 days, 7499 at 30 days)
- **Metrics:** RMSE and MAE in metres. **Skill = 1 - RMSE(model) / RMSE(persistence)** per station. Above 0 means it beats persistence.
- **Persistence baseline:** "the level in 7 (or 30) days equals the level today".

---

## 6. Results

### 6.1 The baseline ladder (test set, pooled RMSE in metres)

| Model | 7 days | 30 days |
|---|---|---|
| Persistence | 1.258 | 2.652 |
| Shared LSTM, absolute levels (first attempt) | 1.435 | 2.909 |
| Shared LSTM, levels relative to today | 1.232 | 2.509 |
| GCN + GRU (seed 0) | 1.216 | 2.486 |
| Same model without the graph (seed 0) | 1.209 | 2.418 |
| GCN + GRU (seed 1) | 1.221 | 2.489 |
| Without the graph (seed 1) | 1.216 | 2.469 |

Learned models beat persistence by about 3% at 7 days and 6 to 9% at 30 days. The gains are modest because weekly changes are tiny (median 7-day change about 0.19 m, close to sensor resolution).

### 6.2 What we diagnosed and fixed
The first LSTM was **worse** than persistence. Cause: it took absolute levels as input, and many stations' test levels fell outside the range seen in training (median 20% of test days outside the training range, 100% for stations 0010, 0004, 0039). Feeding levels **relative to today** fixed it. Typical weekly change sizes were similar across periods (0.188 m train, 0.188 m calibration, 0.196 m test), so a "calmer training period" explanation was wrong; the problem was the level range.

### 6.3 Ablations
- **Rainfall matters.** Without the rain features the model is worse than persistence (test RMSE 1.270 at 7 days and 2.692 at 30 days). Median skill drops from 0.050 to -0.060 at 7 days.
- **The graph gave no reliable accuracy gain.** With one seed it looked helpful at 7 days; the second seed removed that. The no-graph model is slightly better on pooled RMSE in both seeds. The only consistent hint is that the graph model beats persistence at more stations at 7 days (24 and 27 of 28 vs 16 and 23).
- **Low-data test (5 stations limited to about 6 months of training):** mixed. At 7 days the graph model held up better; at 30 days the no-graph model did. Only 5 stations, so weak evidence. Fixed stations: 010212, 0007, 0019, 0029, 0039.

### 6.4 Prediction intervals (the main contribution)

Method: **split conformal prediction.** On the calibration year, take the error size that 90% of cases stayed below (per station, per horizon), and use forecast plus or minus that margin. Coverage is then measured on the test set.

| Method (test set) | 7-day coverage | 7-day mean width | 30-day coverage | 30-day mean width |
|---|---|---|---|---|
| Conformal, one width for all stations | 0.908 | 2.85 m | 0.928 | 7.42 m |
| **Conformal, per station** | **0.931** | **2.52 m** | **0.955** | **6.73 m** |
| Persistence + per-station conformal | 0.930 | 2.65 m | 0.953 | 6.63 m |
| Raw quantile regression (5th to 95th percentile) | 0.857 | 1.77 m | 0.842 | 4.07 m |

Target coverage was 0.90.

- A single shared width hits 90% on average but is uneven: the worst station was covered only 38% of the time.
- Raw quantile regression **under-covers** (about 85% instead of 90%), even on the calibration year (84.6% and 75.4%). Its intervals look narrower, but they cover less, so the widths are not comparable.
- Conformal per station over-covers a little because the calibration year was harder than the test year.
- Model intervals are only slightly narrower than persistence's (about 5% at 7 days; at 30 days the GCN is slightly wider). That matches the small accuracy gain.
- A few stations under-cover (below 85%): at 7 days 0007 (78%), 0006 (81%), 0019 (84%); at 30 days 0019 (74.5%) and 0031 (82.5%). At 0007 the test errors were 2.6 times the calibration errors.
- Volatile stations (0019, 0024, 0032) get very wide intervals (tens of metres at 30 days). They are honest but not very useful.

Headline sentence: **a model trained on one period gives overconfident quantile intervals later, and conformal calibration on a later period restores the coverage, at the cost of wider intervals.** One split and one seed.

---

## 7. The demo app

- A map app (React + Leaflet) in `app/`, fed by `app/public/demo_data.json` created by `src/export_demo.py` from our result files.
- Each dot is a station. Colour by skill vs persistence, interval width, or test coverage. Dashed ring = coverage under 85%.
- Click a station for the chart: **black dots = real sensor readings, blue line = the model's forecast, light blue band = the 90% interval.** Hover shows date, observed, forecast and interval.
- **It is a backtest.** Every forecast shown is for a date between July 2025 and 4 June 2026 that the model never trained on, made using only the 30 days before the forecast. It is not a live forecast. A live forecast would need fresh station data and fresh IMD rainfall (the data we have ends on 4 June 2026).

---

## 8. Sign check

We checked that the correlation between rainfall and the next 30-day change in level is positive. It was positive at **all 29 stations** (median 0.56, lowest 0.18). That supports the reading that a rise toward zero means the water table is rising. Caveats: correlation, not causation; rain and groundwater both follow the monsoon; windows overlap. It confirms direction, not the exact "metres below ground" label, which comes from station metadata.

---

## 9. Limitations (say these before Sir finds them)

- One time split and mostly one seed; differences between the neural models are small.
- Validation window is only 2 months and was a weak guide for early stopping (the no-rain model had lower validation loss but worse test results).
- The spike filter misses bursts of consecutive outliers (visible at 0007). We did not re-clean after seeing test results.
- 2026 rainfall comes from provisional IMD files.
- CGWBNG0018 has no test data. CGWBNG0026 has only 99 real training days.
- Low-data test uses only 5 stations, and the scalers still use full training statistics.
- Some stations are under-covered; very volatile stations get wide intervals.
- We looked at test results while diagnosing, so test numbers are not a pristine one-shot evaluation. Cleaning rules were not changed afterwards.

---

## 10. Likely questions from Sir

- **Why didn't the graph help?** Weekly changes are tiny and the shared model already learns across stations. Seed-to-seed differences were as large as the graph effect. We report it as a tested idea with no reliable gain.
- **Why is coverage above 90%?** The calibration year was harder than the test year, so the margins were generous.
- **Why a 90% interval and not 95%?** It was our design choice; the method works for any level.
- **Is this a live forecast?** No. It is a backtest on held-out dates. The same pipeline can forecast live once the latest data is downloaded.
- **Why did the first LSTM fail?** It used absolute levels and test levels fell outside the training range. Relative levels fixed it.
- **What is persistence and why compare to it?** "Tomorrow equals today". Groundwater changes slowly, so it is hard to beat, and any model must beat it to be useful.

---

## 11. Glossary

- **RMSE / MAE:** average error size in metres (RMSE punishes big errors more).
- **Skill:** 1 minus (model error / persistence error). 0.10 means 10% lower error than persistence.
- **GCN:** a graph step where each station mixes information with its nearest neighbours.
- **GRU / LSTM:** neural networks that read a sequence (our 30 days).
- **Conformal prediction:** a way to build intervals from past errors on held-out data, aiming for a chosen coverage.
- **Coverage:** the share of real readings that fell inside the interval.
- **Backtest:** testing on past dates the model never trained on.
- **Pinball loss / quantile regression:** training a model to predict the 5th and 95th percentile directly.

---

## 12. Code map and how to run

Run from the project root, inside the `.venv`.

| File | Purpose |
|---|---|
| `src/config.py` | Dropped stations, cutoffs, mask ranges, time split |
| `src/data.py`, `src/build_clean.py` | Load Excel files, clean the series |
| `src/get_imd.py`, `src/fill_imd_realtime.py`, `src/build_rain.py` | Download and match rainfall |
| `src/build_panel.py` | Features and scaling into `data/clean/panel.csv` |
| `src/windows.py`, `src/evaluate.py` | Samples, split masks, metrics |
| `src/baseline_persistence.py` | Persistence baseline |
| `src/lstm_baseline.py` | Shared LSTM |
| `src/graph.py`, `src/graph_data.py`, `src/gcn_gru.py` | Graph, data layout, GCN + GRU |
| `src/gcn_quantile.py` | Quantile regression comparison |
| `src/conformal.py`, `src/conformal_stations.py` | Conformal intervals and per-station table |
| `src/compare_runs.py` | Compare runs (ablations, seeds, low-data) |
| `src/plot_intervals.py` | Interval plots |
| `src/check_shift.py`, `src/check_sign.py` | Diagnostics |
| `src/export_demo.py`, `app/` | Map demo |

Typical commands:

```bash
python src/lstm_baseline.py
python src/gcn_gru.py graph            # or: nograph, norain, lowdata, seed=1
python src/gcn_quantile.py
python src/conformal.py gcn nograph
python src/conformal_stations.py gcn
python src/compare_runs.py gcn gcn_norain gcn_s1 nograph nograph_s1 gcn_lowdata nograph_lowdata
python src/plot_intervals.py gcn
```

Results are written to `results/`. Data and model checkpoints are excluded by `.gitignore`.
