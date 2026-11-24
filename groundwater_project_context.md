# Project Context: Groundwater Level Forecasting (AIML course project)

Put this file in the project root (rename it to `CLAUDE.md` if you want Claude Code to load it automatically).
It explains what the project is, what has been decided, and how to work with me. It deliberately has no full implementation. Please build step by step with me.

## 1. How to work with me

- I am a second-year CS student. I write casually ("bro"), so explain simply, then add technical terms with their meaning.
- Work **one phase at a time** (see section 9). Do not generate the whole project in one go. Before each phase, say what you will do and why; after, say what to check.
- **Inspect the real data before assuming anything** (column names, units, sign of values, date ranges). Print shapes, head(), and missing-value counts.
- Keep functions small and readable so I can explain them to my teacher. Short comments on the "why".
- Test each part on small fake data first (for example 10 stations of sine waves plus noise), then on real data.
- **Never invent results, numbers or citations.** If something is unknown, say so and tell me how to check it. Until we run experiments, wording must be "we will evaluate", not "we achieved".
- Set random seeds, save the best checkpoint, and keep experiments reproducible.
- Hardware: laptop with an NVIDIA RTX 3050 6GB. The model is small, so the laptop is enough. First check `torch.cuda.is_available()`. Check my OS before suggesting OS-specific things.
- Ask me when something is ambiguous instead of guessing.

## 2. What the project is (one paragraph)

Forecast groundwater level at monitoring stations in Bangalore (urban and rural) for the next **7 and 30 days**, and attach a **90% prediction interval** to every forecast. Stations are connected in a graph by distance so each station can use information from its neighbours. One shared model is trained for all stations.

## 3. Novelty and research gap (be honest about it)

- Most groundwater ML models forecast one well at a time and output a single number.
- Graph neural networks for groundwater already exist (for example Bai and Tahmasebi 2023, plus several 2024 and 2025 papers), but they give **point forecasts**.
- Uncertainty methods for groundwater (LUBE, bootstrap, repeated runs) treat wells independently. Conformal prediction has been applied to streams, river discharge and nitrate, but I found none for groundwater level.
- **Our novelty is the combination:** graph forecasting plus calibrated conformal intervals, on dense Indian telemetric data. A graph model alone is NOT the novelty.
- My literature search (24 papers) is not exhaustive. Do not claim "nobody has done this"; say "we found none in our survey".

## 4. Data

**Groundwater levels:** India-WRIS (India Water Resources Information System, Ministry of Jal Shakti, published by NWIC). Portal filters I used: dataset Ground Water Level, state Karnataka, district Bangalore (urban and rural), mode of acquisition Telemetry, agencies CGWB (central) and the state agency. Portal is slow and sometimes lags; I download per station or district and keep the files locally.

**Station metadata columns (data dictionary):** index, Station Name, Station Code, Station Type, Station Status, Data Acquisition Type, Agency Name, Agency Id, Well Type, Well Aquifer Type, Latitude, Longitude, Date of Establishment, Data Available From, Availability of Discharge Data, Discharge Volume, Depth of Well, Data Date, Time of Data taken, Data Type Code, Data Type Description, Data Value.

**Per-station time series file (what I actually downloaded):** a few header lines (including a "Data Source" line), then a table with columns `Data Type Code` (GGZ), `Data Type Description` (GPRS-Water Level), `Data Time` (ISO format like 2023-07-15T18:00:00), `Data Value`, `Unit` (m).
- **4 readings per day** at 00:00, 06:00, 12:00, 18:00.
- Values are **negative** (for example -4.07 m). I believe this means depth below ground (closer to 0 = water nearer the surface), but this is NOT confirmed. Check the file header and tell me before we rely on it.
- Example station: Channenahalli_1, Station Code CGWBNG0016, CGWB, Telemetric, Bangalore rural, Hoskote, lat 12.6667, lon 77.25, bore well, data from 2023-07-15 to 2026-06-04.
- **History length differs by station** (this one has about 3 years, others may have more). Do not assume 2021 to 2026 for every station.
- Central (CGWB) and state data may contain duplicate stations. Dedupe by Station Code and by coordinates.

**Weather:** IMD gridded daily rainfall (0.25 degree) and temperature (1.0 degree) via the `imdlib` Python package, matched to each station by nearest grid cell. Check the latest date IMD actually provides; the test period may have to end earlier. The IMD server is sometimes down.

**Unknown for now (please help me find out):** number of usable stations, date range per station, share of missing days, whether negative sign means depth, and IMD's latest available date.

## 5. Preprocessing decisions

1. Merge all files into one table keyed by Station Code and time. Drop duplicate stations.
2. Average the 4 daily readings to one **daily** value per station.
3. Remove sensor spikes (unrealistic jump between consecutive values) and flat-lined stretches.
4. Interpolate only short gaps (a few days). Keep a **missing flag** as an input. Mask missing targets in the loss. Drop stations with too few days (my starting rule: less than 2 years, or many gaps; tune after seeing the data).
5. Scale each station separately. **Fit scalers on training data only.**
6. Only **past** values are ever used as inputs. Never use future values of any station.

## 6. Features and targets

- Window length T = 30 days (tunable up to 60).
- Per station per day, F = 7 features: scaled water level, daily rainfall, 7-day rainfall sum, 30-day rainfall sum, month as sin and cos, missing flag. Temperature is optional (weaker signal; drop it if it complicates things). Fixed station features (latitude, longitude, well depth, aquifer type) can be added later.
- Targets: water level at t+7 and t+30 days for every station, so output shape per sample is (stations, 2).
- Input tensor shape: (batch, T, N stations, F).

## 7. Split (by time, never random)

Intended: train 2021 to 2024 (last part of it as validation for early stopping), calibration 2025, test 2026 up to the latest date. **Adjust to the real date range once we see the data.** Split windows by the date of the target so splits do not overlap.

## 8. Model, uncertainty, baselines, evaluation (design only)

**Graph:** nodes are stations. Connect each station to its k nearest stations (start with k = 6, try 5 to 8) using Haversine distance from lat/long. Edge weight is a Gaussian of distance (sigma = median neighbour distance). Make it symmetric, add self-loops, and normalize symmetrically (D^-1/2 (A+I) D^-1/2). The graph is small (dozens to a few hundred stations), so a **dense adjacency matrix** is enough; PyTorch Geometric is not required.

**Main model (GCN + GRU):** two GCN layers (hidden 32) applied at every time step to mix neighbour information (space), then one GRU (hidden 64) reading each station's sequence of mixed vectors (time), then a linear head to 2 outputs (7-day and 30-day). One shared model for all stations. Masked MSE loss, Adam (lr 1e-3), early stopping (patience about 7), keep the best checkpoint.

**Uncertainty (the main novelty):**
- Split conformal prediction: predict on the 2025 calibration set, convert to metres, take absolute errors per horizon, use the finite-sample corrected 90th percentile as the margin, and set interval = prediction plus or minus margin.
- Comparison: quantile regression (5th and 95th percentile, pinball loss).
- Known catch: the guarantee assumes calibration and test data behave alike, and time series can drift. So always **measure real coverage on the test set**, do not just trust it. Constant-width intervals may be too wide or too narrow per station; per-station margins are a possible extension.

**Baselines (build these BEFORE the main model):** persistence (next = last), one LSTM per station, one shared LSTM for all stations without a graph.

**Evaluation (test period, per horizon):** RMSE, MAE, PICP (prediction interval coverage probability, target about 90%), mean interval width. Ablations: graph vs identity adjacency (no neighbours), with vs without rainfall. Low-data test: restrict a few stations to about 6 months of history and see whether neighbours help. Report honestly even if the graph model only wins slightly or not at all.

**Demo (last, optional):** save predictions and intervals to a file; React plus Leaflet map where clicking a station shows its forecast chart with the shaded range. FastAPI only if live predictions are wanted.

## 9. Phases (do in order)

- **A. Setup:** environment (pandas, numpy, torch with CUDA build, scikit-learn, imdlib, matplotlib), check the GPU.
- **B. Data:** load the downloaded files, merge, dedupe, daily average, clean. Print station count, date range per station, missing-day counts. I will paste these numbers into my report.
- **C. Weather:** IMD download, nearest grid cell matching, merge.
- **D. Baselines:** persistence, per-station LSTM, shared LSTM on the time split.
- **E. Main model:** graph, windows, GCN + GRU, training. Test on fake data first.
- **F. Uncertainty:** conformal margins and quantile regression comparison.
- **G. Evaluation:** metrics, ablations, low-data test, plots (actual vs predicted with shaded interval).
- **H. Demo and report.**

If time gets tight, cut in this order: map demo, then temperature, then the 30-day horizon. Never cut the baselines or the conformal step.

Suggested folders: `data/raw`, `data/clean`, `notebooks`, `src` (data, graph, models, conformal), `results`, `app`.

## 10. Current status

- Done: topic chosen, abstract and 24-paper literature survey (in a Claude doc), methodology doc, project plan PDF, flowcharts, viewed the India-WRIS portal and one sample station file.
- Not done: bulk download for all Bangalore stations, any cleaning code, IMD download, any training. **No results exist yet.**
- Next step: Phase A and B.

## 11. Terms (full forms)

GCN: Graph Convolutional Network. GRU: Gated Recurrent Unit. LSTM: Long Short-Term Memory. GNN: Graph Neural Network. RMSE: Root Mean Square Error. MAE: Mean Absolute Error. PICP: Prediction Interval Coverage Probability. CGWB: Central Ground Water Board. IMD: India Meteorological Department. NWIC: National Water Informatics Centre. DWLR: Digital Water Level Recorder.
