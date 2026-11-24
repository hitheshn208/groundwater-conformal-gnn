import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import SPLIT

FEATURES = ["level_in", "rain_s", "rain7_s", "rain30_s",
            "month_sin", "month_cos", "missing"]          # F = 7

clean = pd.read_csv("data/clean/daily_clean.csv", dtype={"station_code": str},
                    parse_dates=["date"])
rain = pd.read_csv("data/clean/rain_daily.csv", dtype={"station_code": str},
                    parse_dates=["date"])

# 1. rain features use today and PAST days only
rain = rain.sort_values(["station_code", "date"])
by_station = rain.groupby("station_code")["rain"]
rain["rain7"] = by_station.transform(lambda s: s.rolling(7).sum())
rain["rain30"] = by_station.transform(lambda s: s.rolling(30).sum())

panel = clean[["station_code", "date", "level", "missing"]].merge(
    rain[["station_code", "date", "rain", "rain7", "rain30"]],
    on=["station_code", "date"], how="left")

# 2. month as a circle, so December and January are neighbours
month = panel["date"].dt.month
panel["month_sin"] = np.sin(2 * np.pi * month / 12)
panel["month_cos"] = np.cos(2 * np.pi * month / 12)

# 3. scale each station using TRAINING days only
is_train = panel["date"] <= SPLIT["train_end"]
real_train = panel[is_train & (panel["missing"] == 0)]
stats = real_train.groupby("station_code")["level"].agg(mean="mean", std="std", n="size")
stats["std"] = stats["std"].fillna(1.0).replace(0, 1.0)

no_train = sorted(set(panel["station_code"]) - set(stats.index))
if no_train:
    print("DROPPED (no real training days):", no_train)
    panel = panel[~panel["station_code"].isin(no_train)].copy()
    is_train = panel["date"] <= SPLIT["train_end"]

code = panel["station_code"]
panel["level_s"] = (panel["level"] - code.map(stats["mean"])) / code.map(stats["std"])

rain_scale = {}
for c in ["rain", "rain7", "rain30"]:
    logc = np.log1p(panel[c])                      # rain is very skewed, so log it
    mu, sd = logc[is_train].mean(), logc[is_train].std()
    rain_scale[c] = {"mean": float(mu), "std": float(sd)}
    panel[c + "_s"] = (logc - mu) / sd

# 4. a missing level becomes 0 (= station average); the missing flag says so
panel["level_in"] = panel["level_s"].fillna(0.0)

Path("data/clean").mkdir(parents=True, exist_ok=True)
panel.to_csv("data/clean/panel.csv", index=False)
scalers = {
    "level": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
            for k, v in stats.iterrows()},
    "rain": rain_scale, "split": SPLIT, "features": FEATURES,
}
with open("data/clean/scalers.json", "w") as f:
    json.dump(scalers, f, indent=1)

# ---- checks ----
print("stations:", panel["station_code"].nunique(), "| rows:", len(panel))
print("NaN in model inputs:", int(panel[FEATURES].isna().sum().sum()))
print("real training days per station: min", int(stats["n"].min()),
    "| median", int(stats["n"].median()))
few = stats[stats["n"] < 100]
print("stations with < 100 real training days:", list(few.index))
tr = panel[is_train & (panel["missing"] == 0)]
print("scaled level on train: mean", round(tr["level_s"].mean(), 3),
    "| std", round(tr["level_s"].std(), 3))
print("share of days flagged missing:", round(panel["missing"].mean(), 3))