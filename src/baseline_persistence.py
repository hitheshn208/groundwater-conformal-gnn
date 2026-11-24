import numpy as np
import pandas as pd
from config import SPLIT

HORIZONS = [7, 30]

clean = pd.read_csv("data/clean/daily_clean.csv", dtype={"station_code": str},
                    parse_dates=["date"])


def make_samples(g, h):
    """One station. Forecast made on day t says: level on day t+h = level on day t."""
    g = g.set_index("date").sort_index()
    target = g["level"].where(g["missing"] == 0)      # only REAL readings are targets
    out = pd.DataFrame({
        "target_date": g.index + pd.Timedelta(days=h),
        "pred": g["level"].values,                    # today's (cleaned) level
    })
    out["true"] = target.reindex(pd.DatetimeIndex(out["target_date"])).values
    return out.dropna(subset=["pred", "true"])


def metrics(df):
    err = df["pred"] - df["true"]
    return pd.Series({"n": len(df),
                      "RMSE": np.sqrt((err ** 2).mean()),
                      "MAE": err.abs().mean()})


rows, per_station = [], []
for h in HORIZONS:
    parts = []
    for code, g in clean.groupby("station_code"):
        s = make_samples(g, h)
        s["station_code"] = code
        parts.append(s)
    samples = pd.concat(parts, ignore_index=True)

    td = samples["target_date"]
    samples["split"] = np.select(
        [td <= SPLIT["train_end"], td <= SPLIT["calib_end"]],
        ["train", "calib"], default="test")

    for split in ["train", "calib", "test"]:
        m = metrics(samples[samples["split"] == split])
        m["horizon"], m["split"] = h, split
        rows.append(m)

    test = samples[samples["split"] == "test"].copy()
    test["err"] = test["pred"] - test["true"]
    per = test.groupby("station_code")["err"].agg(
        n="size",
        RMSE=lambda e: np.sqrt((e ** 2).mean()),
        MAE=lambda e: e.abs().mean())
    per["horizon"] = h
    per_station.append(per.reset_index())

summary = pd.DataFrame(rows)[["horizon", "split", "n", "RMSE", "MAE"]]
pd.set_option("display.width", 200)
print(summary.round(3).to_string(index=False))

per_station = pd.concat(per_station, ignore_index=True)
per_station.to_csv("results/persistence_by_station.csv", index=False)
for h in HORIZONS:
    p = per_station[per_station["horizon"] == h]
    print(f"\nh={h}: median per-station test RMSE = {p['RMSE'].median():.3f} m "
            f"over {len(p)} stations")
    print("  worst 3:", p.nlargest(3, "RMSE")[["station_code", "RMSE"]]
            .round(2).values.tolist())
    print("  best 3: ", p.nsmallest(3, "RMSE")[["station_code", "RMSE"]]
            .round(2).values.tolist())