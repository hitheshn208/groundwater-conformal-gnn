import json
import numpy as np
import pandas as pd
from config import SPLIT

H = 7
panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
scalers = json.load(open("data/clean/scalers.json"))
t, c = pd.Timestamp(SPLIT["train_end"]), pd.Timestamp(SPLIT["calib_end"])


def period_of(d):
    return "train" if d <= t else ("calib" if d <= c else "test")


rows = []
for code, g in panel.groupby("station_code"):
    g = g.sort_values("date").set_index("date")
    # level_s is already scaled with TRAIN mean/std, so std of level_s later = spread ratio
    s = g["level_s"].where(g["missing"] == 0)
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    std_m = scalers["level"][code]["std"]

    tr = full[full.index <= t].dropna()
    te = full[full.index > c].dropna()
    row = {"station": code,
            "train_std_m": std_m,
            # share of test days outside the min..max the model saw in training
            "pct_outside": 100 * ((te < tr.min()) | (te > tr.max())).mean() if len(te) and len(tr) else np.nan,
            "spread_ratio": te.std() / tr.std() if len(te) > 5 and len(tr) > 5 else np.nan}

    # size of a 7-day change, grouped by the TARGET date, in metres
    change = (full.shift(-H) - full).dropna() * std_m
    target_dates = change.index + pd.Timedelta(days=H)
    periods = np.array([period_of(d) for d in target_dates])
    for p in ["train", "calib", "test"]:
        x = change[periods == p]
        row[f"chg7_{p}_m"] = x.abs().median() if len(x) else np.nan
    rows.append(row)

out = pd.DataFrame(rows).set_index("station")
pd.set_option("display.width", 200)
print(out.round(2).sort_values("pct_outside", ascending=False).to_string())
print("\nmedian over stations:")
print(out[["spread_ratio", "pct_outside", "chg7_train_m", "chg7_calib_m", "chg7_test_m"]].median().round(3))
out.to_csv("results/shift_check.csv")