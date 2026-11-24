import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SPLIT
from windows import split_masks, HORIZONS
from evaluate import to_metres
from graph_data import build_dense, build_targets

NAME = sys.argv[1] if len(sys.argv) > 1 else "gcn"
SHOW = sys.argv[2:] or ["CGWBNG0017", "CGWBNG0008", "CGWBNG0019", "CGWBNG0007"]
ALPHA = 0.10
HI = 0                                           # 0 = 7-day horizon, 1 = 30-day
H = HORIZONS[HI]
T = 30


def conformal_q(abs_err, alpha=ALPHA):
    n = len(abs_err)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(abs_err)[k - 1]


panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
scalers = json.load(open("data/clean/scalers.json"))
stations, dates, F, L, real, ok = build_dense(panel, scalers["features"], T=T)
origins, Y, R, TD = build_targets(dates, L, real, ok, T=T)
masks = split_masks(R, TD, SPLIT)

# rebuild the row order used when the predictions were saved (same rule as flatten())
anyr = np.zeros((Y.shape[0], Y.shape[2]), bool)
for m in masks.values():
    anyr |= m.any(axis=1)
o, s = np.where(anyr)
d = np.load(f"results/{NAME}_pred.npz")
assert len(o) == len(d["ST"]) and (np.array(stations)[s] == d["ST"]).all(), "row order mismatch"

ST = d["ST"]
true_m = to_metres(d["Y"][:, HI], ST, scalers)
pred_m = to_metres(d["pred"][:, HI], ST, scalers)
target_date = dates[origins[o] + H]
err = np.abs(pred_m - true_m)
mc, mt = d["mask_calib"][:, HI], d["mask_test"][:, HI]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, code in zip(axes.ravel(), SHOW):
    c, t = mc & (ST == code), mt & (ST == code)
    if t.sum() == 0:
        ax.set_title(f"{code}: no test data"); continue
    q = conformal_q(err[c])                                   # this station's own calibration margin
    order = np.argsort(target_date[t])
    x = target_date[t][order]
    ax.fill_between(x, (pred_m[t] - q)[order], (pred_m[t] + q)[order], alpha=0.25, label="90% interval")
    ax.plot(x, pred_m[t][order], lw=1.2, label="forecast")
    ax.plot(x, true_m[t][order], ".", ms=3, color="k", label="observed")
    cov = (err[t] <= q).mean()
    ax.set_title(f"{code}  |  width {2 * q:.2f} m  |  test coverage {cov:.0%}")
    ax.set_ylabel("level (m)")
    ax.legend(loc="best", fontsize=8)
fig.suptitle(f"{NAME}: {H}-day forecasts on the test period (target dates)")
fig.autofmt_xdate()
Path("results").mkdir(exist_ok=True)
out = f"results/intervals_{NAME}_h{H}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)