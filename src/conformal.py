import json
import sys
import numpy as np
import pandas as pd
from evaluate import to_metres
from windows import HORIZONS

NAMES = sys.argv[1:] or ["gcn"]          # e.g. python src/conformal.py gcn nograph
ALPHA = 0.10                              # 90% intervals
scalers = json.load(open("data/clean/scalers.json"))


def conformal_q(abs_err, alpha=ALPHA):
    """Finite-sample rule: the ceil((n+1)(1-alpha))-th smallest calibration error (in metres)."""
    n = len(abs_err)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(abs_err)[k - 1]


def evaluate_intervals(label, pred_m, true_m, ST, mc, mt):
    """mc / mt = calibration / test masks for ONE horizon. Returns two rows (global, per-station)."""
    err = np.abs(pred_m - true_m)
    q_global = conformal_q(err[mc])
    rows = []

    # (a) one width for all stations
    covered = err[mt] <= q_global
    st_cov = pd.Series(covered).groupby(ST[mt]).mean()
    rows.append(("global", int(mc.sum()), int(mt.sum()), covered.mean(), 2 * q_global,
                st_cov.min(), st_cov.median()))

    # (b) one width per station, from that station's own calibration errors
    q_by = {}
    for s in np.unique(ST[mt]):
        q = conformal_q(err[mc & (ST == s)])
        q_by[s] = q_global if np.isinf(q) else q          # too few calibration points -> global width
    q_pt = np.array([q_by[s] for s in ST[mt]])
    covered = err[mt] <= q_pt
    st_cov = pd.Series(covered).groupby(ST[mt]).mean()
    rows.append(("per-station", int(mc.sum()), int(mt.sum()), covered.mean(), 2 * q_pt.mean(),
                st_cov.min(), st_cov.median()))
    return [(label,) + r for r in rows]


all_rows = []
for name in NAMES:
    d = np.load(f"results/{name}_pred.npz")
    ST = d["ST"]
    true_m = to_metres(d["Y"], ST, scalers)
    methods = {name: to_metres(d["pred"], ST, scalers),
                "persistence": np.repeat(to_metres(d["today"], ST, scalers)[:, None], len(HORIZONS), axis=1)}
    for label, pred_m in methods.items():
        if name != NAMES[0] and label == "persistence":
            continue                                        # persistence only needs doing once
        for hi, h in enumerate(HORIZONS):
            for r in evaluate_intervals(label, pred_m[:, hi], true_m[:, hi], ST,
                                        d["mask_calib"][:, hi], d["mask_test"][:, hi]):
                all_rows.append((h,) + r)

cols = ["horizon", "model", "calibration", "n_calib", "n_test", "coverage", "mean_width_m",
        "station_cov_min", "station_cov_median"]
out = pd.DataFrame(all_rows, columns=cols).sort_values(["horizon", "calibration", "model"])
pd.set_option("display.width", 200)
print(f"target coverage = {1 - ALPHA:.0%}\n")
print(out.round(3).to_string(index=False))
out.to_csv("results/conformal_summary.csv", index=False)