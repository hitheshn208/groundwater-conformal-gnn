import json
import sys
import numpy as np
import pandas as pd
from evaluate import to_metres
from windows import HORIZONS

ALPHA = 0.10
name = sys.argv[1] if len(sys.argv) > 1 else "gcn"
scalers = json.load(open("data/clean/scalers.json"))
d = np.load(f"results/{name}_pred.npz")
ST = d["ST"]
true_m = to_metres(d["Y"], ST, scalers)
pred_m = to_metres(d["pred"], ST, scalers)


def conformal_q(abs_err, alpha=ALPHA):
    n = len(abs_err)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(abs_err)[k - 1]


pd.set_option("display.width", 200)
for hi, h in enumerate(HORIZONS):
    err = np.abs(pred_m[:, hi] - true_m[:, hi])
    mc, mt = d["mask_calib"][:, hi], d["mask_test"][:, hi]
    rows = []
    for s in np.unique(ST[mt]):
        c, t = mc & (ST == s), mt & (ST == s)
        q = conformal_q(err[c])
        rows.append({"station": s, "n_calib": int(c.sum()), "n_test": int(t.sum()),
                     "width_m": 2 * q, "coverage": (err[t] <= q).mean(),
                     # >1 means test errors were bigger than calibration errors -> under-coverage
                     "err_ratio": err[t].mean() / err[c].mean()})
    out = pd.DataFrame(rows).set_index("station").sort_values("coverage")
    print(f"\n=== {name}, horizon {h} days (sorted by coverage, worst first) ===")
    print(out.round(3).to_string())
    print(f"median station width {out['width_m'].median():.2f} m | "
          f"stations below 85% coverage: {int((out['coverage'] < 0.85).sum())} of {len(out)}")
    out.to_csv(f"results/conformal_stations_{name}_h{h}.csv")