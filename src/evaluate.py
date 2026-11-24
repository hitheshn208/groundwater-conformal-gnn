import numpy as np
import pandas as pd
from windows import HORIZONS


def to_metres(arr_s, stations, scalers):
    """Scaled level -> metres, using each station's own mean and std."""
    mean = np.array([scalers["level"][c]["mean"] for c in stations])
    std = np.array([scalers["level"][c]["std"] for c in stations])
    if arr_s.ndim == 2:
        return arr_s * std[:, None] + mean[:, None]
    return arr_s * std + mean


def evaluate(pred_s, X, Y, masks, ST, scalers, name="model"):
    """Compare a model with persistence on the SAME samples, in metres."""
    pred_m = to_metres(pred_s, ST, scalers)
    true_m = to_metres(Y, ST, scalers)
    base_m = to_metres(X[:, -1, 0], ST, scalers)        # persistence = today's level

    rows = []
    for split in ["train", "val", "calib", "test"]:
        for hi, h in enumerate(HORIZONS):
            m = masks[split][:, hi]
            if m.sum() == 0:
                continue
            e_mod = pred_m[m, hi] - true_m[m, hi]
            e_per = base_m[m] - true_m[m, hi]
            rows.append({"horizon": h, "split": split, "n": int(m.sum()),
                         f"RMSE_{name}": np.sqrt((e_mod ** 2).mean()),
                         "RMSE_persist": np.sqrt((e_per ** 2).mean()),
                         f"MAE_{name}": np.abs(e_mod).mean(),
                         "MAE_persist": np.abs(e_per).mean()})
    table = pd.DataFrame(rows)

    per_station = {}
    for hi, h in enumerate(HORIZONS):
        m = masks["test"][:, hi]
        df = pd.DataFrame({"station": ST[m],
                           "se_mod": (pred_m[m, hi] - true_m[m, hi]) ** 2,
                           "se_per": (base_m[m] - true_m[m, hi]) ** 2})
        per = np.sqrt(df.groupby("station")[["se_mod", "se_per"]].mean())
        per.columns = [f"RMSE_{name}", "RMSE_persist"]
        per["skill"] = 1 - per[f"RMSE_{name}"] / per["RMSE_persist"]   # >0 = beats persistence
        per_station[h] = per
    return table, per_station