import numpy as np
import pandas as pd
from windows import HORIZONS, split_masks


def build_dense(panel, features, T=30, horizons=HORIZONS):
    """Panel (long) -> arrays over (date, station). Same samples as windows.py, but all
    stations of one day stay together so the graph can mix them.
    F_arr (D,S,F) | L_arr (D,S) scaled level | real (D,S) | origin_ok (D,S) | dates (D,)"""
    stations = sorted(panel["station_code"].unique())
    dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    D, S, F = len(dates), len(stations), len(features)
    F_arr = np.zeros((D, S, F), np.float32)
    L_arr = np.full((D, S), np.nan, np.float32)
    real = np.zeros((D, S), bool)
    start_ok = np.zeros((D, S), bool)
    for s, code in enumerate(stations):
        g = panel[panel["station_code"] == code].sort_values("date")
        idx = dates.get_indexer(g["date"])
        F_arr[idx, s] = np.nan_to_num(g[features].to_numpy(np.float32))
        L_arr[idx, s] = g["level_s"].to_numpy(np.float32)
        real[idx, s] = g["missing"].to_numpy() == 0
        start_ok[idx[0] + T - 1:, s] = True            # window must lie inside the station's own history
    origin_ok = start_ok & ~np.isnan(L_arr)             # we forecast only from days that have a level
    return stations, dates.to_numpy(), F_arr, L_arr, real, origin_ok


def build_targets(dates, L_arr, real, origin_ok, T=30, horizons=HORIZONS):
    """For every origin day i (>= T-1): Y (n,H,S), R (n,H,S) real-target mask, TD (n,H,S) target dates."""
    D, S = L_arr.shape
    origins = np.arange(T - 1, D)
    Y = np.full((len(origins), len(horizons), S), np.nan, np.float32)
    R = np.zeros(Y.shape, bool)
    TD = np.full(Y.shape, np.datetime64("NaT", "ns"))
    for hi, h in enumerate(horizons):
        j = origins + h
        inside = j < D
        jj = np.where(inside, j, 0)
        Y[:, hi] = np.where(inside[:, None], L_arr[jj], np.nan)
        R[:, hi] = inside[:, None] & real[jj] & ~np.isnan(L_arr[jj]) & origin_ok[origins]
        TD[:, hi] = np.where(inside[:, None], dates[jj][:, None], np.datetime64("NaT", "ns"))
    return origins, Y, R, TD


def flatten(origins, Y, masks, origin_ok, L_arr, stations, pred=None):
    """(origin, station) pairs that have at least one real target -> rows, like windows.py."""
    anyr = np.zeros(Y.shape[0:1] + (Y.shape[2],), bool)
    for m in masks.values():
        anyr |= m.any(axis=1)
    o, s = np.where(anyr)
    out = {"Y": Y[o, :, s], "ST": np.array(stations)[s],
            "today": L_arr[origins[o], s],
            "masks": {k: m[o, :, s] for k, m in masks.items()}}
    if pred is not None:
        out["pred"] = pred[o, :, s]
    return out