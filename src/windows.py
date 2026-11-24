import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HORIZONS = (7, 30)


def build_windows(panel, features, T=30, horizons=HORIZONS):
    """Turn the daily panel into samples.
    X: (N, T, F) the last T days up to the origin day t
    Y: (N, H) scaled level at t+h        R: (N, H) True if that target is a REAL reading
    TD: (N, H) target dates              ST: (N,) station code of each sample"""
    Xs, Ys, Rs, TDs, STs = [], [], [], [], []
    for code, g in panel.groupby("station_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < T + max(horizons):
            continue
        feats = g[features].to_numpy(np.float32)
        level_s = g["level_s"].to_numpy(np.float32)           # NaN where no level
        real = g["missing"].to_numpy() == 0
        dates = g["date"].to_numpy()

        origin = np.arange(T - 1, n)                          # window = origin-T+1 .. origin
        win = sliding_window_view(feats, T, axis=0)           # (n-T+1, F, T)
        X = np.transpose(win, (0, 2, 1))                      # (n-T+1, T, F)

        Y, R, TD = [], [], []
        for h in horizons:
            j = origin + h
            inside = j < n
            jj = np.where(inside, j, 0)
            y = np.where(inside, level_s[jj], np.nan)
            Y.append(y)
            R.append(inside & real[jj] & ~np.isnan(y))
            TD.append(np.where(inside, dates[jj], np.datetime64("NaT", "ns")))
        Y, R, TD = np.stack(Y, 1), np.stack(R, 1), np.stack(TD, 1)

        # we only forecast from days that have a level, and keep useful samples
        keep = ~np.isnan(level_s[origin]) & R.any(axis=1)
        Xs.append(X[keep]); Ys.append(Y[keep]); Rs.append(R[keep]); TDs.append(TD[keep])
        STs.append(np.full(keep.sum(), code))
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Rs),
            np.concatenate(TDs), np.concatenate(STs))


def split_masks(R, TD, split):
    """Per-horizon masks, by TARGET date."""
    v, t, c = (np.datetime64(split[k]) for k in ("val_start", "train_end", "calib_end"))
    return {
        "train": R & (TD < v),
        "val":   R & (TD >= v) & (TD <= t),
        "calib": R & (TD > t) & (TD <= c),
        "test":  R & (TD > c),
    }