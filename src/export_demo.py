import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from config import SPLIT
from windows import split_masks, HORIZONS
from evaluate import to_metres
from graph_data import build_dense, build_targets

# Exports the GCN+GRU test-period results for the static map demo in app/.
# Every number is recomputed from results/gcn_pred.npz and cross-checked against the saved CSVs.
NAME = "gcn"
ALPHA = 0.10
T = 30
LOW_COV = 0.85
OUT = Path("app/public/demo_data.json")


def conformal_q(abs_err, alpha=ALPHA):
    """Same finite-sample rule as conformal.py: the ceil((n+1)(1-alpha))-th smallest error."""
    n = len(abs_err)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(abs_err)[k - 1]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r3(x):
    return [round(float(v), 3) for v in x]


def target_dates(d):
    """Rebuild the target date of every saved row (same row rule as flatten(), as in plot_intervals.py)."""
    panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
    scalers = json.load(open("data/clean/scalers.json"))
    stations, dates, F, L, real, ok = build_dense(panel, scalers["features"], T=T)
    origins, Y, R, TD = build_targets(dates, L, real, ok, T=T)
    masks = split_masks(R, TD, SPLIT)
    anyr = np.zeros((Y.shape[0], Y.shape[2]), bool)
    for m in masks.values():
        anyr |= m.any(axis=1)
    o, s = np.where(anyr)
    assert len(o) == len(d["ST"]) and (np.array(stations)[s] == d["ST"]).all(), "row order mismatch"
    # one column per horizon, so dates[:, hi] lines up with pred[:, hi]; NaT past the panel end
    # (those rows have no real target at that horizon, so no mask selects them)
    j = origins[o][:, None] + np.array(HORIZONS)[None, :]
    return np.where(j < len(dates), dates[np.minimum(j, len(dates) - 1)], np.datetime64("NaT", "ns"))


def station_horizon(err, true_m, pred_m, base_m, td, mc, mt):
    """Metrics and test series for one station and one horizon (inputs already sliced to that station)."""
    q = conformal_q(err[mc])
    assert np.isfinite(q), "too few calibration rows for a finite interval"
    r_model, r_pers = rmse(pred_m[mt], true_m[mt]), rmse(base_m[mt], true_m[mt])
    cov = float((err[mt] <= q).mean())
    order = np.argsort(td[mt])
    p = pred_m[mt][order]
    metrics = {"rmse_model": r_model, "rmse_persistence": r_pers, "skill": 1 - r_model / r_pers,
               "q_m": float(q), "width_m": float(2 * q), "coverage": cov,
               "err_ratio": float(err[mt].mean() / err[mc].mean()),
               "n_calib": int(mc.sum()), "n_test": int(mt.sum()), "low_coverage": bool(cov < LOW_COV)}
    series = {"date": [str(x)[:10] for x in td[mt][order]], "observed": r3(true_m[mt][order]),
              "forecast": r3(p), "lower": r3(p - q), "upper": r3(p + q)}
    return metrics, series


def check_against_csvs(per_station, headline):
    """Stop if anything we recomputed differs from the numbers already saved in results/."""
    summ = pd.read_csv("results/conformal_summary.csv")
    glob = pd.read_csv(f"results/{NAME}_metrics.csv")
    for h in HORIZONS:
        by = pd.read_csv(f"results/{NAME}_by_station_h{h}.csv", dtype={"station": str}).set_index("station")
        cf = pd.read_csv(f"results/conformal_stations_{NAME}_h{h}.csv", dtype={"station": str}).set_index("station")
        for code, m in per_station.items():
            m = m.get(str(h))
            if m is None:
                continue
            assert np.isclose(m["rmse_model"], by.loc[code, f"RMSE_{NAME}"]), (code, h, "rmse model")
            assert np.isclose(m["rmse_persistence"], by.loc[code, "RMSE_persist"]), (code, h, "rmse persist")
            assert np.isclose(m["skill"], by.loc[code, "skill"]), (code, h, "skill")
            for k in ("width_m", "coverage", "err_ratio", "n_calib", "n_test"):
                assert np.isclose(m[k], cf.loc[code, k]), (code, h, k)
        g = glob[(glob.horizon == h) & (glob.split == "test")].iloc[0]
        s = summ[(summ.horizon == h) & (summ.model == NAME) & (summ.calibration == "per-station")].iloc[0]
        hl = headline[str(h)]
        assert np.isclose(hl["rmse_model"], g[f"RMSE_{NAME}"]) and np.isclose(hl["rmse_persistence"], g["RMSE_persist"])
        assert np.isclose(hl["coverage"], s["coverage"]) and np.isclose(hl["mean_width_m"], s["mean_width_m"])


def main():
    scalers = json.load(open("data/clean/scalers.json"))
    d = np.load(f"results/{NAME}_pred.npz")
    ST = d["ST"]
    true_m = to_metres(d["Y"], ST, scalers)
    pred_m = to_metres(d["pred"], ST, scalers)
    base_m = to_metres(d["today"], ST, scalers)          # persistence = today's level
    TD = target_dates(d)

    st = pd.read_csv("data/clean/stations.csv", dtype={"station_code": str})
    st = st[st["station_code"].isin(set(ST))].sort_values("station_code")
    per_station = {c: {} for c in st["station_code"]}
    series = {c: {} for c in st["station_code"]}
    headline = {}

    for hi, h in enumerate(HORIZONS):
        err = np.abs(pred_m[:, hi] - true_m[:, hi])
        mc, mt = d["mask_calib"][:, hi], d["mask_test"][:, hi]
        # test rows must all have target dates after calib_end, otherwise the "test period" note is wrong
        assert (TD[mt, hi] > np.datetime64(SPLIT["calib_end"])).all()
        covered, widths = [], []
        for code in per_station:
            sel = ST == code
            if (mt & sel).sum() == 0:
                continue                                  # e.g. CGWBNG0018: shown as "no test data"
            m, s = station_horizon(err[sel], true_m[sel, hi], pred_m[sel, hi], base_m[sel],
                                   TD[sel, hi], mc[sel], mt[sel])
            per_station[code][str(h)], series[code][str(h)] = m, s
            covered.append(err[sel & mt] <= m["q_m"])
            widths.append(np.full(m["n_test"], m["width_m"]))
        # pooled over all test rows, same definitions as gcn_metrics.csv and conformal.py
        headline[str(h)] = {"rmse_model": rmse(pred_m[mt, hi], true_m[mt, hi]),
                            "rmse_persistence": rmse(base_m[mt], true_m[mt, hi]),
                            "coverage": float(np.concatenate(covered).mean()),
                            "mean_width_m": float(np.concatenate(widths).mean()),
                            "n_test": int(mt.sum())}

    check_against_csvs(per_station, headline)

    stations = [{"code": r.station_code, "name": r.station_name, "lat": float(r.lat), "lon": float(r.lon),
                 "has_test": bool(per_station[r.station_code])} for r in st.itertuples()]
    out = {"meta": {"model": "GCN+GRU", "horizons": list(HORIZONS), "split": SPLIT,
                    "coverage_target": 1 - ALPHA, "low_coverage_threshold": LOW_COV,
                    "note": f"Test-period forecasts (target dates after {SPLIT['calib_end']}), not live forecasts."},
           "headline": headline, "stations": stations, "metrics": per_station, "series": series}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))

    report(headline, per_station)


def report(headline, per_station):
    """Print the checks requested for this export; exit non-zero if they fail."""
    expect = {"7": (1.258, 7727), "30": (2.652, 7499)}
    ok = True
    for h, (rp, n) in expect.items():
        hl = headline[h]
        good = round(hl["rmse_persistence"], 3) == rp and hl["n_test"] == n
        ok &= good
        print(f"h={h:>2}: persistence RMSE {hl['rmse_persistence']:.3f} (expect {rp}) | n_test {hl['n_test']} "
              f"(expect {n}) | model RMSE {hl['rmse_model']:.3f} | coverage {hl['coverage']:.3f} | "
              f"mean width {hl['mean_width_m']:.3f} m  -> {'OK' if good else 'MISMATCH'}")
    no_test = [c for c, m in per_station.items() if not m]
    low = {h: sorted(c for c, m in per_station.items() if m and m[h]["low_coverage"]) for h in expect}
    print("stations:", len(per_station), "| no test data:", no_test)
    print("low coverage (<0.85):", low)
    print("all per-station numbers match results/*.csv")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    if not ok:
        sys.exit("checks FAILED - not using this export")


if __name__ == "__main__":
    main()
