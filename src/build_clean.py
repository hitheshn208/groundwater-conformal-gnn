from pathlib import Path
import pandas as pd
from config import DROP, VALID_UNTIL, MASK_RANGES
from data import load_all_stations, to_daily, clean_station

readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)

parts, rows = [], []
for code, g in daily.groupby("station_code"):
    if code in DROP:
        continue
    c = clean_station(g, valid_until=VALID_UNTIL.get(code), mask_ranges=MASK_RANGES.get(code, ()))
    c["station_code"] = code
    parts.append(c.reset_index())
    rows.append({
        "station_code": code,
        "calendar_days": len(c),
        "real": int((c["missing"] == 0).sum()),
        "interpolated": int(((c["missing"] == 1) & c["level"].notna()).sum()),
        "above_0": int(c["range_bad"].sum()),
        "flat": int(c["flat"].sum()),
        "spikes": int(c["spike"].sum()),
        "after_cutoff": int(c["after_cutoff"].sum()),
    })

clean = pd.concat(parts, ignore_index=True)
kept = stations[~stations["station_code"].isin(DROP)]

Path("data/clean").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)
clean.to_csv("data/clean/daily_clean.csv", index=False)
kept.to_csv("data/clean/stations.csv", index=False)
daily.to_csv("data/clean/daily_raw.csv", index=False)
summary = pd.DataFrame(rows)
summary["real_pct"] = (100 * summary["real"] / summary["calendar_days"]).round(1)
pd.set_option("display.width", 220)
print("stations kept:", len(summary), "| dropped:", len(DROP))
print(summary.to_string(index=False))
summary.to_csv("results/clean_summary.csv", index=False)