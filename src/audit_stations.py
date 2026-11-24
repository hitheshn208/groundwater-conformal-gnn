import pandas as pd
from data import load_all_stations, to_daily

pd.set_option("display.width", 220)
readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)
depth = pd.to_numeric(stations.set_index("station_code")["well_depth"], errors="coerce")

def longest_run(mask):
    """Longest stretch of consecutive True values, in days."""
    run_id = (mask != mask.shift()).cumsum()
    return int(mask.groupby(run_id).sum().max()) + 1

rows = []
for code, g in daily.groupby("station_code"):
    s = g.sort_values("date").set_index("date")["level"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))

    step = s.diff().abs()
    wd = depth.get(code)
    deeper = int((s < -wd).sum()) if pd.notna(wd) else None
    roll = s.rolling(10, min_periods=8)
    rng = roll.max() - roll.min()      # how much the level moved in the last 10 days
    rows.append({
        "station_code": code,
        "well_depth": wd,
        "min": round(s.min(), 1),
        "max": round(s.max(), 1),
        "days_above_0": int((s > 0).sum()),
        "days_deeper_than_well": deeper,       
        "flat_days_0.05": int((rng <= 0.05).sum()),
        "flat_days_0.10": int((rng <= 0.10).sum()),
    })

audit = pd.DataFrame(rows).sort_values("flat_days_0.05", ascending=False)
print(audit.to_string(index=False))
audit.to_csv("results/station_audit.csv", index=False)