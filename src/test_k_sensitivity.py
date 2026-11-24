import pandas as pd
from data import load_all_stations, to_daily, flag_spikes

readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)

for code, g in daily.groupby("station_code"):
    s = g.set_index("date")["level"].sort_index()
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    total = full.notna().sum()
    print(code, "days with data:", total)
    for k in [3, 4, 5, 6, 8]:
        n = int(flag_spikes(full, k=k).sum())
        print(f"  k={k}: flagged {n} ({100 * n / total:.1f}%)")