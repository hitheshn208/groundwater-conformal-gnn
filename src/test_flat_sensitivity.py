import pandas as pd
from config import DROP
from data import flag_flat

raw = pd.read_csv("data/clean/daily_raw.csv", dtype={"station_code": str},
                parse_dates=["date"])

rows = []
for code, g in raw.groupby("station_code"):
    if code in DROP:
        continue
    s = g.set_index("date")["level"].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    s = s.mask(s > 0)                       # range rule first, like the cleaner
    row = {"station_code": code}
    for tol in [0.0, 0.01, 0.02, 0.03, 0.05]:
        row[f"tol_{tol}"] = int(flag_flat(s, window=10, tol=tol).sum())
    rows.append(row)

print(pd.DataFrame(rows).to_string(index=False))