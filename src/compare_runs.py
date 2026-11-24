import sys
from pathlib import Path
import pandas as pd

names = sys.argv[1:]
low_file = Path("results/low_stations.txt")
LOW = low_file.read_text().split() if low_file.exists() else []
rows = []
for h in (7, 30):
    for n in names:
        p = pd.read_csv(f"results/{n}_by_station_h{h}.csv", index_col="station", dtype={"station": str})
        r = {"horizon": h, "run": n,
            "median_RMSE_m": p[f"RMSE_{n}"].median(),
            "median_skill": p["skill"].median(),
            "wins": f"{int((p['skill'] > 0).sum())}/{len(p)}"}
        if LOW:
            r["low_stations_median_skill"] = p.loc[p.index.intersection(LOW), "skill"].median()
        rows.append(r)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))