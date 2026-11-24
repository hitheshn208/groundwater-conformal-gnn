import numpy as np
import pandas as pd

H = 30
panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
rows = []
for code, g in panel.groupby("station_code"):
    g = g.sort_values("date").set_index("date")
    lvl = g["level_s"].where(g["missing"] == 0)          # real readings only (scaling keeps the sign)
    full = lvl.reindex(pd.date_range(lvl.index.min(), lvl.index.max(), freq="D"))
    rain = g["rain30_s"].reindex(full.index)              # rain30_s at day d = rain over the 30 days up to d
    change = full.shift(-H) - full                        # level change from d to d+30
    rain_after = rain.shift(-H)                           # rain that fell between d and d+30
    ok = change.notna() & rain_after.notna()
    if ok.sum() > 100:
        rows.append({"station": code, "n": int(ok.sum()), "corr": np.corrcoef(change[ok], rain_after[ok])[0, 1]})
out = pd.DataFrame(rows).set_index("station").sort_values("corr")
print(out.round(2).to_string())
print(f"\nmedian correlation {out['corr'].median():.2f} | positive at {(out['corr'] > 0).sum()} of {len(out)} stations")