import time
from pathlib import Path
import pandas as pd
import imdlib as imd

folder = Path("data/raw/imd_realtime")
names = [p.name for p in folder.rglob("*") if p.is_file()]

wanted = pd.date_range("2026-01-01", "2026-06-04", freq="D")
missing = [d for d in wanted
        if not any(n.startswith(f"rain_ind0.25_{d:%y_%m_%d}") for n in names)]
print("files found:", len(names), "| missing days:", len(missing))
print([d.strftime("%Y-%m-%d") for d in missing])

for d in missing:
    day = d.strftime("%Y-%m-%d")
    for attempt in range(1, 4):
        try:
            imd.get_real_data("rain", day, day, str(folder))
            print(day, "downloaded")
            break
        except Exception as e:
            print(day, f"attempt {attempt} failed:", e)
            time.sleep(5)