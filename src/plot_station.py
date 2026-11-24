import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from data import load_all_stations, to_daily, flag_spikes

# station codes come from the command line, or use these three by default
codes = sys.argv[1:] or ["CGWBNG0018", "CGWBNG0030", "CGWBNG0009"]

Path("results").mkdir(exist_ok=True)
readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)
names = dict(zip(stations["station_code"], stations["station_name"]))

fig, axes = plt.subplots(len(codes), 1, figsize=(12, 3.5 * len(codes)), squeeze=False)
for ax, code in zip(axes[:, 0], codes):
    g = daily[daily["station_code"] == code].set_index("date")["level"].sort_index()
    full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    spikes = flag_spikes(full)
    ax.plot(full.index[~spikes], full[~spikes], ".", markersize=2, label="kept")
    ax.plot(full.index[spikes], full[spikes], "r.", markersize=5, label="flagged")
    ax.set_title(f"{code} {names.get(code, '')}   flagged: {int(spikes.sum())}")
    ax.legend()

out = f"results/check_{codes[0]}.png"
fig.savefig(out, dpi=120)
print("saved", out)