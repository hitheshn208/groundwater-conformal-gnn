from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from data import load_all_stations, to_daily, flag_spikes

Path("results").mkdir(exist_ok=True)
readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)

codes = sorted(daily["station_code"].unique())
fig, axes = plt.subplots(len(codes), 1, figsize=(12, 3.5 * len(codes)), squeeze=False)

for ax, code in zip(axes[:, 0], codes):
    g = daily[daily["station_code"] == code].set_index("date")["level"].sort_index()
    full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    spikes = flag_spikes(full)
    print(f"{code}: flagged {int(spikes.sum())} of {full.notna().sum()} days")
    ax.plot(full.index[~spikes], full[~spikes], ".", markersize=2, label="kept")
    ax.plot(full.index[spikes], full[spikes], "r.", markersize=5, label="flagged")
    ax.set_title(code)
    ax.legend()

fig.tight_layout()
fig.savefig("results/spikes_flagged.png", dpi=120)
print("saved results/spikes_flagged.png")