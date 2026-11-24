from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from data import load_all_stations, to_daily

Path("results").mkdir(exist_ok=True)
readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)

# ---- 1. plot each station (dots, so gaps show as real gaps) ----
codes = sorted(daily["station_code"].unique())
fig, axes = plt.subplots(len(codes), 1, figsize=(12, 3.5 * len(codes)), squeeze=False)
for ax, code in zip(axes[:, 0], codes):
    g = daily[daily["station_code"] == code].sort_values("date")
    ax.plot(g["date"], g["level"], ".", markersize=2)
    ax.set_title(code)
    ax.set_ylabel("water level (m)")
fig.tight_layout()
fig.savefig("results/daily_levels.png", dpi=120)
print("saved results/daily_levels.png")

# ---- 2. day-to-day jump statistics ----
for code, g in daily.groupby("station_code"):
    s = g.sort_values("date").set_index("date")["level"]
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    step = full.diff()          # change from the previous day (NaN across gaps)
    jump = step.abs()

    same = step == 0            # level did not change at all since yesterday
    run_id = (same != same.shift()).cumsum()
    longest_flat = int(same.groupby(run_id).sum().max())

    print("\nstation:", code)
    print("  median jump (m):", round(jump.median(), 3))
    print("  99th percentile jump (m):", round(jump.quantile(0.99), 3))
    print("  5 biggest jumps (m):")
    print(jump.nlargest(5).round(2).to_string())
    print("  longest streak of unchanged days:", longest_flat)