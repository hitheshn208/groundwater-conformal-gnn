import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

codes = sys.argv[1:]
if not codes:
    raise SystemExit("usage: python src/plot_clean.py CODE [CODE ...]")

dtype = {"station_code": str}
raw = pd.read_csv("data/clean/daily_raw.csv", dtype=dtype, parse_dates=["date"])
flags = pd.read_csv("data/clean/daily_clean.csv", dtype=dtype, parse_dates=["date"])
df = raw.merge(
    flags[["station_code", "date", "range_bad", "flat", "spike", "after_cutoff"]],
    on=["station_code", "date"], how="left",
)

Path("results").mkdir(exist_ok=True)
fig, axes = plt.subplots(len(codes), 1, figsize=(14, 3.6 * len(codes)), squeeze=False)
for ax, code in zip(axes[:, 0], codes):
    g = df[df["station_code"] == code]
    removed = (g[["range_bad", "flat", "spike", "after_cutoff"]] == 1).any(axis=1)
    ax.plot(g.loc[~removed, "date"], g.loc[~removed, "level"], ".",
            markersize=2, color="tab:blue", label="kept")
    for col, color in [("flat", "orange"), ("spike", "red"),("range_bad", "purple"), ("after_cutoff", "gray")]:
        sel = g[g[col] == 1]
        ax.plot(sel["date"], sel["level"], ".", markersize=4, color=color,
                label=f"{col} ({len(sel)})")
    ax.set_title(code)
    ax.legend(loc="best", fontsize=8)

fig.tight_layout()
out = f"results/clean_{codes[0]}.png"
fig.savefig(out, dpi=110)
print("saved", out)