import pandas as pd
from data import load_all_stations, summarize_stations
from pathlib import Path

pd.set_option("display.width", 200)

readings, stations = load_all_stations("data/raw")
print("stations:", len(stations))
print("total readings:", len(readings))
print()
print(stations[["station_code", "station_name", "district", "lat", "lon", "file"]])
print()
print(summarize_stations(readings))

# save the summary for your report
Path("results").mkdir(exist_ok=True)
summary = summarize_stations(readings)
summary.to_csv("results/station_summary.csv", index=False)

# same place under two different codes? (central vs state duplicates)
loc = stations.assign(lat3=stations["lat"].round(3), lon3=stations["lon"].round(3))
dups = loc[loc.duplicated(["lat3", "lon3"], keep=False)].sort_values(["lat3", "lon3"])
print("\nstations sharing the same location (rounded to about 100 m):")
print(dups[["station_code", "station_name", "lat", "lon"]] if len(dups) else "none")