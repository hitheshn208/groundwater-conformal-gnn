from pathlib import Path
import pandas as pd
from data import load_station_metadata, load_station_readings

path = sorted(Path("data/raw").glob("*.xlsx"))[0]
print("FILE:", path.name)
print("sheets:", pd.ExcelFile(path).sheet_names)

meta = load_station_metadata(path)
for k in ["Station Code", "Station Name", "District", "Latitude",
        "Longitude", "Well Depth", "Aquifer Type", "Unit"]:
    print(f"  {k}: {meta.get(k)}")

df = load_station_readings(path)
print("\nshape:", df.shape)
print(df.head(3))
print(df.tail(3))
print("\ndate range:", df["Data Time"].min(), "to", df["Data Time"].max())
print("value min/max:", df["Data Value"].min(), df["Data Value"].max())

per_day = df.groupby(df["Data Time"].dt.date).size()
print("\nreadings per day (count of days):")
print(per_day.value_counts().sort_index())