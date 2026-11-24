import pandas as pd
import imdlib as imd

stations = pd.read_csv("data/clean/stations.csv", dtype={"station_code": str})


def extract(ds, stations):
    """Daily rain at the grid cell nearest to each station."""
    var = list(ds.data_vars)[0]
    rows = []
    for st in stations.itertuples():
        cell = ds[var].sel(lat=st.lat, lon=st.lon, method="nearest")
        rows.append(pd.DataFrame({
            "station_code": st.station_code,
            "date": pd.to_datetime(cell["time"].values),
            "rain": cell.values,
            "cell_lat": float(cell["lat"]),
            "cell_lon": float(cell["lon"]),
        }))
    return pd.concat(rows, ignore_index=True)


tables = []
for year in range(2021, 2026):                      # final yearly archive
    ds = imd.open_data("rain", year, year, "yearwise", "data/raw/imd").get_xarray()
    tables.append(extract(ds, stations))

# 2026 comes from the provisional real-time files
ds = imd.open_real_data("rain", "2026-01-01", "2026-06-04",
                        "data/raw/imd_realtime").get_xarray()
tables.append(extract(ds, stations))

rain = pd.concat(tables, ignore_index=True)
rain.to_csv("data/clean/rain_daily.csv", index=False)

# sanity checks
cells = rain[["cell_lat", "cell_lon"]].drop_duplicates()
print("stations:", rain["station_code"].nunique(), "| unique rain cells:", len(cells))
print("rows:", len(rain), "| NaN rain days:", int(rain["rain"].isna().sum()))
print("date range:", rain["date"].min().date(), "to", rain["date"].max().date())
rain["year"] = rain["date"].dt.year
print("\nmean total rain per station by year (mm):")
print(rain.groupby(["station_code", "year"])["rain"].sum()
    .groupby("year").mean().round(0))