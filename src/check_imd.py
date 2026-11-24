import imdlib as imd

for year in range(2021, 2027):
    try:
        data = imd.open_data("rain", year, year, "yearwise", "data/raw/imd")
    except Exception as e:
        print(year, "cannot open:", e)
        continue

    ds = data.get_xarray()
    var = list(ds.data_vars)[0]              # name of the rainfall variable
    da = ds[var]
    has_data = da.notnull().any(dim=["lat", "lon"]).values   # True = day has values
    days = ds.time.values

    print(year, "| variable:", var,
        "| days in file:", len(days),
        "| days with data:", int(has_data.sum()),
        "| last day with data:", str(days[has_data][-1])[:10],
        "| min/max:", round(float(da.min()), 1), round(float(da.max()), 1))