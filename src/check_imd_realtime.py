from pathlib import Path
import imdlib as imd

out_dir = Path("data/raw/imd_realtime")
out_dir.mkdir(parents=True, exist_ok=True)

try:
    data = imd.get_real_data("rain", "2026-01-01", "2026-06-04", str(out_dir))
except Exception as e:
    print("FAILED:", e)       # the error may list which days are missing
    raise SystemExit

ds = data.get_xarray()
var = list(ds.data_vars)[0]
da = ds[var]
has_data = da.notnull().any(dim=["lat", "lon"]).values
days = ds.time.values

print("variable:", var)
print("days requested:", len(days), "| days with data:", int(has_data.sum()))
print("first day:", str(days[0])[:10], "| last day with data:", str(days[has_data][-1])[:10])
print("min/max:", round(float(da.min()), 1), round(float(da.max()), 1))