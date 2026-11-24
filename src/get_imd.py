from pathlib import Path
import imdlib as imd

IMD_DIR = Path("data/raw/imd")
IMD_DIR.mkdir(parents=True, exist_ok=True)

for year in range(2021, 2027):
    try:
        imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(IMD_DIR))
        print(year, "downloaded")
    except Exception as e:
        print(year, "FAILED:", e)