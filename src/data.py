import pandas as pd
from pathlib import Path

def load_station_metadata(path):
    """Read sheet 0 (metadata). Column A = name, column B = value."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    meta = {}
    for key, value in zip(raw.iloc[:, 0], raw.iloc[:, 1]):
        if pd.isna(key) or pd.isna(value):
            continue
        clean_key = " ".join(str(key).split())  # fixes double spaces
        meta[clean_key] = value
    return meta

def load_station_readings(path):
    """Read sheet 1 (readings). Returns Data Time + Data Value only."""
    df = pd.read_excel(path, sheet_name=1, header=6)  # row 7 holds the names
    df["Data Time"] = pd.to_datetime(df["Data Time"], errors="coerce")
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["Data Time", "Data Value"])  # drops disclaimer rows
    return df[["Data Time", "Data Value"]].sort_values("Data Time")

def load_all_stations(raw_dir):
    """Loop over every station file. Returns (readings, stations)."""
    readings_list, meta_list = [], []
    for path in sorted(Path(raw_dir).glob("*.xlsx")):
        if path.name.startswith("~$"):  # Excel temporary lock file, skip
            continue
        meta = load_station_metadata(path)
        df = load_station_readings(path)
        df["station_code"] = meta["Station Code"]
        readings_list.append(df)
        meta_list.append({
            "station_code": meta["Station Code"],
            "station_name": meta.get("Station Name"),
            "district": meta.get("District"),
            "lat": float(meta["Latitude"]),
            "lon": float(meta["Longitude"]),
            "well_depth": meta.get("Well Depth"),
            "aquifer": meta.get("Aquifer Type"),
            "file": path.name,
        })
    readings = pd.concat(readings_list, ignore_index=True)
    stations = pd.DataFrame(meta_list)

    # same station downloaded twice -> keep one copy
    stations = stations.drop_duplicates("station_code")
    readings = readings.drop_duplicates(["station_code", "Data Time"])
    return readings, stations

def summarize_stations(readings):
    """One row per station: date range and missing-day counts."""
    readings = readings.copy()
    readings["date"] = readings["Data Time"].dt.normalize()  # drop the hour
    rows = []
    for code, g in readings.groupby("station_code"):
        first, last = g["date"].min(), g["date"].max()
        span = (last - first).days + 1          # days from first to last
        have = g["date"].nunique()              # days that have any reading
        rows.append({
            "station_code": code,
            "first_day": first.date(),
            "last_day": last.date(),
            "span_days": span,
            "days_with_data": have,
            "missing_days": span - have,
            "missing_pct": round(100 * (span - have) / span, 1),
        })
    return pd.DataFrame(rows)

def to_daily(readings):
    """Average the readings of each day -> one value per station per day."""
    df = readings.copy()
    df["date"] = df["Data Time"].dt.normalize()
    daily = (
        df.groupby(["station_code", "date"])["Data Value"]
        .agg(level="mean", n_readings="count")  # also keep how many readings
        .reset_index()
    )
    return daily


def gap_lengths(daily_one):
    """Lengths of every run of missing days for ONE station."""
    full = pd.date_range(daily_one["date"].min(), daily_one["date"].max(), freq="D")
    present = pd.Series(1, index=daily_one["date"]).reindex(full).fillna(0)
    missing = present == 0
    # a new run id starts each time missing/not-missing flips
    run_id = (missing != missing.shift()).cumsum()
    runs = missing.groupby(run_id).agg(["first", "size"])
    return runs.loc[runs["first"], "size"].tolist()  # keep only missing runs

def flag_spikes(level, window=7, k=5, min_limit=0.3, scale_window=61, centered=True):
    """True = spike: far from the median of its neighbours.
    The noise scale is measured locally (+-30 days), so noisy stretches get a
    looser limit than quiet ones."""
    if centered:
        baseline = level.rolling(window, center=True, min_periods=4).median()
    else:
        baseline = level.shift(1).rolling(window, min_periods=3).median()
    resid = level - baseline
    local = resid.abs().rolling(scale_window, center=True, min_periods=15).median()
    global_limit = max(k * 1.4826 * resid.abs().median(), min_limit)
    limit = (k * 1.4826 * local).clip(lower=min_limit).fillna(global_limit)
    return resid.abs() > limit

def clean_station(daily_one, k=5, max_gap=3, valid_until=None, mask_ranges=()):
    """Clean ONE station. Returns a full daily calendar with level (cleaned),
    missing (1 = no trusted real reading) and one column per removal rule."""
    s = daily_one.set_index("date")["level"].sort_index()
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))

    # 0. manual decisions: a cutoff date and date ranges we remove by hand
    after_cutoff = pd.Series(False, index=full.index)
    if valid_until is not None:
        after_cutoff = pd.Series(full.index > pd.Timestamp(valid_until), index=full.index)
    for start, end in mask_ranges:
        after_cutoff |= (full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))
    work = full.mask(after_cutoff)

    # 1. impossible values (above ground)
    bad_range = flag_out_of_range(work)
    work = work.mask(bad_range)

    # 2. stuck sensor / floor plateau
    bad_flat = flag_flat(work)
    work = work.mask(bad_flat)

    # 3. spikes (judged on the already-cleaned series)
    spikes = flag_spikes(work, k=k)
    cleaned = work.mask(spikes)
    observed = cleaned.notna()                    # real, trusted readings

    # 4. interpolate only gaps of max_gap days or shorter
    run_id = (observed != observed.shift()).cumsum()
    run_len = (~observed).groupby(run_id).transform("sum")
    short_gap = (~observed) & (run_len <= max_gap)
    interp = cleaned.interpolate(method="linear", limit_area="inside")
    level = cleaned.where(~short_gap, interp)

    out = pd.DataFrame({
        "level": level,
        "missing": (~observed).astype(int),
        "after_cutoff": (after_cutoff & full.notna()).astype(int),
        "range_bad": bad_range.astype(int),
        "flat": bad_flat.astype(int),
        "spike": spikes.astype(int),
    })
    out.index.name = "date"
    return out

def flag_out_of_range(level):
    """Water level above ground (above 0) is treated as invalid."""
    return level > 0


def flag_flat(level, window=10, tol=0.01):
    """True for days inside a stretch where the level barely moves
    (stuck sensor, or water below the sensor = a floor)."""
    roll = level.rolling(window, min_periods=int(window * 0.8))
    still = (roll.max() - roll.min()) <= tol   # the window ENDING today is still
    # a still window ending on day t makes days t-9..t flat, so look forward
    flat = still[::-1].astype(int).rolling(window, min_periods=1).max()[::-1] > 0
    return flat & level.notna()