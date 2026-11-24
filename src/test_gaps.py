from data import load_all_stations, to_daily, gap_lengths

readings, stations = load_all_stations("data/raw")
daily = to_daily(readings)
print("daily table shape:", daily.shape)
print(daily.head(3))
print()

for code, g in daily.groupby("station_code"):
    gaps = gap_lengths(g)
    print("station:", code)
    print("  days with data:", len(g))
    print("  number of gaps:", len(gaps))
    print("  longest gap (days):", max(gaps) if gaps else 0)
    print("  gaps of 1-3 days:", sum(x <= 3 for x in gaps))
    print("  gaps of 4-14 days:", sum(4 <= x <= 14 for x in gaps))
    print("  gaps over 14 days:", sum(x > 14 for x in gaps))
    print("  readings per day:", g["n_readings"].value_counts().sort_index().to_dict())
    print()