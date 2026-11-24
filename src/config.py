# Stations we removed after looking at their plots (reason in the comment)
DROP = {
    "CGWBNG0020": "unusable: swings between -73 and +25 m",
    "CGWBNG0033": "smooth drift above ground level (up to +4.5 m)",
    "CGWBNG0009": "only ~13 months trustworthy, then climbs to +60 m",
    "CGWBNG0030": "only ~15 months trustworthy, then plateaus and wild swings",
}

# Keep readings only up to this date (chaotic after it)
VALID_UNTIL = {
    "CGWBNG0018": "2025-06-30",
}

# Date ranges removed by hand (sensor floor plateaus)
MASK_RANGES = {
    "CGWBNG0032": [("2024-01-01", "2024-07-10")],   # flat at about -70.4 m
}

# Time split by TARGET date (Option 1). Test = everything after calib_end.
SPLIT = {"val_start": "2024-05-01",   # last 2 months of training = early-stopping set
        "train_end": "2024-06-30",
        "calib_end": "2025-06-30"}