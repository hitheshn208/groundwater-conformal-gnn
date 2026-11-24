from pathlib import Path

RAW_DIR = Path("data/raw")

# 1. what files do we have?
files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
print("number of files:", len(files))
print("file types:", sorted({p.suffix for p in files}))
print()

# 2. show the first 15 raw lines of the first file, exactly as stored
first = files[0]
print("FILE:", first.name)
print("-" * 40)
with open(first, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= 15:
            break
        print(i, repr(line))