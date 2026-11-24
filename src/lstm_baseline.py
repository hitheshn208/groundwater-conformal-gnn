import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import SPLIT
from windows import build_windows, split_masks, HORIZONS
from evaluate import evaluate

SEED, T = 0, 30
HIDDEN, LR, BATCH = 64, 1e-3, 256
MAX_EPOCHS, PATIENCE = 60, 7

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
Path("results").mkdir(exist_ok=True)

# ---------------- data ----------------
panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
scalers = json.load(open("data/clean/scalers.json"))
X, Y, R, TD, ST = build_windows(panel, scalers["features"], T=T)
masks = split_masks(R, TD, SPLIT)
print("samples:", len(X), "| shape of one sample:", X.shape[1:])
for s in masks:
    print(f"  {s}: targets at 7d = {int(masks[s][:, 0].sum())}, 30d = {int(masks[s][:, 1].sum())}")

Xt = torch.from_numpy(X.astype(np.float32)).to(device)
Yt = torch.from_numpy(np.nan_to_num(Y).astype(np.float32)).to(device)   # NaN -> 0, mask ignores
Mtr = torch.from_numpy(masks["train"]).to(device)
Mva = torch.from_numpy(masks["val"]).to(device)
train_rows = torch.from_numpy(np.where(masks["train"].any(axis=1))[0]).to(device)
val_rows = torch.from_numpy(np.where(masks["val"].any(axis=1))[0]).to(device)


# ---------------- model ----------------
class SharedLSTM(nn.Module):
    """One LSTM shared by all stations. Output = today's level + a learned change."""
    def __init__(self, n_features, hidden=64, n_out=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        today = x[:, -1, 0:1]                       # today's scaled level (kept for the output)
        rel = x[:, :, 0] - today                    # level relative to today: no absolute depth
        rel = rel * (1 - x[:, :, 6])                # days without a real reading -> 0
        x_in = torch.cat([rel.unsqueeze(-1), x[:, :, 1:]], dim=2)
        out, _ = self.lstm(x_in)
        return today + self.head(out[:, -1])          # feature 0 = today's scaled level


def masked_mse(pred, y, mask):
    mask = mask.float()
    return (((pred - y) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


model = SharedLSTM(X.shape[2], HIDDEN, len(HORIZONS)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
print("parameters:", sum(p.numel() for p in model.parameters()))

# ---------------- training ----------------
best, bad = float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_rows), device=device)
    total, count = 0.0, 0
    for s in range(0, len(perm), BATCH):
        b = train_rows[perm[s:s + BATCH]]
        opt.zero_grad()
        loss = masked_mse(model(Xt[b]), Yt[b], Mtr[b])
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
        count += len(b)

    model.eval()
    with torch.no_grad():
        val = masked_mse(model(Xt[val_rows]), Yt[val_rows], Mva[val_rows]).item()
    print(f"epoch {epoch:2d} | train loss {total / count:.4f} | val loss {val:.4f}")

    if val < best - 1e-5:                         # improved -> save this version
        best, bad = val, 0
        torch.save(model.state_dict(), "results/lstm_best.pt")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping")
            break

# ---------------- evaluation (best checkpoint) ----------------
model.load_state_dict(torch.load("results/lstm_best.pt", map_location=device))
model.eval()
with torch.no_grad():
    pred = torch.cat([model(Xt[i:i + 2048]) for i in range(0, len(Xt), 2048)]).cpu().numpy()

table, per = evaluate(pred, X, Y, masks, ST, scalers, name="lstm")
pd.set_option("display.width", 200)
print()
print(table.round(3).to_string(index=False))
table.to_csv("results/lstm_metrics.csv", index=False)

for h in HORIZONS:
    p = per[h]
    better = int((p["skill"] > 0).sum())
    print(f"\nh={h} test, per station: median RMSE lstm {p['RMSE_lstm'].median():.3f} m "
        f"vs persistence {p['RMSE_persist'].median():.3f} m | "
        f"LSTM better at {better} of {len(p)} stations")
    print("  best skill: ", p["skill"].nlargest(3).round(2).to_dict())
    print("  worst skill:", p["skill"].nsmallest(3).round(2).to_dict())
    p.to_csv(f"results/lstm_by_station_h{h}.csv")