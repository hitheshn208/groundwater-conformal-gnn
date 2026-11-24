import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import SPLIT
from windows import split_masks, HORIZONS
from evaluate import evaluate
from graph import build_adjacency
from graph_data import build_dense, build_targets, flatten

# "graph" = stations share information with neighbours; "nograph" = same model, graph switched off
MODE = sys.argv[1] if len(sys.argv) > 1 else "graph"
assert MODE in ("graph", "nograph"), "use: python src/gcn_gru.py graph|nograph"
NAME = "gcn" if MODE == "graph" else "nograph"
OPTS = sys.argv[2:]     
SEED, T = 0, 30
for o in OPTS:
    if o.startswith("seed="):
        SEED = int(o[5:])
        NAME += f"_s{SEED}"
HIDDEN, LR, BATCH = 64, 1e-3, 16          # BATCH = days per step (every day holds all stations)
MAX_EPOCHS, PATIENCE, K_NEIGHBOURS = 60, 7, 6

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("mode:", MODE, "| device:", device)
Path("results").mkdir(exist_ok=True)

# ---------------- data ----------------
panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
scalers = json.load(open("data/clean/scalers.json"))
features = scalers["features"]
stations, dates, F_arr, L_arr, real, ok = build_dense(panel, features, T=T)
origins, Y, R, TD = build_targets(dates, L_arr, real, ok, T=T)
masks = split_masks(R, TD, SPLIT)
if "norain" in OPTS:                      # ablation: switch the 3 rain features off
    F_arr[..., 1:4] = 0.0
    NAME += "_norain"
LOW = []
if "lowdata" in OPTS:                     # low-data test: 5 stations keep only ~6 months of training
    LOW = stations[::7]
    for s_i in range(0, len(stations), 7):
        first = int(np.argmax(real[:, s_i]))                       # first real reading of this station
        masks["train"][:, :, s_i] &= TD[:, :, s_i] < dates[first + 183]
    NAME += "_lowdata"
    print("low-data stations:", LOW)
    Path("results/low_stations.txt").write_text("\n".join(LOW))
print("days:", len(dates), "| stations:", len(stations), "| features:", len(features))
for s in masks:   # these counts must equal the LSTM run (e.g. test 7727 / 7499)
    print(f"  {s}: targets at 7d = {int(masks[s][:, 0].sum())}, 30d = {int(masks[s][:, 1].sum())}")

# ---------------- graph ----------------
meta = pd.read_csv("data/clean/stations.csv", dtype={"station_code": str}).set_index("station_code").loc[stations]
if MODE == "graph":
    A, sigma = build_adjacency(meta["lat"].to_numpy(), meta["lon"].to_numpy(), k=K_NEIGHBOURS)
    print(f"graph: k={K_NEIGHBOURS}, sigma = {sigma:.2f} km, edges = {int(((A > 0).sum() - len(A)) / 2)}")
else:
    A = np.eye(len(stations), dtype=np.float32)        # identity = every station only sees itself

# ---------------- tensors ----------------
Ft = torch.from_numpy(F_arr).to(device)                                   # (D, S, F)
origin_t = torch.from_numpy(origins).to(device)                          # origin day index of each sample
Yt = torch.from_numpy(np.nan_to_num(Y)).to(device)                       # (n, H, S)
Mtr = torch.from_numpy(masks["train"]).to(device)
Mva = torch.from_numpy(masks["val"]).to(device)
train_rows = torch.from_numpy(np.where(masks["train"].any(axis=(1, 2)))[0]).to(device)
val_rows = torch.from_numpy(np.where(masks["val"].any(axis=(1, 2)))[0]).to(device)
offsets = torch.arange(-T + 1, 1, device=device)


def get_x(rows):
    """Window of the last T days for ALL stations: (B, T, S, F)."""
    idx = origin_t[rows][:, None] + offsets[None, :]
    return Ft[idx]


# ---------------- model ----------------
class GCNGRU(nn.Module):
    """Each day: stations mix with neighbours (graph step), then a GRU reads the T days.
    Output = today's level + learned change, like the LSTM baseline."""
    def __init__(self, A, n_features, hidden=64, n_out=2):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))
        self.gcn = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                                   # x: (B, T, S, F)
        B, T_, S, _ = x.shape
        today = x[:, -1, :, 0]                              # (B, S) today's scaled level
        rel = (x[..., 0] - today[:, None, :]) * (1 - x[..., 6])   # level relative to today
        x = torch.cat([rel.unsqueeze(-1), x[..., 1:]], dim=-1)
        mixed = torch.einsum("ij,btjf->btif", self.A, x)    # graph step: average over neighbours
        h = torch.relu(self.gcn(mixed))                     # (B, T, S, hidden)
        h = h.permute(0, 2, 1, 3).reshape(B * S, T_, -1)    # one sequence per (day, station)
        out, _ = self.gru(h)
        change = self.head(out[:, -1]).view(B, S, -1)       # (B, S, n_out)
        return (today.unsqueeze(-1) + change).permute(0, 2, 1)   # (B, H, S)


def masked_mse(pred, y, mask):
    mask = mask.float()
    return (((pred - y) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


model = GCNGRU(A, len(features), HIDDEN, len(HORIZONS)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
print("parameters:", sum(p.numel() for p in model.parameters()))

# ---------------- training ----------------
best, bad = float("inf"), 0
ckpt = f"results/{NAME}_best.pt"
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_rows), device=device)
    total, count = 0.0, 0
    for s in range(0, len(perm), BATCH):
        b = train_rows[perm[s:s + BATCH]]
        opt.zero_grad()
        loss = masked_mse(model(get_x(b)), Yt[b], Mtr[b])
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
        count += len(b)

    model.eval()
    with torch.no_grad():
        val = masked_mse(model(get_x(val_rows)), Yt[val_rows], Mva[val_rows]).item()
    print(f"epoch {epoch:2d} | train loss {total / count:.4f} | val loss {val:.4f}")

    if val < best - 1e-5:
        best, bad = val, 0
        torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping")
            break

# ---------------- evaluation (best checkpoint) ----------------
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
all_rows = torch.arange(len(origins), device=device)
with torch.no_grad():
    pred = torch.cat([model(get_x(all_rows[i:i + 64])) for i in range(0, len(all_rows), 64)]).cpu().numpy()

fl = flatten(origins, Y, masks, ok, L_arr, stations, pred)
# evaluate() only needs today's level from X (as persistence), so we pass it as a (N,1,1) array
table, per = evaluate(fl["pred"], fl["today"].reshape(-1, 1, 1), fl["Y"], fl["masks"], fl["ST"], scalers, name=NAME)
pd.set_option("display.width", 200)
print()
print(table.round(3).to_string(index=False))
table.to_csv(f"results/{NAME}_metrics.csv", index=False)

for h in HORIZONS:
    p = per[h]
    print(f"\nh={h} test, per station: median RMSE {NAME} {p[f'RMSE_{NAME}'].median():.3f} m "
            f"vs persistence {p['RMSE_persist'].median():.3f} m | "
            f"better at {int((p['skill'] > 0).sum())} of {len(p)} stations")
    print("  best skill: ", p["skill"].nlargest(3).round(2).to_dict())
    print("  worst skill:", p["skill"].nsmallest(3).round(2).to_dict())
    p.to_csv(f"results/{NAME}_by_station_h{h}.csv")

# saved so the conformal step (Phase F) can reuse the predictions without retraining
np.savez(f"results/{NAME}_pred.npz", pred=fl["pred"], Y=fl["Y"], ST=fl["ST"], today=fl["today"],
         **{f"mask_{k}": v for k, v in fl["masks"].items()})
print(f"\nsaved results/{NAME}_pred.npz")