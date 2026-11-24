import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import SPLIT
from windows import split_masks, HORIZONS
from evaluate import to_metres
from graph import build_adjacency
from graph_data import build_dense, build_targets, flatten

SEED, T = 0, 30
HIDDEN, LR, BATCH = 64, 1e-3, 16
MAX_EPOCHS, PATIENCE, K_NEIGHBOURS = 60, 7, 6
QUANTILES = (0.05, 0.5, 0.95)                 # lower end, median, upper end of a 90% interval
Q = len(QUANTILES)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
Path("results").mkdir(exist_ok=True)

# ---------------- data and graph (same as gcn_gru.py) ----------------
panel = pd.read_csv("data/clean/panel.csv", dtype={"station_code": str}, parse_dates=["date"])
scalers = json.load(open("data/clean/scalers.json"))
features = scalers["features"]
stations, dates, F_arr, L_arr, real, ok = build_dense(panel, features, T=T)
origins, Y, R, TD = build_targets(dates, L_arr, real, ok, T=T)
masks = split_masks(R, TD, SPLIT)
meta = pd.read_csv("data/clean/stations.csv", dtype={"station_code": str}).set_index("station_code").loc[stations]
A, sigma = build_adjacency(meta["lat"].to_numpy(), meta["lon"].to_numpy(), k=K_NEIGHBOURS)

Ft = torch.from_numpy(F_arr).to(device)
origin_t = torch.from_numpy(origins).to(device)
Yt = torch.from_numpy(np.nan_to_num(Y)).to(device)                 # (n, H, S)
Mtr = torch.from_numpy(masks["train"]).to(device)
Mva = torch.from_numpy(masks["val"]).to(device)
train_rows = torch.from_numpy(np.where(masks["train"].any(axis=(1, 2)))[0]).to(device)
val_rows = torch.from_numpy(np.where(masks["val"].any(axis=(1, 2)))[0]).to(device)
offsets = torch.arange(-T + 1, 1, device=device)
q_t = torch.tensor(QUANTILES, device=device).view(1, 1, Q, 1)


def get_x(rows):
    idx = origin_t[rows][:, None] + offsets[None, :]
    return Ft[idx]


# ---------------- model: outputs Q quantiles for each horizon ----------------
class GCNGRUQuantile(nn.Module):
    def __init__(self, A, n_features, hidden=64, n_out=2 * Q):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))
        self.gcn = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                                   # x: (B, T, S, F)
        B, T_, S, _ = x.shape
        today = x[:, -1, :, 0]
        rel = (x[..., 0] - today[:, None, :]) * (1 - x[..., 6])
        x = torch.cat([rel.unsqueeze(-1), x[..., 1:]], dim=-1)
        mixed = torch.einsum("ij,btjf->btif", self.A, x)
        h = torch.relu(self.gcn(mixed)).permute(0, 2, 1, 3).reshape(B * S, T_, -1)
        out, _ = self.gru(h)
        change = self.head(out[:, -1]).view(B, S, len(HORIZONS), Q)      # (B, S, H, Q)
        pred = today[:, :, None, None] + change
        return pred.permute(0, 2, 3, 1)                                   # (B, H, Q, S)


def pinball(pred, y, mask):
    """Pinball loss: errors on the 'wrong side' of a quantile cost more. pred (B,H,Q,S), y (B,H,S)."""
    e = y.unsqueeze(2) - pred
    loss = torch.maximum(q_t * e, (q_t - 1) * e)
    m = mask.unsqueeze(2).float().expand_as(loss)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


model = GCNGRUQuantile(A, len(features)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- training ----------------
best, bad = float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_rows), device=device)
    total, count = 0.0, 0
    for s in range(0, len(perm), BATCH):
        b = train_rows[perm[s:s + BATCH]]
        opt.zero_grad()
        loss = pinball(model(get_x(b)), Yt[b], Mtr[b])
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
        count += len(b)
    model.eval()
    with torch.no_grad():
        val = pinball(model(get_x(val_rows)), Yt[val_rows], Mva[val_rows]).item()
    print(f"epoch {epoch:2d} | train loss {total / count:.4f} | val loss {val:.4f}")
    if val < best - 1e-5:
        best, bad = val, 0
        torch.save(model.state_dict(), "results/gcn_quantile_best.pt")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping")
            break

# ---------------- evaluation: raw quantile intervals on the test set ----------------
model.load_state_dict(torch.load("results/gcn_quantile_best.pt", map_location=device))
model.eval()
all_rows = torch.arange(len(origins), device=device)
with torch.no_grad():
    pred = torch.cat([model(get_x(all_rows[i:i + 64])) for i in range(0, len(all_rows), 64)]).cpu().numpy()
pred = np.sort(pred, axis=2)               # fix any "crossing" so lower <= median <= upper

fl = flatten(origins, Y, masks, ok, L_arr, stations, pred.reshape(len(origins), -1, pred.shape[-1]))
ST = fl["ST"]
rows = []
for hi, h in enumerate(HORIZONS):
    p = fl["pred"].reshape(len(ST), len(HORIZONS), Q)          # (N, H, Q) in scaled units
    true_m = to_metres(fl["Y"][:, hi], ST, scalers)
    lo, med, hi_ = (to_metres(p[:, hi, k], ST, scalers) for k in range(Q))
    for split in ["calib", "test"]:
        m = fl["masks"][split][:, hi]
        inside = (true_m[m] >= lo[m]) & (true_m[m] <= hi_[m])
        st_cov = pd.Series(inside).groupby(ST[m]).mean()
        rows.append({"horizon": h, "split": split, "n": int(m.sum()), "coverage": inside.mean(),
                     "mean_width_m": (hi_[m] - lo[m]).mean(), "station_cov_min": st_cov.min(),
                     "station_cov_median": st_cov.median(),
                     "RMSE_median_forecast": np.sqrt(((med[m] - true_m[m]) ** 2).mean())})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nRaw quantile regression, target coverage 90%\n")
print(out.round(3).to_string(index=False))
out.to_csv("results/quantile_summary.csv", index=False)