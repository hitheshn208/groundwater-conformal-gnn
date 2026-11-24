import numpy as np


def haversine_km(lat, lon):
    """Distance in km between every pair of stations (they are on a sphere, not a flat map)."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_adjacency(lat, lon, k=6):
    """k nearest neighbours, Gaussian weights, symmetric normalisation with self-loops."""
    d = haversine_km(np.asarray(lat, float), np.asarray(lon, float))
    S = len(d)
    k = min(k, S - 1)
    d_no_self = d.copy()
    np.fill_diagonal(d_no_self, np.inf)                 # a station is not its own neighbour
    nbr = np.argsort(d_no_self, axis=1)[:, :k]          # (S, k) indices of nearest stations
    nbr_d = np.take_along_axis(d_no_self, nbr, axis=1)
    sigma = np.median(nbr_d)                            # typical neighbour distance
    if sigma <= 0:
        sigma = 1.0
    W = np.zeros((S, S))
    W[np.repeat(np.arange(S), k), nbr.ravel()] = np.exp(-(nbr_d.ravel() ** 2) / (2 * sigma ** 2))
    W = np.maximum(W, W.T)                              # if A likes B, B likes A
    W = W + np.eye(S)                                   # self-loop: a station keeps its own signal
    dinv = 1.0 / np.sqrt(W.sum(axis=1))
    return (W * dinv[:, None] * dinv[None, :]).astype(np.float32), float(sigma)