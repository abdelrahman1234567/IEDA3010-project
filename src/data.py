"""Data loading for the Shenzhen (Futian + Luohu) passenger request dataset.

data_processed/shenzhen.npz is produced by src/extract_mat.py from the original
.mat files (unchanged in 'Project E datasets/'). It holds:
  dist   : 1858 x 1858 all-pairs shortest road distance (km)
  coord  : 1858 x 2 planar node coordinates (m)
  t_xxx, o_xxx, d_xxx : request time (h), origin node, destination node (1-based)
                        for gamma = 000, 025, 050, 075, 100
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data_processed" / "shenzhen.npz"

_cache = {}


def load_network():
    if "net" not in _cache:
        z = np.load(NPZ)
        _cache["net"] = (z["dist"].astype(np.float32), z["coord"])
        _cache["z"] = z
    return _cache["net"]


def load_requests(gamma="050", sample_frac=0.02, seed=3010, t_max_min=180.0):
    """Return a reproducible random sample of requests.

    Output dict of numpy arrays: rid, t (minutes), o, d (0-based nodes).
    Same seed -> same sampled request indices for every gamma, so scenarios
    differ only in destinations (origins are identical across gamma files).
    """
    load_network()
    z = _cache["z"]
    t = z["t_" + gamma] * 60.0
    o = z["o_" + gamma] - 1
    d = z["d_" + gamma] - 1
    rng = np.random.default_rng(seed)
    keep = rng.random(len(t)) < sample_frac
    keep &= (o != d) & (t <= t_max_min)
    idx = np.flatnonzero(keep)
    return {"rid": np.arange(len(idx)), "t": t[idx], "o": o[idx].astype(int),
            "d": d[idx].astype(int)}


def make_zones(n_zones=16, seed=3010, iters=50):
    """Partition road nodes into zones with k-means on planar coordinates.
    Returns zone label per node and a 'centre' node per zone (node nearest the
    centroid), used as the repositioning target."""
    dist, coord = load_network()
    X = (coord - coord.mean(0)) / 1000.0
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), n_zones, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k]
                         for k in range(n_zones)])
        if np.allclose(newC, C):
            break
        C = newC
    centre = np.array([np.flatnonzero(lab == k)[np.argmin(((X[lab == k] - C[k]) ** 2).sum(1))]
                       for k in range(n_zones)])
    return lab, centre
