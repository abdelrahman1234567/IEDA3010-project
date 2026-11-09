"""One-off conversion of the original .mat files into data_processed/shenzhen.npz.
Each request_gamma_*.mat already contains the precomputed all-pairs shortest
distance matrix 'alldists' (km) and node coordinates 'N_coord'."""
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Project E datasets" / "Shenzhen_PassengerRequestDataset" / "request_gamma_000.mat"
d = sio.loadmat(SRC)
out = {}
for g in ["000", "025", "050", "075", "100"]:
    r = d["request_gamma_" + g][0, 0]
    out["t_" + g] = r["time"].ravel()
    out["o_" + g] = r["orig"].ravel().astype(np.int32)
    out["d_" + g] = r["dest"].ravel().astype(np.int32)
(ROOT / "data_processed").mkdir(exist_ok=True)
np.savez_compressed(ROOT / "data_processed" / "shenzhen.npz",
                    dist=d["alldists"].astype(np.float32), coord=d["N_coord"], **out)
print("saved")
