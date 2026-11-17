"""Convert the Shenzhen MATLAB dataset into a compressed NumPy archive."""

from pathlib import Path

import numpy as np
from scipy import io


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "Project E datasets"
    / "Shenzhen_PassengerRequestDataset"
    / "request_gamma_000.mat"
)
DESTINATION = ROOT / "data_processed" / "shenzhen.npz"
GAMMAS = ("000", "025", "050", "075", "100")


def extract(source=SOURCE, destination=DESTINATION):
    """Extract all five scenarios stored in the source MATLAB file."""
    matlab_data = io.loadmat(source)
    requests = {}

    for gamma in GAMMAS:
        scenario = matlab_data[f"request_gamma_{gamma}"][0, 0]
        requests[f"t_{gamma}"] = scenario["time"].ravel()
        requests[f"o_{gamma}"] = scenario["orig"].ravel().astype(np.int32)
        requests[f"d_{gamma}"] = scenario["dest"].ravel().astype(np.int32)

    destination = Path(destination)
    destination.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        destination,
        dist=matlab_data["alldists"].astype(np.float32),
        coord=matlab_data["N_coord"],
        **requests,
    )


def main():
    extract()
    print("saved")


if __name__ == "__main__":
    main()
