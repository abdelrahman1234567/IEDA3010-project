"""Load Shenzhen road-network data, passenger requests, and spatial zones."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data_processed" / "shenzhen.npz"
_cache = {}


def load_network():
    """Return the distance matrix (km) and planar node coordinates (m)."""
    if "net" not in _cache:
        dataset = np.load(NPZ)
        _cache["net"] = (dataset["dist"].astype(np.float32), dataset["coord"])
        _cache["z"] = dataset

    return _cache["net"]


def load_requests(gamma="050", sample_frac=0.02, seed=3010, t_max_min=180.0):
    """Sample requests reproducibly; return rid, t (minutes), o, and d (zero-based)."""
    load_network()
    dataset = _cache["z"]

    request_times = dataset[f"t_{gamma}"] * 60.0
    origins = dataset[f"o_{gamma}"] - 1
    destinations = dataset[f"d_{gamma}"] - 1

    rng = np.random.default_rng(seed)
    selected = rng.random(len(request_times)) < sample_frac
    selected &= (origins != destinations) & (request_times <= t_max_min)
    indices = np.flatnonzero(selected)

    # The random draw is shared across gamma scenarios; same-node trips are filtered afterward.
    return {
        "rid": np.arange(len(indices)),
        "t": request_times[indices],
        "o": origins[indices].astype(int),
        "d": destinations[indices].astype(int),
    }


def make_zones(n_zones=16, seed=3010, iters=50):
    """Cluster road nodes and choose each zone's nearest node to its centroid."""
    _, coordinates = load_network()
    scaled_coordinates = (coordinates - coordinates.mean(axis=0)) / 1000.0

    rng = np.random.default_rng(seed)
    centroids = scaled_coordinates[
        rng.choice(len(scaled_coordinates), n_zones, replace=False)
    ]

    for _ in range(iters):
        squared_distances = (
            (scaled_coordinates[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(axis=-1)
        zone_labels = np.argmin(squared_distances, axis=1)
        updated_centroids = np.array(
            [
                scaled_coordinates[zone_labels == zone].mean(axis=0)
                if np.any(zone_labels == zone)
                else centroids[zone]
                for zone in range(n_zones)
            ]
        )
        if np.allclose(updated_centroids, centroids):
            break
        centroids = updated_centroids

    centre_nodes = []
    for zone in range(n_zones):
        nodes = np.flatnonzero(zone_labels == zone)
        offsets = scaled_coordinates[nodes] - centroids[zone]
        closest_node = nodes[np.argmin((offsets**2).sum(axis=1))]
        centre_nodes.append(closest_node)

    return zone_labels, np.array(centre_nodes)
