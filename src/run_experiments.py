"""Run resumable experiment grids E1-E5 and append results to CSV files."""

import csv
import itertools
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sim import Params, run


OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [3010, 3011, 3012, 3013, 3014]
GAMMAS = ["000", "025", "050", "075", "100"]
POLICIES = [
    ("fcfs", False),
    ("greedy", False),
    ("milp", False),
    ("milp", True),
    ("stable", False),
]

_CACHE = {}


def grid(name):
    """Yield the original experiment configurations in their original order."""
    if name == "E1":
        for gamma, (policy, reposition), seed in itertools.product(
            GAMMAS, POLICIES, SEEDS
        ):
            yield dict(gamma=gamma, policy=policy, reposition=reposition, seed=seed)

    elif name == "E2":
        for gamma, seed in itertools.product(["000", "050"], SEEDS):
            yield dict(gamma=gamma, policy="milp", seed=seed)
            yield dict(gamma=gamma, policy="stable", seed=seed)
            for rho in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
                yield dict(gamma=gamma, policy="nearstable", rho=rho, seed=seed)

    elif name == "E3":
        for interval, policy, seed in itertools.product(
            [1.0, 2.0, 5.0, 10.0], ["fcfs", "milp"], SEEDS
        ):
            yield dict(gamma="050", policy=policy, dt=interval, seed=seed)

    elif name == "E4":
        policies = [("fcfs", False), ("milp", False), ("milp", True)]
        for fleet, (policy, reposition), seed in itertools.product(
            [150, 200, 250, 300, 350], policies, SEEDS
        ):
            yield dict(
                gamma="000",
                policy=policy,
                reposition=reposition,
                fleet=fleet,
                seed=seed,
            )

    elif name == "E5":
        for theta, seed in itertools.product([1.5, 2.0, 3.0, 4.0, 5.0], SEEDS):
            yield dict(
                gamma="000",
                policy="milp",
                reposition=True,
                theta=theta,
                rmax=max(5.0, theta),
                seed=seed,
            )


def _match(path, config):
    """Check whether a CSV already contains this experiment configuration."""
    if path not in _CACHE:
        _CACHE[path] = pd.read_csv(path, dtype=str)

    rows = _CACHE[path]
    params = Params(**config)
    matching = np.ones(len(rows), dtype=bool)

    for field in config:
        value = getattr(params, field)
        column = rows[field]
        if isinstance(value, bool):
            matching &= column == str(value)
        elif isinstance(value, (int, float)):
            matching &= np.isclose(column.astype(float), float(value))
        else:
            matching &= column.str.zfill(3) == str(value)

    return bool(matching.any())


def main(names, budget_s=1e9):
    """Run grids until the time budget is reached, skipping saved configurations."""
    started_at = time.time()

    for name in names:
        path = OUT / f"{name}.csv"
        output_file = None
        writer = None
        try:
            for config in grid(name):
                if time.time() - started_at > budget_s:
                    print("BUDGET REACHED")
                    return
                if path.exists() and _match(path, config):
                    continue

                try:
                    result, _ = run(**config)
                except Exception as exc:
                    print("FAILED", config, exc)
                    traceback.print_exc()
                    continue

                if writer is None:
                    is_new = not path.exists()
                    output_file = open(path, "a", newline="")
                    writer = csv.DictWriter(output_file, fieldnames=list(result.keys()))
                    if is_new:
                        writer.writeheader()

                writer.writerow(result)
                output_file.flush()
                _CACHE.pop(path, None)
                print(name, config, "service=%.3f" % result["service_rate"], flush=True)
        finally:
            if output_file is not None:
                output_file.close()


def cli():
    names = sys.argv[1:]
    budget = 1e9
    if names and names[0].startswith("--budget="):
        budget = float(names.pop(0).split("=")[1])
    main(names or ["E1", "E2", "E3", "E4", "E5"], budget)


if __name__ == "__main__":
    cli()
