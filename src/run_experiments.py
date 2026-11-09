"""Experiment grid. Usage: python3 run_experiments.py E1 [E2 ...]
Results are appended to results/<exp>.csv (one row per simulation run)."""
import sys, csv, itertools, traceback
import numpy as np
from pathlib import Path
from sim import run

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [3010, 3011, 3012, 3013, 3014]
GAMMAS = ["000", "025", "050", "075", "100"]
POLICIES = [("fcfs", False), ("greedy", False), ("milp", False), ("milp", True), ("stable", False)]


def grid(name):
    if name == "E1":   # main comparison: policy x OD imbalance x seed
        for g, (pol, rep), s in itertools.product(GAMMAS, POLICIES, SEEDS):
            yield dict(gamma=g, policy=pol, reposition=rep, seed=s)
    elif name == "E2":  # efficiency vs stability frontier
        for g, s in itertools.product(["000", "050"], SEEDS):
            yield dict(gamma=g, policy="milp", seed=s)
            yield dict(gamma=g, policy="stable", seed=s)
            for rho in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                yield dict(gamma=g, policy="nearstable", rho=rho, seed=s)
    elif name == "E3":  # decision interval
        for dt, pol, s in itertools.product([1.0, 2.0, 5.0, 10.0], ["fcfs", "milp"], SEEDS):
            yield dict(gamma="050", policy=pol, dt=dt, seed=s)
    elif name == "E4":  # fleet size
        for f, (pol, rep), s in itertools.product([150, 200, 250, 300, 350],
                                                  [("fcfs", False), ("milp", False), ("milp", True)], SEEDS):
            yield dict(gamma="000", policy=pol, reposition=rep, fleet=f, seed=s)
    elif name == "E5":  # repositioning value parameter
        for th, s in itertools.product([1.5, 2.0, 3.0, 4.0, 5.0], SEEDS):
            yield dict(gamma="000", policy="milp", reposition=True, theta=th, rmax=max(5.0, th), seed=s)


def _key(d):
    return tuple(sorted((k, str(d[k])) for k in d))


def main(names, budget_s=1e9):
    """Resumable: configurations already present in results/<name>.csv are skipped,
    and the run stops cleanly after budget_s seconds."""
    import time
    start = time.time()
    for name in names:
        path = OUT / f"{name}.csv"
        writer, fh = None, None
        for kw in grid(name):
            if time.time() - start > budget_s:
                print("BUDGET REACHED"); return
            if path.exists() and _match(path, kw):
                continue
            try:
                res, _ = run(**kw)
            except Exception as e:
                print("FAILED", kw, e); traceback.print_exc(); continue
            if writer is None:
                new = not path.exists()
                fh = open(path, "a", newline="")
                writer = csv.DictWriter(fh, fieldnames=list(res.keys()))
                if new:
                    writer.writeheader()
            writer.writerow(res); fh.flush()
            print(name, kw, "service=%.3f" % res["service_rate"], flush=True)
        if fh:
            fh.close()


_CACHE = {}


def _match(path, kw):
    """True if a row in the CSV has the same values for every key of kw."""
    import pandas as pd
    if path not in _CACHE:
        _CACHE[path] = pd.read_csv(path, dtype=str)
    df = _CACHE[path]
    from sim import Params
    p = Params(**kw)
    mask = np.ones(len(df), bool)
    for k in kw:
        v = getattr(p, k)
        col = df[k]
        if isinstance(v, bool):
            mask &= col == str(v)
        elif isinstance(v, (int, float)):
            mask &= np.isclose(col.astype(float), float(v))
        else:
            mask &= col.str.zfill(3) == str(v)
    return bool(mask.any())


if __name__ == "__main__":
    args = sys.argv[1:]
    budget = 1e9
    if args and args[0].startswith("--budget="):
        budget = float(args.pop(0).split("=")[1])
    main(args or ["E1", "E2", "E3", "E4", "E5"], budget)
