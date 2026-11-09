"""Rolling-horizon ride-hailing dispatch simulator (Shenzhen Futian + Luohu).

Decision timing: epochs at t = 0, dt, 2dt, ... (minutes). At each epoch the
platform (1) adds requests released in (t - dt, t], (2) drops requests that can
no longer be picked up within the waiting tolerance W, (3) builds the feasible
vehicle-request graph, (4) solves the matching policy and commits every match
immediately (the vehicle departs at t), (5) optionally repositions idle
vehicles, and (6) carries unmatched, still-feasible requests to the next epoch.
"""
import time
from dataclasses import dataclass, field, asdict
import numpy as np

from data import load_network, load_requests, make_zones
from models import solve_assignment, solve_reposition


@dataclass
class Params:
    gamma: str = "050"          # OD-imbalance scenario of the dataset
    sample_frac: float = 0.02    # share of the 159,769 requests kept
    fleet: int = 250
    dt: float = 2.0             # epoch length (min)
    W: float = 10.0             # max wait from request to pickup (min); dataset 'wtol'
    speed: float = 20.0         # km/h, constant network speed
    K: int = 8                  # keep the K nearest vehicles per request
    A: float = 10.0             # reward per served request (km-equivalent)
    beta: float = 0.05          # extra reward per minute already waited
    fare_base: float = 10.0     # CNY for first 2 km (Shenzhen-style taxi fare, assumed)
    fare_km: float = 2.6        # CNY per km beyond 2 km
    cost_km: float = 0.5        # driver cost CNY per driven km
    policy: str = "milp"        # fcfs | greedy | milp | stable | nearstable
    rho: float = 1.0            # penalty per blocking pair (nearstable)
    reposition: bool = False
    n_zones: int = 16
    fc_window: float = 10.0     # demand forecast look-back (min)
    horizon: float = 10.0       # repositioning look-ahead (min)
    theta: float = 3.0          # value of covering one forecast shortage (km)
    rmax: float = 5.0           # max repositioning distance (km)
    seed: int = 3010
    t_end: float = 180.0


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


class Simulator:
    def __init__(self, p: Params):
        self.p = p
        self.D, _ = load_network()
        self.R = load_requests(p.gamma, p.sample_frac, p.seed, p.t_end)
        self.zone_of, self.zone_centre = make_zones(p.n_zones, p.seed)
        rng = np.random.default_rng(p.seed + 1)
        # initial fleet: drawn from the origin distribution of the first 30 min
        early = self.R["o"][self.R["t"] < 30]
        n = p.fleet
        self.pos = rng.choice(early, n)          # node where vehicle becomes free
        self.free = np.zeros(n)                  # time (min) it becomes free there
        self.repos = np.zeros(n, bool)           # currently repositioning
        self.earn = np.zeros(n)
        self.occ_km = np.zeros(n)
        self.pick_km = np.zeros(n)
        self.rep_km = np.zeros(n)
        self.occ_time = np.zeros(n)
        self.ntrips = np.zeros(n, int)
        nr = len(self.R["t"])
        self.L = self.D[self.R["o"], self.R["d"]].astype(float)
        self.fare = self.p.fare_base + self.p.fare_km * np.maximum(0, self.L - 2.0)
        self.status = np.zeros(nr, int)          # 0 not yet / open, 1 served, 2 expired
        self.wait = np.full(nr, np.nan)
        self.epoch_log = []

    # ------------------------------------------------------------------ helpers
    def km2min(self, km):
        return km / self.p.speed * 60.0

    def build_graph(self, t, open_j):
        p = self.p
        avail = np.flatnonzero(self.free <= t + p.dt)
        if len(avail) == 0 or len(open_j) == 0:
            return avail, [], {}, {}, {}, {}
        lag = np.maximum(0.0, self.free[avail] - t)
        pk = self.D[np.ix_(self.pos[avail], self.R["o"][open_j])].astype(float)   # km
        eta = lag[:, None] + self.km2min(pk)
        deadline = self.R["t"][open_j] + p.W - t
        feas = eta <= deadline[None, :]
        edges, score, dpref, ppref, ETA = [], {}, {}, {}, {}
        for c, j in enumerate(open_j):
            rows = np.flatnonzero(feas[:, c])
            if len(rows) == 0:
                continue
            rows = rows[np.argsort(eta[rows, c])[:p.K]]
            age = t - self.R["t"][j]
            for r in rows:
                i = int(avail[r]); j = int(j)
                e = (i, j)
                edges.append(e)
                score[e] = p.A + p.beta * age - pk[r, c]
                dpref[e] = self.fare[j] - p.cost_km * (pk[r, c] + self.L[j])
                ppref[e] = -eta[r, c]
                ETA[e] = (eta[r, c], pk[r, c])
        return avail, edges, score, dpref, ppref, ETA

    @staticmethod
    def blocking_pairs(edges, matched, dpref, ppref):
        mi = {i: j for (i, j) in matched}
        mj = {j: i for (i, j) in matched}
        nb = 0
        for (i, j) in edges:
            if mi.get(i) == j:
                continue
            di = i not in mi or (dpref[i, j], -j) > (dpref[i, mi[i]], -mi[i])
            pj = j not in mj or (ppref[i, j], -i) > (ppref[mj[j], j], -mj[j])
            nb += di and pj
        return nb

    # ---------------------------------------------------------------- policies
    def match(self, t, edges, score, dpref, ppref, ETA):
        p = self.p
        info = {"time": 0.0, "nvars": 0, "ncons": 0}
        if not edges:
            return [], info
        t0 = time.perf_counter()
        if p.policy == "fcfs":
            # requests in order of release; each takes its nearest (min ETA) free vehicle
            by_j = {}
            for e in edges:
                by_j.setdefault(e[1], []).append(e)
            used, out = set(), []
            for j in sorted(by_j, key=lambda j: self.R["t"][j]):
                for e in sorted(by_j[j], key=lambda e: ETA[e][0]):
                    if e[0] not in used:
                        used.add(e[0]); out.append(e); break
            info["time"] = time.perf_counter() - t0
            return out, info
        if p.policy == "greedy":
            # repeatedly take the highest-score remaining pair
            used_i, used_j, out = set(), set(), []
            for e in sorted(edges, key=lambda e: -score[e]):
                if e[0] not in used_i and e[1] not in used_j:
                    used_i.add(e[0]); used_j.add(e[1]); out.append(e)
            info["time"] = time.perf_counter() - t0
            return out, info
        mode = {"milp": "efficient", "stable": "stable", "nearstable": "nearstable"}[p.policy]
        return solve_assignment(edges, score, dpref, ppref, mode=mode, rho=p.rho)

    def reposition(self, t, open_j, matched_i):
        p = self.p
        Z = p.n_zones
        # demand forecast per zone over the next `horizon` minutes
        recent = (self.R["t"] > t - p.fc_window) & (self.R["t"] <= t)
        lam = np.bincount(self.zone_of[self.R["o"][recent]], minlength=Z) * p.horizon / p.fc_window
        pend = np.bincount(self.zone_of[self.R["o"][open_j]], minlength=Z) if len(open_j) else np.zeros(Z)
        demand = lam + pend
        # supply: vehicles that will be free in the zone within the horizon
        soon = self.free <= t + p.horizon
        supply = np.bincount(self.zone_of[self.pos[soon]], minlength=Z)
        idle = np.flatnonzero((self.free <= t) & ~np.isin(np.arange(p.fleet), list(matched_i)))
        idle_cnt = np.bincount(self.zone_of[self.pos[idle]], minlength=Z)
        deficit = {z: float(np.floor(demand[z] - supply[z])) for z in range(Z) if demand[z] - supply[z] >= 1}
        surplus = {z: int(min(idle_cnt[z], np.floor(supply[z] - demand[z])))
                   for z in range(Z) if supply[z] - demand[z] >= 1 and idle_cnt[z] > 0}
        if not deficit or not surplus:
            return [], {"time": 0.0, "nvars": 0, "ncons": 0}
        cands, cost, vz = [], {}, {}
        for i in idle:
            zi = int(self.zone_of[self.pos[i]])
            if zi not in surplus:
                continue
            vz[int(i)] = zi
            dz = sorted(((float(self.D[self.pos[i], self.zone_centre[z]]), z) for z in deficit))
            for d, z in dz[:3]:
                if d <= p.rmax and d < p.theta:
                    cands.append((int(i), z)); cost[int(i), z] = d
        if not cands:
            return [], {"time": 0.0, "nvars": 0, "ncons": 0}
        return solve_reposition(cands, cost, deficit, surplus, vz, theta=p.theta)

    # -------------------------------------------------------------------- run
    def run(self):
        p = self.p
        Rt = self.R["t"]
        t = 0.0
        tic = time.perf_counter()
        while t <= p.t_end + 1e-9:
            released = Rt <= t
            expired = released & (self.status == 0) & (Rt + p.W < t)
            self.status[expired] = 2
            open_j = np.flatnonzero(released & (self.status == 0))
            avail, edges, score, dpref, ppref, ETA = self.build_graph(t, open_j)
            matched, info = self.match(t, edges, score, dpref, ppref, ETA)
            nblock = self.blocking_pairs(edges, matched, dpref, ppref) if edges else 0
            for (i, j) in matched:
                eta, pk = ETA[i, j]
                pickup_t = t + eta
                self.wait[j] = pickup_t - Rt[j]
                self.status[j] = 1
                self.pick_km[i] += pk
                self.occ_km[i] += self.L[j]
                ride = self.km2min(self.L[j])
                self.occ_time[i] += ride
                self.free[i] = pickup_t + ride
                self.pos[i] = self.R["d"][j]
                self.repos[i] = False
                self.earn[i] += dpref[i, j]
                self.ntrips[i] += 1
            rinfo = {"time": 0.0, "nvars": 0, "ncons": 0}
            nmoves = 0
            if p.reposition:
                still_open = np.flatnonzero(released & (self.status == 0))
                moves, rinfo = self.reposition(t, still_open, {i for i, _ in matched})
                for (i, z) in moves:
                    c = self.zone_centre[z]
                    d = float(self.D[self.pos[i], c])
                    self.rep_km[i] += d
                    self.earn[i] -= p.cost_km * d
                    self.free[i] = t + self.km2min(d)
                    self.pos[i] = c
                    self.repos[i] = True
                nmoves = len(moves)
            self.epoch_log.append({"t": t, "open": len(open_j), "avail": len(avail),
                                   "edges": len(edges), "matched": len(matched),
                                   "blocking": nblock, "solve_time": info["time"],
                                   "nvars": info["nvars"], "ncons": info["ncons"],
                                   "repos_moves": nmoves, "repos_time": rinfo["time"]})
            t += p.dt
        self.status[self.status == 0] = 2
        self.runtime = time.perf_counter() - tic
        return self.metrics()

    def metrics(self):
        p = self.p
        served = self.status == 1
        w = self.wait[served]
        empty = self.pick_km + self.rep_km
        tot_km = empty + self.occ_km
        ep = self.epoch_log
        matched_pairs = sum(e["matched"] for e in ep)
        m = {
            # passenger
            "requests": int(len(served)), "served": int(served.sum()),
            "service_rate": float(served.mean()),
            "wait_mean": float(w.mean()) if len(w) else np.nan,
            "wait_p95": float(np.percentile(w, 95)) if len(w) else np.nan,
            "pickup_km_mean": float(self.pick_km.sum() / max(1, served.sum())),
            "service_h1_offpeak": float(served[self.R["t"] < 60].mean()),
            "service_h2_peak": float(served[(self.R["t"] >= 60) & (self.R["t"] < 120)].mean()),
            "service_h3_offpeak": float(served[self.R["t"] >= 120].mean()),
            # driver
            "utilisation": float(self.occ_time.sum() / (p.fleet * p.t_end)),
            "earn_mean": float(self.earn.mean()),
            "earn_gini": gini(np.maximum(self.earn, 0)),
            "trips_per_veh_sd": float(self.ntrips.std()),
            # system
            "vkt_total": float(tot_km.sum()), "vkt_empty": float(empty.sum()),
            "empty_share": float(empty.sum() / max(1e-9, tot_km.sum())),
            "repos_km": float(self.rep_km.sum()),
            "blocking_pairs": int(sum(e["blocking"] for e in ep)),
            "blocking_per_match": float(sum(e["blocking"] for e in ep) / max(1, matched_pairs)),
            # computational
            "solve_mean_s": float(np.mean([e["solve_time"] for e in ep])),
            "solve_max_s": float(np.max([e["solve_time"] for e in ep])),
            "edges_mean": float(np.mean([e["edges"] for e in ep])),
            "edges_max": int(np.max([e["edges"] for e in ep])),
            "nvars_max": int(np.max([e["nvars"] for e in ep])),
            "ncons_max": int(np.max([e["ncons"] for e in ep])),
            "repos_time_mean_s": float(np.mean([e["repos_time"] for e in ep])),
            "runtime_s": self.runtime,
        }
        return {**asdict(p), **m}


def run(**kw):
    sim = Simulator(Params(**kw))
    res = sim.run()
    return res, sim
