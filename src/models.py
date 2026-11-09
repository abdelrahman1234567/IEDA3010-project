"""Gurobi optimization models solved at every decision epoch.

Notation (one epoch, decision time t):
  I        available vehicles (idle, repositioning, or finishing a trip within the epoch)
  J        open requests (released, not yet served, not yet expired)
  E        feasible vehicle-request pairs (i, j): t + eta_ij <= r_j + W, pruned to the
           K vehicles with smallest eta for each request
  w_ij     platform score of pair (i, j)
  x_ij     1 if vehicle i is dispatched to request j now

Preferences used for stability (strict, ties broken by index):
  passenger j prefers smaller pickup ETA
  driver i prefers higher trip profit  p_ij = fare(L_j) - c_km * (pickup_km_ij + L_j)
"""
import time
import gurobipy as gp
from gurobipy import GRB

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def solve_assignment(edges, score, drv_pref, pax_pref, mode="efficient",
                     rho=0.0, time_limit=20.0, threads=1):
    """Epoch matching MILP.

    edges    : list of (i, j)
    score    : dict (i, j) -> w_ij
    drv_pref : dict (i, j) -> driver utility (higher = better)
    pax_pref : dict (i, j) -> passenger utility (higher = better)
    mode     : 'efficient'  max sum w x
               'stable'     + hard stability constraints (no blocking pair)
               'nearstable' + soft stability: slack s_ij in {0,1} penalised by rho
    Returns (list of matched pairs, info dict).
    """
    t0 = time.perf_counter()
    m = gp.Model(env=env())
    m.Params.TimeLimit = time_limit
    m.Params.Threads = threads
    m.Params.MIPGap = 1e-6
    x = m.addVars(edges, vtype=GRB.BINARY, name="x")
    by_i, by_j = {}, {}
    for (i, j) in edges:
        by_i.setdefault(i, []).append(j)
        by_j.setdefault(j, []).append(i)
    # each vehicle serves at most one request, each request gets at most one vehicle
    m.addConstrs((x.sum(i, "*") <= 1 for i in by_i), name="veh")
    m.addConstrs((x.sum("*", j) <= 1 for j in by_j), name="req")
    obj = gp.quicksum(score[e] * x[e] for e in edges)

    if mode in ("stable", "nearstable"):
        # (i, j) must not block: either matched together, or i gets a request it
        # strictly prefers to j, or j gets a vehicle it strictly prefers to i.
        if mode == "nearstable":
            s = m.addVars(edges, vtype=GRB.BINARY, name="s")
            obj -= rho * s.sum()
        for (i, j) in edges:
            better_j = [jj for jj in by_i[i] if (drv_pref[i, jj], -jj) > (drv_pref[i, j], -j)]
            better_i = [ii for ii in by_j[j] if (pax_pref[ii, j], -ii) > (pax_pref[i, j], -i)]
            lhs = x[i, j] + gp.quicksum(x[i, jj] for jj in better_j) \
                + gp.quicksum(x[ii, j] for ii in better_i)
            if mode == "stable":
                m.addConstr(lhs >= 1, name=f"stab[{i},{j}]")
            else:
                m.addConstr(lhs + s[i, j] >= 1, name=f"stab[{i},{j}]")

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    build_solve = time.perf_counter() - t0
    if m.SolCount == 0:
        return [], {"status": m.Status, "time": build_solve, "nvars": m.NumVars,
                    "ncons": m.NumConstrs, "gap": None}
    matched = [e for e in edges if x[e].X > 0.5]
    info = {"status": m.Status, "time": build_solve, "solver_time": m.Runtime,
            "nvars": m.NumVars, "ncons": m.NumConstrs, "obj": m.ObjVal,
            "gap": m.MIPGap if m.IsMIP else 0.0}
    m.dispose()
    return matched, info


def solve_reposition(cands, cost, deficit, surplus, veh_zone, theta=3.0,
                     time_limit=10.0, threads=1):
    """Idle-vehicle repositioning MILP (extension).

    cands   : list of (i, z) candidate moves of idle vehicle i to zone z
    cost    : dict (i, z) -> empty km of the move
    deficit : dict z -> forecast shortage of vehicles in zone z (>= 0)
    surplus : dict zone -> number of vehicles the zone can give away
    veh_zone: dict i -> current zone of vehicle i
    theta   : value (km-equivalent) of covering one unit of forecast shortage

    max  theta * sum_z u_z - sum_{i,z} c_iz y_iz
    s.t. sum_z y_iz <= 1                    each vehicle moves at most once
         u_z <= sum_i y_iz,  u_z <= D_z     coverage counts only up to the shortage
         sum_{i in zone s} sum_z y_iz <= S_s  do not drain a zone below its own forecast
         y binary, u >= 0
    """
    t0 = time.perf_counter()
    m = gp.Model(env=env())
    m.Params.TimeLimit = time_limit
    m.Params.Threads = threads
    y = m.addVars(cands, vtype=GRB.BINARY, name="y")
    zones = sorted({z for (_, z) in cands})
    u = m.addVars(zones, lb=0.0, ub={z: deficit[z] for z in zones}, name="u")
    vehs = sorted({i for (i, _) in cands})
    m.addConstrs((y.sum(i, "*") <= 1 for i in vehs), name="once")
    m.addConstrs((u[z] <= y.sum("*", z) for z in zones), name="cover")
    for s_zone, cap in surplus.items():
        members = [(i, z) for (i, z) in cands if veh_zone[i] == s_zone]
        if members:
            m.addConstr(gp.quicksum(y[e] for e in members) <= cap, name=f"src[{s_zone}]")
    m.setObjective(theta * u.sum() - gp.quicksum(cost[e] * y[e] for e in cands), GRB.MAXIMIZE)
    m.optimize()
    moves = [e for e in cands if y[e].X > 0.5] if m.SolCount else []
    info = {"time": time.perf_counter() - t0, "nvars": m.NumVars, "ncons": m.NumConstrs}
    m.dispose()
    return moves, info
