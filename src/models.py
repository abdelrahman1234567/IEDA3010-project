"""Gurobi models for dispatch matching and idle-vehicle repositioning."""

import time

import gurobipy as gp
from gurobipy import GRB


_ENV = None


def env():
    """Reuse a silent Gurobi environment across simulation epochs."""
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def _group_assignment_edges(edges):
    requests_by_vehicle = {}
    vehicles_by_request = {}

    for vehicle, request in edges:
        requests_by_vehicle.setdefault(vehicle, []).append(request)
        vehicles_by_request.setdefault(request, []).append(vehicle)

    return requests_by_vehicle, vehicles_by_request


def _add_stability_constraints(
    model,
    assignments,
    edges,
    requests_by_vehicle,
    vehicles_by_request,
    driver_preferences,
    passenger_preferences,
    slacks=None,
):
    """Exclude blocking pairs, or allow them via binary near-stability slacks.

    Unmatched participants prefer any feasible match. Preference ties are broken
    by the lower participant index, consistent with the simulator's diagnostics.
    """
    for vehicle, request in edges:
        preferred_requests = [
            alternative
            for alternative in requests_by_vehicle[vehicle]
            if (driver_preferences[vehicle, alternative], -alternative)
            > (driver_preferences[vehicle, request], -request)
        ]
        preferred_vehicles = [
            alternative
            for alternative in vehicles_by_request[request]
            if (passenger_preferences[alternative, request], -alternative)
            > (passenger_preferences[vehicle, request], -vehicle)
        ]

        satisfied = (
            assignments[vehicle, request]
            + gp.quicksum(assignments[vehicle, other] for other in preferred_requests)
            + gp.quicksum(assignments[other, request] for other in preferred_vehicles)
        )
        if slacks is not None:
            satisfied += slacks[vehicle, request]

        model.addConstr(satisfied >= 1, name=f"stab[{vehicle},{request}]")


def solve_assignment(
    edges,
    score,
    drv_pref,
    pax_pref,
    mode="efficient",
    rho=0.0,
    time_limit=20.0,
    threads=1,
):
    """Maximise assignment score subject to matching and optional stability.

    In near-stable mode each blocking pair incurs a penalty of ``rho``.
    Returns matched (vehicle, request) pairs and solver diagnostics.
    """
    started_at = time.perf_counter()
    model = gp.Model(env=env())
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.MIPGap = 1e-6

    assignments = model.addVars(edges, vtype=GRB.BINARY, name="x")
    requests_by_vehicle, vehicles_by_request = _group_assignment_edges(edges)

    model.addConstrs(
        (assignments.sum(vehicle, "*") <= 1 for vehicle in requests_by_vehicle),
        name="veh",
    )
    model.addConstrs(
        (assignments.sum("*", request) <= 1 for request in vehicles_by_request),
        name="req",
    )

    objective = gp.quicksum(score[edge] * assignments[edge] for edge in edges)
    if mode in ("stable", "nearstable"):
        slacks = None
        if mode == "nearstable":
            slacks = model.addVars(edges, vtype=GRB.BINARY, name="s")
            objective -= rho * slacks.sum()

        _add_stability_constraints(
            model,
            assignments,
            edges,
            requests_by_vehicle,
            vehicles_by_request,
            drv_pref,
            pax_pref,
            slacks,
        )

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    elapsed = time.perf_counter() - started_at

    if model.SolCount == 0:
        info = {
            "status": model.Status,
            "time": elapsed,
            "nvars": model.NumVars,
            "ncons": model.NumConstrs,
            "gap": None,
        }
        model.dispose()
        return [], info

    matched = [edge for edge in edges if assignments[edge].X > 0.5]
    info = {
        "status": model.Status,
        "time": elapsed,
        "solver_time": model.Runtime,
        "nvars": model.NumVars,
        "ncons": model.NumConstrs,
        "obj": model.ObjVal,
        "gap": model.MIPGap if model.IsMIP else 0.0,
    }
    model.dispose()
    return matched, info


def solve_reposition(
    cands,
    cost,
    deficit,
    surplus,
    veh_zone,
    theta=3.0,
    time_limit=10.0,
    threads=1,
):
    """Move surplus-zone vehicles to forecast-shortage zones.

    Maximises theta times shortage coverage minus empty repositioning distance,
    without moving a vehicle twice or draining a source zone beyond its surplus.
    """
    started_at = time.perf_counter()
    model = gp.Model(env=env())
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    moves = model.addVars(cands, vtype=GRB.BINARY, name="y")
    destination_zones = sorted({zone for _, zone in cands})
    covered_shortage = model.addVars(
        destination_zones,
        lb=0.0,
        ub={zone: deficit[zone] for zone in destination_zones},
        name="u",
    )
    vehicles = sorted({vehicle for vehicle, _ in cands})

    model.addConstrs(
        (moves.sum(vehicle, "*") <= 1 for vehicle in vehicles), name="once"
    )
    model.addConstrs(
        (covered_shortage[zone] <= moves.sum("*", zone) for zone in destination_zones),
        name="cover",
    )

    for source_zone, capacity in surplus.items():
        outgoing = [edge for edge in cands if veh_zone[edge[0]] == source_zone]
        if outgoing:
            model.addConstr(
                gp.quicksum(moves[edge] for edge in outgoing) <= capacity,
                name=f"src[{source_zone}]",
            )

    objective = theta * covered_shortage.sum() - gp.quicksum(
        cost[edge] * moves[edge] for edge in cands
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    selected_moves = (
        [edge for edge in cands if moves[edge].X > 0.5] if model.SolCount else []
    )
    info = {
        "time": time.perf_counter() - started_at,
        "nvars": model.NumVars,
        "ncons": model.NumConstrs,
    }
    model.dispose()
    return selected_moves, info
