"""Rolling-horizon ride-hailing dispatch simulator for Shenzhen.

At each decision epoch, release and expire requests, match feasible vehicle-request
pairs, commit trips, optionally reposition idle vehicles, and record metrics.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from data import load_network, load_requests, make_zones
from models import solve_assignment, solve_reposition


UNASSIGNED = 0
SERVED = 1
EXPIRED = 2


@dataclass
class Params:
    """Simulation settings.

    Times are minutes, distances are kilometres, and fares/costs are CNY.
    W is the pickup deadline, K the candidate limit, A the service reward,
    rho the blocking-pair penalty, and theta the value of shortage coverage.
    """

    gamma: str = "050"
    sample_frac: float = 0.02
    fleet: int = 250
    dt: float = 2.0
    W: float = 10.0
    speed: float = 20.0
    K: int = 8
    A: float = 10.0
    beta: float = 0.05
    fare_base: float = 10.0
    fare_km: float = 2.6
    cost_km: float = 0.5
    policy: str = "milp"
    rho: float = 1.0
    reposition: bool = False
    n_zones: int = 16
    fc_window: float = 10.0
    horizon: float = 10.0
    theta: float = 3.0
    rmax: float = 5.0
    seed: int = 3010
    t_end: float = 180.0


def gini(values):
    """Return the Gini coefficient, treating a zero-sum vector as equality."""
    sorted_values = np.sort(np.asarray(values, float))
    if sorted_values.sum() <= 0:
        return 0.0

    count = len(sorted_values)
    ranks = 2 * np.arange(1, count + 1) - count - 1
    return float(ranks.dot(sorted_values) / (count * sorted_values.sum()))


class Simulator:
    def __init__(self, p: Params):
        self.p = p
        self.distances, _ = load_network()
        self.requests = load_requests(p.gamma, p.sample_frac, p.seed, p.t_end)
        self.zone_by_node, self.zone_centres = make_zones(p.n_zones, p.seed)

        rng = np.random.default_rng(p.seed + 1)
        early_origins = self.requests["o"][self.requests["t"] < 30]
        self.vehicle_positions = rng.choice(early_origins, p.fleet)
        self.available_at = np.zeros(p.fleet)
        self.is_repositioning = np.zeros(p.fleet, dtype=bool)

        self.earnings = np.zeros(p.fleet)
        self.occupied_km = np.zeros(p.fleet)
        self.pickup_km = np.zeros(p.fleet)
        self.reposition_km = np.zeros(p.fleet)
        self.occupied_time = np.zeros(p.fleet)
        self.trip_counts = np.zeros(p.fleet, dtype=int)

        request_count = len(self.requests["t"])
        self.trip_distances = self.distances[
            self.requests["o"], self.requests["d"]
        ].astype(float)
        self.fares = p.fare_base + p.fare_km * np.maximum(0, self.trip_distances - 2.0)
        self.request_status = np.zeros(request_count, dtype=int)
        self.wait_times = np.full(request_count, np.nan)
        self.epoch_log = []

        # Preserve state attributes used by scripts written against the original simulator.
        self.D = self.distances
        self.R = self.requests
        self.zone_of = self.zone_by_node
        self.zone_centre = self.zone_centres
        self.pos = self.vehicle_positions
        self.free = self.available_at
        self.repos = self.is_repositioning
        self.earn = self.earnings
        self.occ_km = self.occupied_km
        self.pick_km = self.pickup_km
        self.rep_km = self.reposition_km
        self.occ_time = self.occupied_time
        self.ntrips = self.trip_counts
        self.L = self.trip_distances
        self.fare = self.fares
        self.status = self.request_status
        self.wait = self.wait_times

    def km2min(self, kilometres):
        return kilometres / self.p.speed * 60.0

    def build_graph(self, t, open_j):
        """Find feasible vehicle-request pairs and their scores/preferences.

        Vehicles that become free within one epoch may be considered. For each
        request, retain only the K feasible vehicles with the lowest pickup ETA.
        """
        params = self.p
        available_vehicles = np.flatnonzero(self.available_at <= t + params.dt)
        if len(available_vehicles) == 0 or len(open_j) == 0:
            return available_vehicles, [], {}, {}, {}, {}

        release_delay = np.maximum(0.0, self.available_at[available_vehicles] - t)
        pickup_distances = self.distances[
            np.ix_(
                self.vehicle_positions[available_vehicles], self.requests["o"][open_j]
            )
        ].astype(float)
        pickup_eta = release_delay[:, None] + self.km2min(pickup_distances)
        remaining_wait = self.requests["t"][open_j] + params.W - t
        feasible = pickup_eta <= remaining_wait[None, :]

        edges = []
        scores = {}
        driver_preferences = {}
        passenger_preferences = {}
        pickup_details = {}

        for request_column, request_index in enumerate(open_j):
            feasible_rows = np.flatnonzero(feasible[:, request_column])
            if len(feasible_rows) == 0:
                continue

            nearest_rows = feasible_rows[
                np.argsort(pickup_eta[feasible_rows, request_column])[: params.K]
            ]
            request_age = t - self.requests["t"][request_index]

            for vehicle_row in nearest_rows:
                vehicle = int(available_vehicles[vehicle_row])
                request = int(request_index)
                edge = (vehicle, request)
                pickup_km = pickup_distances[vehicle_row, request_column]
                eta = pickup_eta[vehicle_row, request_column]

                edges.append(edge)
                scores[edge] = params.A + params.beta * request_age - pickup_km
                driver_preferences[edge] = self.fares[request] - params.cost_km * (
                    pickup_km + self.trip_distances[request]
                )
                passenger_preferences[edge] = -eta
                pickup_details[edge] = (eta, pickup_km)

        return (
            available_vehicles,
            edges,
            scores,
            driver_preferences,
            passenger_preferences,
            pickup_details,
        )

    @staticmethod
    def blocking_pairs(edges, matched, dpref, ppref):
        """Count unmatched pairs that both participants would strictly prefer."""
        request_by_vehicle = {vehicle: request for vehicle, request in matched}
        vehicle_by_request = {request: vehicle for vehicle, request in matched}
        count = 0

        for vehicle, request in edges:
            if request_by_vehicle.get(vehicle) == request:
                continue

            driver_would_switch = vehicle not in request_by_vehicle or (
                dpref[vehicle, request],
                -request,
            ) > (
                dpref[vehicle, request_by_vehicle[vehicle]],
                -request_by_vehicle[vehicle],
            )
            passenger_would_switch = request not in vehicle_by_request or (
                ppref[vehicle, request],
                -vehicle,
            ) > (
                ppref[vehicle_by_request[request], request],
                -vehicle_by_request[request],
            )
            count += driver_would_switch and passenger_would_switch

        return count

    def _match_fcfs(self, edges, pickup_details):
        candidates_by_request = {}
        for edge in edges:
            candidates_by_request.setdefault(edge[1], []).append(edge)

        used_vehicles = set()
        matches = []
        requests_by_release_time = sorted(
            candidates_by_request, key=lambda request: self.requests["t"][request]
        )
        for request in requests_by_release_time:
            nearest = sorted(
                candidates_by_request[request], key=lambda edge: pickup_details[edge][0]
            )
            for edge in nearest:
                if edge[0] not in used_vehicles:
                    used_vehicles.add(edge[0])
                    matches.append(edge)
                    break
        return matches

    @staticmethod
    def _match_greedy(edges, scores):
        used_vehicles = set()
        used_requests = set()
        matches = []

        for edge in sorted(edges, key=lambda candidate: -scores[candidate]):
            vehicle, request = edge
            if vehicle not in used_vehicles and request not in used_requests:
                used_vehicles.add(vehicle)
                used_requests.add(request)
                matches.append(edge)
        return matches

    def match(self, t, edges, score, dpref, ppref, ETA):
        """Dispatch by FCFS, greedy score, or the selected Gurobi formulation."""
        info = {"time": 0.0, "nvars": 0, "ncons": 0}
        if not edges:
            return [], info

        started_at = time.perf_counter()
        if self.p.policy == "fcfs":
            matched = self._match_fcfs(edges, ETA)
        elif self.p.policy == "greedy":
            matched = self._match_greedy(edges, score)
        else:
            solver_mode = {
                "milp": "efficient",
                "stable": "stable",
                "nearstable": "nearstable",
            }[self.p.policy]
            return solve_assignment(
                edges, score, dpref, ppref, mode=solver_mode, rho=self.p.rho
            )

        info["time"] = time.perf_counter() - started_at
        return matched, info

    def reposition(self, t, open_j, matched_i):
        """Select idle vehicles to move toward forecast demand deficits."""
        params = self.p
        zone_count = params.n_zones
        request_times = self.requests["t"]
        origins = self.requests["o"]

        recent = (request_times > t - params.fc_window) & (request_times <= t)
        historical_demand = (
            np.bincount(self.zone_by_node[origins[recent]], minlength=zone_count)
            * params.horizon
            / params.fc_window
        )
        pending_demand = (
            np.bincount(self.zone_by_node[origins[open_j]], minlength=zone_count)
            if len(open_j)
            else np.zeros(zone_count)
        )
        projected_demand = historical_demand + pending_demand

        becoming_available = self.available_at <= t + params.horizon
        projected_supply = np.bincount(
            self.zone_by_node[self.vehicle_positions[becoming_available]],
            minlength=zone_count,
        )
        idle_vehicles = np.flatnonzero(
            (self.available_at <= t)
            & ~np.isin(np.arange(params.fleet), list(matched_i))
        )
        idle_supply = np.bincount(
            self.zone_by_node[self.vehicle_positions[idle_vehicles]],
            minlength=zone_count,
        )

        deficit = {
            zone: float(np.floor(projected_demand[zone] - projected_supply[zone]))
            for zone in range(zone_count)
            if projected_demand[zone] - projected_supply[zone] >= 1
        }
        surplus = {
            zone: int(
                min(
                    idle_supply[zone],
                    np.floor(projected_supply[zone] - projected_demand[zone]),
                )
            )
            for zone in range(zone_count)
            if projected_supply[zone] - projected_demand[zone] >= 1
            and idle_supply[zone] > 0
        }
        empty_info = {"time": 0.0, "nvars": 0, "ncons": 0}
        if not deficit or not surplus:
            return [], empty_info

        candidates = []
        distances = {}
        source_zone_by_vehicle = {}
        for vehicle_index in idle_vehicles:
            vehicle = int(vehicle_index)
            source_zone = int(self.zone_by_node[self.vehicle_positions[vehicle]])
            if source_zone not in surplus:
                continue

            source_zone_by_vehicle[vehicle] = source_zone
            reachable_zones = sorted(
                (
                    float(
                        self.distances[
                            self.vehicle_positions[vehicle], self.zone_centres[zone]
                        ]
                    ),
                    zone,
                )
                for zone in deficit
            )
            for distance, zone in reachable_zones[:3]:
                if distance <= params.rmax and distance < params.theta:
                    candidates.append((vehicle, zone))
                    distances[vehicle, zone] = distance

        if not candidates:
            return [], empty_info

        return solve_reposition(
            candidates,
            distances,
            deficit,
            surplus,
            source_zone_by_vehicle,
            theta=params.theta,
        )

    def _commit_matches(self, t, matches, pickup_details, driver_preferences):
        request_times = self.requests["t"]
        for vehicle, request in matches:
            pickup_eta, pickup_distance = pickup_details[vehicle, request]
            pickup_time = t + pickup_eta
            trip_distance = self.trip_distances[request]
            trip_time = self.km2min(trip_distance)

            self.wait_times[request] = pickup_time - request_times[request]
            self.request_status[request] = SERVED
            self.pickup_km[vehicle] += pickup_distance
            self.occupied_km[vehicle] += trip_distance
            self.occupied_time[vehicle] += trip_time
            self.available_at[vehicle] = pickup_time + trip_time
            self.vehicle_positions[vehicle] = self.requests["d"][request]
            self.is_repositioning[vehicle] = False
            self.earnings[vehicle] += driver_preferences[vehicle, request]
            self.trip_counts[vehicle] += 1

    def _commit_repositioning(self, t, moves):
        for vehicle, zone in moves:
            centre_node = self.zone_centres[zone]
            distance = float(
                self.distances[self.vehicle_positions[vehicle], centre_node]
            )
            self.reposition_km[vehicle] += distance
            self.earnings[vehicle] -= self.p.cost_km * distance
            self.available_at[vehicle] = t + self.km2min(distance)
            self.vehicle_positions[vehicle] = centre_node
            self.is_repositioning[vehicle] = True

    def _run_epoch(self, t):
        request_times = self.requests["t"]
        released = request_times <= t
        expired = (
            released
            & (self.request_status == UNASSIGNED)
            & (request_times + self.p.W < t)
        )
        self.request_status[expired] = EXPIRED
        open_requests = np.flatnonzero(released & (self.request_status == UNASSIGNED))

        (
            available_vehicles,
            edges,
            scores,
            driver_preferences,
            passenger_preferences,
            pickup_details,
        ) = self.build_graph(t, open_requests)
        matched, matching_info = self.match(
            t, edges, scores, driver_preferences, passenger_preferences, pickup_details
        )
        blocking_count = (
            self.blocking_pairs(
                edges, matched, driver_preferences, passenger_preferences
            )
            if edges
            else 0
        )
        self._commit_matches(t, matched, pickup_details, driver_preferences)

        reposition_info = {"time": 0.0, "nvars": 0, "ncons": 0}
        reposition_count = 0
        if self.p.reposition:
            pending = np.flatnonzero(released & (self.request_status == UNASSIGNED))
            moves, reposition_info = self.reposition(
                t, pending, {vehicle for vehicle, _ in matched}
            )
            self._commit_repositioning(t, moves)
            reposition_count = len(moves)

        self.epoch_log.append(
            {
                "t": t,
                "open": len(open_requests),
                "avail": len(available_vehicles),
                "edges": len(edges),
                "matched": len(matched),
                "blocking": blocking_count,
                "solve_time": matching_info["time"],
                "nvars": matching_info["nvars"],
                "ncons": matching_info["ncons"],
                "repos_moves": reposition_count,
                "repos_time": reposition_info["time"],
            }
        )

    def run(self):
        started_at = time.perf_counter()
        t = 0.0
        while t <= self.p.t_end + 1e-9:
            self._run_epoch(t)
            t += self.p.dt

        self.request_status[self.request_status == UNASSIGNED] = EXPIRED
        self.runtime = time.perf_counter() - started_at
        return self.metrics()

    def metrics(self):
        """Return the original flat metrics schema, including all parameter values."""
        served = self.request_status == SERVED
        waiting = self.wait_times[served]
        empty_km = self.pickup_km + self.reposition_km
        total_km = empty_km + self.occupied_km
        matched_count = sum(epoch["matched"] for epoch in self.epoch_log)
        blocking_count = sum(epoch["blocking"] for epoch in self.epoch_log)

        passenger_metrics = {
            "requests": int(len(served)),
            "served": int(served.sum()),
            "service_rate": float(served.mean()),
            "wait_mean": float(waiting.mean()) if len(waiting) else np.nan,
            "wait_p95": float(np.percentile(waiting, 95)) if len(waiting) else np.nan,
            "pickup_km_mean": float(self.pickup_km.sum() / max(1, served.sum())),
            "service_h1_offpeak": float(served[self.requests["t"] < 60].mean()),
            "service_h2_peak": float(
                served[(self.requests["t"] >= 60) & (self.requests["t"] < 120)].mean()
            ),
            "service_h3_offpeak": float(served[self.requests["t"] >= 120].mean()),
        }
        driver_metrics = {
            "utilisation": float(
                self.occupied_time.sum() / (self.p.fleet * self.p.t_end)
            ),
            "earn_mean": float(self.earnings.mean()),
            "earn_gini": gini(np.maximum(self.earnings, 0)),
            "trips_per_veh_sd": float(self.trip_counts.std()),
        }
        system_metrics = {
            "vkt_total": float(total_km.sum()),
            "vkt_empty": float(empty_km.sum()),
            "empty_share": float(empty_km.sum() / max(1e-9, total_km.sum())),
            "repos_km": float(self.reposition_km.sum()),
            "blocking_pairs": int(blocking_count),
            "blocking_per_match": float(blocking_count / max(1, matched_count)),
        }
        computational_metrics = {
            "solve_mean_s": float(np.mean([e["solve_time"] for e in self.epoch_log])),
            "solve_max_s": float(np.max([e["solve_time"] for e in self.epoch_log])),
            "edges_mean": float(np.mean([e["edges"] for e in self.epoch_log])),
            "edges_max": int(np.max([e["edges"] for e in self.epoch_log])),
            "nvars_max": int(np.max([e["nvars"] for e in self.epoch_log])),
            "ncons_max": int(np.max([e["ncons"] for e in self.epoch_log])),
            "repos_time_mean_s": float(
                np.mean([e["repos_time"] for e in self.epoch_log])
            ),
            "runtime_s": self.runtime,
        }
        return {
            **asdict(self.p),
            **passenger_metrics,
            **driver_metrics,
            **system_metrics,
            **computational_metrics,
        }


def run(**kw):
    simulator = Simulator(Params(**kw))
    result = simulator.run()
    return result, simulator
