# IEDA3010 Project E: Efficiency, Stability and Fairness in Dynamic Ride-hailing

Rolling-horizon dispatch on the Shenzhen Futian + Luohu network (the districts facing
Hong Kong across the Lo Wu and Futian/Lok Ma Chau boundary crossings), solved with
Python + Gurobi, compared against simple benchmarks, with two extensions beyond
Najmi et al. (2017) and Wang et al. (2018):

1. **Stable and near-stable matching**: a penalty on blocking pairs traces the full
   efficiency vs stability frontier (price of stability).
2. **Idle-vehicle repositioning under OD imbalance**: a second MILP each epoch moves idle
   vehicles from surplus to shortage zones, tested across the dataset's imbalance
   parameter gamma.

## Folder layout

```
Project E datasets/      original data (unchanged)
data_processed/          shenzhen.npz (distance matrix + requests, from src/extract_mat.py)
src/data.py              loading, request sampling, k-means zones
src/models.py            Gurobi models: epoch matching (efficient / stable / near-stable), repositioning
src/sim.py               rolling-horizon simulator, benchmarks (FCFS, greedy), metrics
src/run_experiments.py   experiment grid E1-E5 (resumable; results/<exp>.csv)
src/analyze.py           summaries (results/*_summary.csv) and figures/*.png
```

Run: `cd src && python3 extract_mat.py && python3 run_experiments.py && python3 analyze.py`
(needs numpy, scipy, pandas, matplotlib, gurobipy). One simulation takes 1 to 8 s.
A single run: `python3 -c "from sim import run; print(run(policy='stable', gamma='000')[0])"`.

## Data and constructed instance

* Source: `request_gamma_*.mat`. Each file also holds `alldists`, the precomputed
  all-pairs shortest road distance (km) between the 1,858 nodes, and `N_coord`
  (planar metres, 12.8 km x 8.4 km area). `wtol = 10` and `mdetour = 0.2` are stored too.
* 159,769 requests over 3 h: about 10k per 15 min in hours 1 and 3, about 20k per 15 min in
  hour 2 (a peak with 2x demand). Origins are identical in all gamma files; only
  destinations change. gamma = 0 is the most imbalanced OD pattern, gamma = 1 the least.
* **Sampling**: each request kept with probability 2% (seed 3010 to 3014), giving
  about 3,200 requests per run. Same seed means the same requests for every gamma.
* **Fleet**: 250 vehicles, initial positions drawn from origins of the first 30 min.
* **Assumptions (document in the report)**: constant speed 20 km/h; max wait W = 10 min
  (the dataset's `wtol`); fare CNY 10 for the first 2 km + 2.6/km (Shenzhen-style
  taxi tariff, assumed); driver cost CNY 0.5/km; 16 k-means zones for repositioning.

## Rolling-horizon procedure

Epochs every dt = 2 min (1, 5 and 10 also tested). At epoch t:

1. add requests released since the last epoch; drop any request with t > r_j + W;
2. available vehicles = those free now or within the next dt (idle, repositioning, or
   finishing a trip); ETA_ij = remaining busy time + d(pos_i, o_j) / v;
3. feasible pairs: t + ETA_ij <= r_j + W; keep the K = 8 smallest-ETA vehicles per request;
4. solve the matching policy and **commit all matches immediately**;
5. (extension) solve the repositioning MILP for idle, unmatched vehicles;
6. unmatched, still-feasible requests carry over to the next epoch.

## Models (one epoch)

Sets: I available vehicles, J open requests, E feasible pairs.
Parameters: pickup distance p_ij (km), trip length L_j, waiting age a_j = t - r_j.

**Efficient matching (core MILP)**

    max   sum_{(i,j) in E} (A + beta * a_j - p_ij) x_ij
    s.t.  sum_j x_ij <= 1      for all i in I     (one request per vehicle)
          sum_i x_ij <= 1      for all j in J     (one vehicle per request)
          x_ij in {0,1}

A = 10 km-equivalent per served request dominates the pickup term (max pickup is about
3.3 km), so the model maximises the number served, then breaks ties by older requests
(beta = 0.05/min) and shorter pickups. Without the stability rows the constraint
matrix is totally unimodular, so the LP relaxation is already integral.

**Preferences and blocking pairs.** Passenger j prefers smaller ETA. Driver i prefers
higher trip profit u_ij = fare(L_j) - c (p_ij + L_j), i.e. long trips with short pickups.
(i, j) in E is a blocking pair if both strictly prefer each other to their current
outcome (being unmatched is worst).

**Stable matching (hard).** For every (i, j) in E:

    x_ij + sum_{j' : u_ij' > u_ij} x_ij' + sum_{i' : ETA_i'j < ETA_ij} x_i'j >= 1

i.e. either i and j are matched, or i got a request it prefers, or j got a vehicle it
prefers. A stable matching always exists, so this model is always feasible.

**Near-stable matching (extension).** Add a slack s_ij in {0,1} to each row and subtract
rho * sum s_ij from the objective. rho = 0 is the efficient model; rho -> infinity is the
stable model. Varying rho traces the price of stability.

**Repositioning (extension).** Forecast zone demand for the next 10 min as the arrivals
in the last 10 min plus pending requests; supply = vehicles free in the zone within 10 min.
Shortage D_z and surplus S_z follow. Candidate moves (i, z): idle vehicle i in a
surplus zone to one of its 3 nearest shortage zones, distance c_iz <= theta.

    max   theta * sum_z u_z - sum_{(i,z)} c_iz y_iz
    s.t.  sum_z y_iz <= 1                      each vehicle moves at most once
          u_z <= sum_i y_iz,  0 <= u_z <= D_z   coverage counted up to the shortage
          sum_{i in zone s, z} y_iz <= S_s      do not drain a zone below its own forecast
          y_iz in {0,1}

theta is the value (km-equivalent) the platform puts on covering one unit of shortage.

**Benchmarks.** FCFS nearest driver: requests in release order, each takes the free
vehicle with the smallest ETA. Greedy best pair: repeatedly take the highest-score pair.
Both use the same feasible graph, epochs and simulator.

## Experiments (5 seeds each, mean reported, sd in results/*_summary.csv)

| Exp | What varies | Fixed |
|---|---|---|
| E1 | policy x gamma in {0, .25, .5, .75, 1} | fleet 250, dt 2 |
| E2 | near-stable rho in {0, .05, ..., 5, inf} | gamma 0 and 0.5 |
| E3 | decision interval dt in {1, 2, 5, 10} min | gamma 0.5 |
| E4 | fleet size 150 to 350 | gamma 0 |
| E5 | repositioning value theta 1.5 to 5 km | gamma 0 |

## Key results

**E1, gamma = 0.5 (policy comparison)**

| Policy | Served | Peak-hour served | Mean wait (min) | Pickup (km) | Utilisation | Driver profit (CNY/3 h) | Profit Gini | Empty-km share | Blocking pairs / match |
|---|---|---|---|---|---|---|---|---|---|
| FCFS nearest driver | 87.1% | 76.0% | 5.72 | 0.96 | 58.1% | 132.3 | 0.149 | 23.4% | 2.21 |
| Greedy best pair | 84.5% | 72.9% | 4.60 | 0.77 | 56.1% | 129.0 | 0.204 | 19.8% | 1.45 |
| MILP (efficient) | 88.7% | 78.3% | 4.83 | 0.87 | 59.2% | 135.3 | 0.152 | 21.7% | 1.39 |
| MILP + repositioning | 89.3% | 79.3% | 4.72 | 0.83 | 59.8% | 135.4 | 0.136 | 24.7% | 1.39 |
| Stable MILP | 76.7% | 59.0% | 4.77 | 1.01 | 57.0% | 122.3 | 0.152 | 22.4% | 0 |

Service rate across gamma (0 / 0.25 / 0.5 / 0.75 / 1):
FCFS 72.1 / 80.0 / 87.1 / 95.5 / 99.6;
MILP 75.0 / 82.3 / 88.7 / 96.6 / 99.8;
MILP + repositioning 76.0 / 82.6 / 89.3 / 97.0 / 99.8;
Stable 60.7 / 67.2 / 76.7 / 88.1 / 98.9.

Takeaways
* The MILP beats FCFS everywhere: +2.9 points of service at gamma = 0, +1.7 at 0.5, and
  0.9 min shorter mean wait at gamma = 0.5. Greedy best pair cuts pickup distance but
  serves fewer than FCFS; myopic "nearest pair first" strands requests.
* Imbalance is the main driver of lost service (72% to 99.6% for FCFS across gamma).
  Most requests are lost in peak hour 2. At gamma = 0.5 the MILP serves 99.5% in hour 1
  and 98.7% in hour 3, but at gamma = 0 hour 3 recovers only to 82.5%: vehicles stay
  stranded in drop-off zones after the peak, which is what repositioning targets.
* **Price of stability is large**: hard stability costs 12 points of service at
  gamma = 0.5 (14 at gamma = 0) because drivers prefer long trips while passengers prefer
  nearby drivers. All stable matchings serve the same set of agents (rural hospitals
  theorem), so the optimiser cannot recover efficiency inside the stable set.
* **Near-stability is cheap at first** (E2, gamma = 0.5): rho = 0.05 removes 38% of
  blocking pairs for 1.1 points of service; rho = 1 removes 88% for 6.8 points.
* **Repositioning pays under imbalance** (E5, gamma = 0): theta = 3 km gives +1.0 point,
  theta = 4 gives +3.5, theta = 5 gives +4.1 (75.0% to 79.2%), at a cost of 1,280 empty km
  over 3 h (about 5 km per vehicle; empty-km share 20.5% to 26.5%). At gamma = 1 it adds
  empty km and no service.
* **Decision interval** (E3): shorter is better here because W = 10 min is tight:
  MILP 89.3% (1 min), 88.7% (2), 83.2% (5), 64.9% (10). The MILP's edge over FCFS grows
  with batch size: +1.1, +1.7, +4.1, +3.9 points.
* **Fleet size** (E4, gamma = 0): MILP + repositioning with about 280 vehicles matches
  FCFS with 300 (linear interpolation between 250 and 300), roughly 6% fewer vehicles.
* **Computation**: mean solve time about 0.02 s per epoch, worst epoch 0.49 s over all
  355 runs. The largest models had 1,434 variables (near-stable, dt = 2) and 1,612
  (efficient, dt = 10), with at most 894 constraints, all within the Gurobi free licence
  limit of 2,000. Use the HKUST academic licence for bigger instances.

Figures: `figures/fig0_zone_imbalance.png` (where vehicles pile up under each gamma),
`fig1_policy_vs_gamma`, `fig2_hourly_service`, `fig3_stability_frontier`,
`fig4_decision_interval`, `fig5_fleet_size`, `fig6_repositioning_tradeoff`.

## Limitations to state in the report

Constant network speed (no congestion); one passenger per vehicle (no pooling);
myopic epoch matching (no anticipation of future requests in the matching itself);
stability evaluated within each epoch on the pruned graph (K = 8 nearest vehicles);
fares, costs and preferences are assumptions, not observed behaviour; 2% demand sample
with a 250-vehicle fleet, scaled together.

## Possible next steps

* Hong Kong scenario: a boundary-crossing surge at the Lo Wu / Futian checkpoint zones,
  or a rain scenario (lower speed), to test repositioning and stability under disruption.
* Driver-income fairness: add a term or constraint that balances cumulative earnings
  (profit Gini is already reported).
* Joint matching + repositioning in a single MILP, or a demand forecast that uses the
  known peak profile.
