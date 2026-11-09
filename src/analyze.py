"""Aggregate results/*.csv (mean and sd over seeds) and draw figures/*.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
# categorical palette, fixed order (dataviz reference palette, light mode)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
GLAB = {"000": "0", "025": "0.25", "050": "0.5", "075": "0.75", "100": "1"}


def label(r):
    if r["policy"] == "milp":
        return "MILP + repositioning" if r["reposition"] else "MILP (efficient)"
    return {"fcfs": "FCFS nearest driver", "greedy": "Greedy best pair",
            "stable": "Stable MILP", "nearstable": f"Near-stable rho={r['rho']}"}[r["policy"]]


def load(name):
    p = RES / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"gamma": str})
    df["gamma"] = df["gamma"].str.zfill(3)
    df["label"] = df.apply(label, axis=1)
    return df


METRICS = ["service_rate", "service_h2_peak", "wait_mean", "wait_p95", "pickup_km_mean",
           "utilisation", "earn_mean", "earn_gini", "empty_share", "vkt_total", "repos_km",
           "blocking_per_match", "blocking_pairs", "solve_mean_s", "solve_max_s",
           "nvars_max", "ncons_max", "runtime_s"]
ORDER = ["FCFS nearest driver", "Greedy best pair", "MILP (efficient)", "MILP + repositioning", "Stable MILP"]


def summarise(df, keys):
    g = df.groupby(keys)[METRICS]
    return g.mean().round(4).join(g.std().round(4), rsuffix="_sd").reset_index()


def line(ax, xs, df, xkey, ykey, labels, scale=1.0):
    for k, lab in enumerate(labels):
        d = df[df["label"] == lab].groupby(xkey)[ykey].agg(["mean", "std"]).reindex(xs)
        ax.errorbar(range(len(xs)) if xkey == "gamma" else xs, d["mean"] * scale, yerr=d["std"] * scale,
                    color=SLOT[k], marker=MARK[k], ms=6, lw=2, capsize=3, label=lab)


def e1():
    df = load("E1")
    if df is None: return
    summarise(df, ["gamma", "label"]).to_csv(RES / "E1_summary.csv", index=False)
    xs = list(GLAB)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (m, t, s) in zip(axs, [("service_rate", "Service rate (%)", 100),
                                    ("wait_mean", "Mean wait (min)", 1),
                                    ("empty_share", "Empty km share (%)", 100)]):
        line(ax, xs, df, "gamma", m, ORDER, s)
        ax.set_xticks(range(5)); ax.set_xticklabels([GLAB[g] for g in xs])
        ax.set_xlabel("OD balance gamma (0 = most imbalanced)"); ax.set_title(t, loc="left", color=INK)
    axs[0].legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "fig1_policy_vs_gamma.png"); plt.close(fig)

    # peak vs off-peak service at gamma=0.5
    d = df[df["gamma"] == "050"].groupby("label")[["service_h1_offpeak", "service_h2_peak", "service_h3_offpeak"]].mean().reindex(ORDER) * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.16
    for k, lab in enumerate(ORDER):
        ax.bar(np.arange(3) + (k - 2) * w, d.loc[lab], w - 0.02, color=SLOT[k], label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Hour 1 (off-peak)", "Hour 2 (peak, 2x demand)", "Hour 3 (off-peak)"])
    ax.set_ylabel("Service rate (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Service rate by hour, gamma = 0.5", loc="left", color=INK, pad=42)
    fig.savefig(FIG / "fig2_hourly_service.png"); plt.close(fig)


def e2():
    df = load("E2")
    if df is None: return
    df["rho_eff"] = np.where(df["policy"] == "milp", 0.0, np.where(df["policy"] == "stable", np.inf, df["rho"]))
    s = df.groupby(["gamma", "rho_eff"])[METRICS].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    s.reset_index().to_csv(RES / "E2_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, g in enumerate(["050", "000"]):
        d = s.loc[g].reset_index()
        ax.plot(d["blocking_per_match_mean"] * 100, d["service_rate_mean"] * 100, color=SLOT[k],
                marker=MARK[k], ms=7, lw=2, label=f"gamma = {GLAB[g]}")
        for _, r in d.iterrows():
            txt = "efficient" if r["rho_eff"] == 0 else ("stable" if np.isinf(r["rho_eff"]) else f"rho={r['rho_eff']:g}")
            if txt in ("efficient", "stable", "rho=0.2", "rho=1"):
                ax.annotate(txt, (r["blocking_per_match_mean"] * 100, r["service_rate_mean"] * 100),
                            textcoords="offset points", xytext=(4, -13), fontsize=8, color=INK2)
    ax.set_xlabel("Blocking pairs per 100 matches"); ax.set_ylabel("Service rate (%)")
    ax.set_ylim(57, 91)
    ax.set_title("Price of stability: efficiency vs blocking pairs", loc="left", color=INK)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig3_stability_frontier.png"); plt.close(fig)


def e3():
    df = load("E3")
    if df is None: return
    summarise(df, ["dt", "label"]).to_csv(RES / "E3_summary.csv", index=False)
    xs = sorted(df["dt"].unique()); labs = ["FCFS nearest driver", "MILP (efficient)"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (m, t, sc) in zip(axs, [("service_rate", "Service rate (%)", 100), ("wait_mean", "Mean wait (min)", 1),
                                     ("pickup_km_mean", "Mean pickup distance (km)", 1)]):
        for k, lab in enumerate(labs):
            d = df[df["label"] == lab].groupby("dt")[m].agg(["mean", "std"]).reindex(xs)
            ax.errorbar(xs, d["mean"] * sc, yerr=d["std"] * sc, color=SLOT[[0, 2][k]], marker=MARK[[0, 2][k]],
                        lw=2, ms=6, capsize=3, label=lab)
        ax.set_xticks(xs); ax.set_xlabel("Decision interval (min)"); ax.set_title(t, loc="left", color=INK)
    axs[0].legend(fontsize=8)
    fig.savefig(FIG / "fig4_decision_interval.png"); plt.close(fig)


def e4():
    df = load("E4")
    if df is None: return
    summarise(df, ["fleet", "label"]).to_csv(RES / "E4_summary.csv", index=False)
    xs = sorted(df["fleet"].unique())
    labs = ["FCFS nearest driver", "MILP (efficient)", "MILP + repositioning"]; slots = [0, 2, 3]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (m, t, sc) in zip(axs, [("service_rate", "Service rate (%)", 100), ("utilisation", "Driver utilisation (%)", 100)]):
        for k, lab in enumerate(labs):
            d = df[df["label"] == lab].groupby("fleet")[m].agg(["mean", "std"]).reindex(xs)
            ax.errorbar(xs, d["mean"] * sc, yerr=d["std"] * sc, color=SLOT[slots[k]], marker=MARK[slots[k]],
                        lw=2, ms=6, capsize=3, label=lab)
        ax.set_xlabel("Fleet size (vehicles)"); ax.set_title(t + ", gamma = 0", loc="left", color=INK)
    axs[0].legend(fontsize=8)
    fig.savefig(FIG / "fig5_fleet_size.png"); plt.close(fig)


def e5():
    df = load("E5")
    if df is None: return
    summarise(df, ["theta"]).to_csv(RES / "E5_summary.csv", index=False)
    base = load("E1")
    base = base[(base["gamma"] == "000") & (base["label"] == "MILP (efficient)")]
    pts = [(0.0, base["service_rate"].mean(), base["service_rate"].std(), "no repositioning")]
    for th, d in df.groupby("theta"):
        pts.append((d["repos_km"].mean(), d["service_rate"].mean(), d["service_rate"].std(), f"theta={th:g} km"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x, y, e, t = zip(*pts)
    ax.errorbar(x, np.array(y) * 100, yerr=np.array(e) * 100, color=SLOT[3], marker=MARK[3], ms=7, lw=2, capsize=3)
    for xi, yi, ti in zip(x, y, t):
        if ti in ("theta=1.5 km", "theta=2 km"):
            continue   # crowded near the origin; values are in E5_summary.csv
        off = (8, 6) if ti == "no repositioning" else (6, -12)
        ax.annotate(ti, (xi, yi * 100), textcoords="offset points", xytext=off, fontsize=8, color=INK2)
    ax.set_xlabel("Total repositioning distance over 3 h (km, 250 vehicles)"); ax.set_ylabel("Service rate (%)")
    ax.set_title("Repositioning: service gained vs empty km spent, gamma = 0", loc="left", color=INK)
    fig.savefig(FIG / "fig6_repositioning_tradeoff.png"); plt.close(fig)


def network_map():
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from data import load_network, load_requests, make_zones
    _, C = load_network(); lab, cen = make_zones(16, 3010)
    X = (C - C.min(0)) / 1000
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, constrained_layout=True)
    nets = {}
    for g in ["000", "050", "100"]:
        r = load_requests(g)
        nets[g] = np.bincount(lab[r["d"]], minlength=16) - np.bincount(lab[r["o"]], minlength=16)
    v = max(np.abs(n).max() for n in nets.values())   # one shared scale for all panels
    for ax, g in zip(axs, nets):
        sc = ax.scatter(X[:, 0], X[:, 1], c=nets[g][lab], cmap="RdBu_r", vmin=-v, vmax=v, s=3)
        ax.set_title(f"gamma = {GLAB[g]}", loc="left", color=INK)
        ax.set_xlabel("km"); ax.set_aspect("equal"); ax.grid(False)
    axs[0].set_ylabel("km")
    cb = fig.colorbar(sc, ax=axs, shrink=0.85)
    cb.set_label("Net vehicle inflow per zone, 3 h\n(drop-offs minus pickups)")
    fig.savefig(FIG / "fig0_zone_imbalance.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5, network_map):
        f()
    print("done")
