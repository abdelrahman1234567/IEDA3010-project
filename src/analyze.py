"""Summarise experiment CSVs and generate the original figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SLOT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GLAB = {"000": "0", "025": "0.25", "050": "0.5", "075": "0.75", "100": "1"}

METRICS = [
    "service_rate",
    "service_h2_peak",
    "wait_mean",
    "wait_p95",
    "pickup_km_mean",
    "utilisation",
    "earn_mean",
    "earn_gini",
    "empty_share",
    "vkt_total",
    "repos_km",
    "blocking_per_match",
    "blocking_pairs",
    "solve_mean_s",
    "solve_max_s",
    "nvars_max",
    "ncons_max",
    "runtime_s",
]
ORDER = [
    "FCFS nearest driver",
    "Greedy best pair",
    "MILP (efficient)",
    "MILP + repositioning",
    "Stable MILP",
]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def label(row):
    if row["policy"] == "milp":
        return "MILP + repositioning" if row["reposition"] else "MILP (efficient)"

    labels = {
        "fcfs": "FCFS nearest driver",
        "greedy": "Greedy best pair",
        "stable": "Stable MILP",
        "nearstable": f"Near-stable rho={row['rho']}",
    }
    return labels[row["policy"]]


def load(name):
    path = RES / f"{name}.csv"
    if not path.exists():
        return None

    data = pd.read_csv(path, dtype={"gamma": str})
    data["gamma"] = data["gamma"].str.zfill(3)
    data["label"] = data.apply(label, axis=1)
    return data


def summarise(df, keys):
    grouped = df.groupby(keys)[METRICS]
    means = grouped.mean().round(4)
    deviations = grouped.std().round(4)
    return means.join(deviations, rsuffix="_sd").reset_index()


def line(ax, xs, df, xkey, ykey, labels, scale=1.0):
    for style_index, name in enumerate(labels):
        statistics = (
            df[df["label"] == name].groupby(xkey)[ykey].agg(["mean", "std"]).reindex(xs)
        )
        x_values = range(len(xs)) if xkey == "gamma" else xs
        ax.errorbar(
            x_values,
            statistics["mean"] * scale,
            yerr=statistics["std"] * scale,
            color=SLOT[style_index],
            marker=MARK[style_index],
            ms=6,
            lw=2,
            capsize=3,
            label=name,
        )


def _save_figure(figure, filename):
    figure.savefig(FIG / filename)
    plt.close(figure)


def _plot_hourly_service(data):
    hourly_columns = ["service_h1_offpeak", "service_h2_peak", "service_h3_offpeak"]
    hourly_service = (
        data[data["gamma"] == "050"]
        .groupby("label")[hourly_columns]
        .mean()
        .reindex(ORDER)
        * 100
    )

    figure, axis = plt.subplots(figsize=(7, 3.6))
    bar_width = 0.16
    for index, name in enumerate(ORDER):
        positions = np.arange(3) + (index - 2) * bar_width
        axis.bar(
            positions,
            hourly_service.loc[name],
            bar_width - 0.02,
            color=SLOT[index],
            label=name,
        )

    axis.set_xticks(range(3))
    axis.set_xticklabels(
        ["Hour 1 (off-peak)", "Hour 2 (peak, 2x demand)", "Hour 3 (off-peak)"]
    )
    axis.set_ylabel("Service rate (%)")
    axis.set_ylim(0, 105)
    axis.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axis.set_title("Service rate by hour, gamma = 0.5", loc="left", color=INK, pad=42)
    _save_figure(figure, "fig2_hourly_service.png")


def e1():
    data = load("E1")
    if data is None:
        return

    summarise(data, ["gamma", "label"]).to_csv(RES / "E1_summary.csv", index=False)
    gammas = list(GLAB)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ("service_rate", "Service rate (%)", 100),
        ("wait_mean", "Mean wait (min)", 1),
        ("empty_share", "Empty km share (%)", 100),
    ]
    for axis, (metric, title, scale) in zip(axes, panels):
        line(axis, gammas, data, "gamma", metric, ORDER, scale)
        axis.set_xticks(range(5))
        axis.set_xticklabels([GLAB[gamma] for gamma in gammas])
        axis.set_xlabel("OD balance gamma (0 = most imbalanced)")
        axis.set_title(title, loc="left", color=INK)

    axes[0].legend(loc="lower right", fontsize=8)
    _save_figure(figure, "fig1_policy_vs_gamma.png")
    _plot_hourly_service(data)


def _stability_label(rho):
    if rho == 0:
        return "efficient"
    if np.isinf(rho):
        return "stable"
    return f"rho={rho:g}"


def e2():
    data = load("E2")
    if data is None:
        return

    data["rho_eff"] = np.where(
        data["policy"] == "milp",
        0.0,
        np.where(data["policy"] == "stable", np.inf, data["rho"]),
    )
    statistics = data.groupby(["gamma", "rho_eff"])[METRICS].agg(["mean", "std"])
    statistics.columns = ["_".join(column) for column in statistics.columns]
    statistics.reset_index().to_csv(RES / "E2_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(6.5, 4))
    for style_index, gamma in enumerate(["050", "000"]):
        group = statistics.loc[gamma].reset_index()
        blocking = group["blocking_per_match_mean"] * 100
        service = group["service_rate_mean"] * 100
        axis.plot(
            blocking,
            service,
            color=SLOT[style_index],
            marker=MARK[style_index],
            ms=7,
            lw=2,
            label=f"gamma = {GLAB[gamma]}",
        )
        for _, row in group.iterrows():
            annotation = _stability_label(row["rho_eff"])
            if annotation in ("efficient", "stable", "rho=0.2", "rho=1"):
                axis.annotate(
                    annotation,
                    (
                        row["blocking_per_match_mean"] * 100,
                        row["service_rate_mean"] * 100,
                    ),
                    textcoords="offset points",
                    xytext=(4, -13),
                    fontsize=8,
                    color=INK2,
                )

    axis.set_xlabel("Blocking pairs per 100 matches")
    axis.set_ylabel("Service rate (%)")
    axis.set_ylim(57, 91)
    axis.set_title(
        "Price of stability: efficiency vs blocking pairs", loc="left", color=INK
    )
    axis.legend(fontsize=8)
    _save_figure(figure, "fig3_stability_frontier.png")


def e3():
    data = load("E3")
    if data is None:
        return

    summarise(data, ["dt", "label"]).to_csv(RES / "E3_summary.csv", index=False)
    intervals = sorted(data["dt"].unique())
    policies = ["FCFS nearest driver", "MILP (efficient)"]
    panels = [
        ("service_rate", "Service rate (%)", 100),
        ("wait_mean", "Mean wait (min)", 1),
        ("pickup_km_mean", "Mean pickup distance (km)", 1),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, (metric, title, scale) in zip(axes, panels):
        for style_index, name in enumerate(policies):
            statistics = (
                data[data["label"] == name]
                .groupby("dt")[metric]
                .agg(["mean", "std"])
                .reindex(intervals)
            )
            colour_slot = [0, 2][style_index]
            axis.errorbar(
                intervals,
                statistics["mean"] * scale,
                yerr=statistics["std"] * scale,
                color=SLOT[colour_slot],
                marker=MARK[colour_slot],
                lw=2,
                ms=6,
                capsize=3,
                label=name,
            )
        axis.set_xticks(intervals)
        axis.set_xlabel("Decision interval (min)")
        axis.set_title(title, loc="left", color=INK)

    axes[0].legend(fontsize=8)
    _save_figure(figure, "fig4_decision_interval.png")


def e4():
    data = load("E4")
    if data is None:
        return

    summarise(data, ["fleet", "label"]).to_csv(RES / "E4_summary.csv", index=False)
    fleet_sizes = sorted(data["fleet"].unique())
    policies = ["FCFS nearest driver", "MILP (efficient)", "MILP + repositioning"]
    style_slots = [0, 2, 3]
    panels = [
        ("service_rate", "Service rate (%)", 100),
        ("utilisation", "Driver utilisation (%)", 100),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, (metric, title, scale) in zip(axes, panels):
        for style_index, name in enumerate(policies):
            statistics = (
                data[data["label"] == name]
                .groupby("fleet")[metric]
                .agg(["mean", "std"])
                .reindex(fleet_sizes)
            )
            colour_slot = style_slots[style_index]
            axis.errorbar(
                fleet_sizes,
                statistics["mean"] * scale,
                yerr=statistics["std"] * scale,
                color=SLOT[colour_slot],
                marker=MARK[colour_slot],
                lw=2,
                ms=6,
                capsize=3,
                label=name,
            )
        axis.set_xlabel("Fleet size (vehicles)")
        axis.set_title(title + ", gamma = 0", loc="left", color=INK)

    axes[0].legend(fontsize=8)
    _save_figure(figure, "fig5_fleet_size.png")


def e5():
    data = load("E5")
    if data is None:
        return

    summarise(data, ["theta"]).to_csv(RES / "E5_summary.csv", index=False)
    baseline = load("E1")
    baseline = baseline[
        (baseline["gamma"] == "000") & (baseline["label"] == "MILP (efficient)")
    ]
    points = [
        (
            0.0,
            baseline["service_rate"].mean(),
            baseline["service_rate"].std(),
            "no repositioning",
        )
    ]
    for theta, group in data.groupby("theta"):
        points.append(
            (
                group["repos_km"].mean(),
                group["service_rate"].mean(),
                group["service_rate"].std(),
                f"theta={theta:g} km",
            )
        )

    x_values, rates, errors, labels = zip(*points)
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    axis.errorbar(
        x_values,
        np.array(rates) * 100,
        yerr=np.array(errors) * 100,
        color=SLOT[3],
        marker=MARK[3],
        ms=7,
        lw=2,
        capsize=3,
    )
    for distance, rate, name in zip(x_values, rates, labels):
        if name in ("theta=1.5 km", "theta=2 km"):
            continue
        offset = (8, 6) if name == "no repositioning" else (6, -12)
        axis.annotate(
            name,
            (distance, rate * 100),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            color=INK2,
        )
    axis.set_xlabel("Total repositioning distance over 3 h (km, 250 vehicles)")
    axis.set_ylabel("Service rate (%)")
    axis.set_title(
        "Repositioning: service gained vs empty km spent, gamma = 0",
        loc="left",
        color=INK,
    )
    _save_figure(figure, "fig6_repositioning_tradeoff.png")


def network_map():
    from data import load_network, load_requests, make_zones

    _, coordinates = load_network()
    zone_labels, _ = make_zones(16, 3010)
    points = (coordinates - coordinates.min(axis=0)) / 1000

    net_inflow = {}
    for gamma in ("000", "050", "100"):
        requests = load_requests(gamma)
        dropoffs = np.bincount(zone_labels[requests["d"]], minlength=16)
        pickups = np.bincount(zone_labels[requests["o"]], minlength=16)
        net_inflow[gamma] = dropoffs - pickups

    limit = max(np.abs(values).max() for values in net_inflow.values())
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.4), sharey=True, constrained_layout=True
    )
    for axis, (gamma, inflow) in zip(axes, net_inflow.items()):
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            c=inflow[zone_labels],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            s=3,
        )
        axis.set_title(f"gamma = {GLAB[gamma]}", loc="left", color=INK)
        axis.set_xlabel("km")
        axis.set_aspect("equal")
        axis.grid(False)

    axes[0].set_ylabel("km")
    colourbar = figure.colorbar(scatter, ax=axes, shrink=0.85)
    colourbar.set_label("Net vehicle inflow per zone, 3 h\n(drop-offs minus pickups)")
    _save_figure(figure, "fig0_zone_imbalance.png")


def main():
    for analysis in (e1, e2, e3, e4, e5, network_map):
        analysis()
    print("done")


if __name__ == "__main__":
    main()
