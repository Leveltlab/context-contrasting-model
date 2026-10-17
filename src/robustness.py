"""Deterministic one-at-a-time sweeps of the same six YAML models."""

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from .model import CCNeuron
from plotting.plot import COLORS, save
from .simulate import ROOT, load_config, run_phase, stimuli, train

WEIGHTS = ("w_ff", "w_fb", "w_lat", "w_pv_lat", "W_pv")
SCALARS = (
    "lr_ff", "lr_fb", "lr_lat", "pyc_decay", "pv_decay",
    "apical_drive_threshold", "apical_gain_strength", "apical_gain_k",
    "apical_gain_threshold", "baseline_drive_sigma", "divisive_gain",
    "pv_noise_sigma", "ff_accumulator_alpha_factor",
    "ff_accumulator_power", "ff_accumulator_scale",
)
FACTORS = (.05, .1, .25, .5, .75, 1, 1.25, 1.5, 2, 3, 5)
WEIGHT_VALUES = (.01, .02, .05, .1, .2, .3, .5, .75, 1)  # absolute, within the allowed [0, 1]
RATES = ("pyc_decay", "pv_decay")  # leak fractions per step, capped at 1
LABELS = {
    "w_ff": "PyC FF vector", "w_fb": "PyC FB vector", "w_lat": "PV to PyC", "w_pv_lat": "PyC to PV",
    "W_pv": "PV FF vector", "lr_ff": "FF learning rate", "lr_fb": "FB learning rate",
    "lr_lat": "Lateral learning rate", "pyc_decay": "PyC leak rate", "pv_decay": "PV leak rate",
    "apical_drive_threshold": "Apical drive threshold", "apical_gain_strength": "Apical gain strength",
    "apical_gain_k": "Apical gain steepness", "apical_gain_threshold": "Apical gain threshold",
    "baseline_drive_sigma": "PyC baseline noise s.d.", "divisive_gain": "Divisive gain",
    "pv_noise_sigma": "PV noise s.d.", "ff_accumulator_alpha_factor": "Accumulator alpha factor",
    "ff_accumulator_power": "Accumulator power", "ff_accumulator_scale": "Accumulator scale",
}
HEATMAPS = {
    "scalar": ("scalar_parameter", "Dynamics and plasticity parameter robustness"),
    "coordinate": ("initial_weight_coordinate", "Individual initial-weight robustness"),
    "family": ("initial_weight_family", "Coherent initial-weight-family robustness"),
}


def design(cell, parameter=None, factors=None):
    """Each tuple changes one scalar, one weight, or one whole weight family."""
    yield ("baseline", "baseline", -1, 1.)
    p = cell["parameters"]
    factors = FACTORS if factors is None else tuple(factors)
    for name in (*WEIGHTS, *SCALARS):
        if parameter is not None and name != parameter:
            continue
        if name in WEIGHTS:
            for i, base in enumerate(p[name]):
                for value in sorted(set(WEIGHT_VALUES) | {base}):
                    yield ("coordinate", name, i, value)
            for factor in factors:
                yield ("family", name, -1, factor)
        else:
            for value in sorted(set(min(f * p[name], 1) if name in RATES else f * p[name] for f in factors)):
                yield ("scalar", name, -1, value)


def run_point(cell, protocol, change, seed=None):
    torch.set_num_threads(1)
    kind, parameter, coordinate, value = change
    p = copy.deepcopy(cell["parameters"])
    if seed is not None:
        p["seed"] = seed
    if kind == "coordinate":
        p[parameter][coordinate] = value
    elif kind == "family":
        p[parameter] = [min(w * value, 1) for w in p[parameter]]
    elif kind == "scalar":
        p[parameter] = value
    model = CCNeuron(p)
    n, trials = protocol["steps_per_trial"], protocol["test_trials"]
    images = (0, 1) if cell["source"] == "familiar" else (2,)
    responses = {}
    mean = std = np.nan
    for phase in ("naive", "expert"):
        if phase == "expert":
            torch.manual_seed(200_000 + p["seed"])
            train(model, protocol)
        # Reuse measurement noise only across the naive/expert probes of this point.
        torch.manual_seed(100_000 + p["seed"])
        traces = {}
        for image in images:
            x, c = stimuli([image] * trials, protocol, post=False)
            for response in ("NO", "O"):
                traces[image, response] = run_phase(model, x if response == "NO" else torch.zeros_like(x), c).reshape(trials, n)
        if phase == "naive":
            baseline = np.concatenate([v[:, :3 * n // 4].ravel() for v in traces.values()])
            mean, std = baseline.mean(), baseline.std(ddof=1)
        for response in ("NO", "O"):
            activity = np.mean([v[:, 3 * n // 4:].mean() for (_, r), v in traces.items() if r == response])
            responses[f"{phase}_{response}"] = (activity - mean) / std if std > 1e-12 else np.nan
    dno = responses["expert_NO"] - responses["naive_NO"]
    do = responses["expert_O"] - responses["naive_O"]
    angle = np.arctan2(do, dno)
    direction = ("+O axis" if np.pi / 4 <= angle < 3 * np.pi / 4 else
                 "-NO axis" if angle >= 3 * np.pi / 4 or angle < -3 * np.pi / 4 else
                 "-O axis" if -3 * np.pi / 4 <= angle < -np.pi / 4 else "+NO axis")
    return dict(cell_id=cell["id"], source=cell["source"], sector=cell["sector"], seed=p["seed"], kind=kind,
                parameter=parameter, coordinate=coordinate, value=value, **responses,
                delta_NO=dno, delta_O=do, magnitude=np.hypot(dno, do), direction=direction,
                naive_baseline_mean=mean, naive_baseline_std=std)


def plot(results_csv, configs, directory):
    """Retention heatmaps, sensitivity ranking, baseline vectors and sweep curves from the saved run."""
    directory.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_csv)
    cells = {c["id"]: c for c in load_config(configs)[0]}
    names = {i: f"{c['source'].capitalize()} {c['sector'].replace(' axis', '')}" for i, c in cells.items()}
    baseline = results.loc[results.kind.eq("baseline")].set_index("cell_id")
    results = results.loc[results.kind.ne("baseline")].copy()
    results["label"] = [LABELS[p] + (f" [{i + 1}]" if k == "coordinate" else "")
                        for k, p, i in zip(results.kind, results.parameter, results.coordinate)]
    # The YAML value marks each curve; families are scaled by factors, so theirs is 1.
    results["reference"] = [1. if k == "family" else cells[c]["parameters"][p][i] if k == "coordinate"
                            else cells[c]["parameters"][p]
                            for k, p, i, c in zip(results.kind, results.parameter, results.coordinate, results.cell_id)]
    shift = np.hypot(results.delta_NO - results.cell_id.map(baseline.delta_NO),
                     results.delta_O - results.cell_id.map(baseline.delta_O))
    results["distance"] = shift / results.cell_id.map(baseline.magnitude)
    summary = results.groupby(["kind", "label", "cell_id"]).agg(
        retained=("retained", "mean"), distance=("distance", "max")).reset_index()
    cmap = plt.get_cmap("RdYlGn")
    for kind, (name, title) in HEATMAPS.items():
        table = summary.loc[summary.kind.eq(kind)].pivot(index="cell_id", columns="label", values="retained")
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 2, 6), layout="constrained")
        image = ax.imshow(table, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(table)), [names[c] for c in table.index])
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=.8, label="Fraction retaining direction and >=25% baseline magnitude")
        save(fig, directory / f"{name}_retention_heatmap")
    # Least retained first; ties broken by the largest displacement from the baseline vector.
    worst = summary.sort_values(["retained", "distance"], ascending=[True, False]).head(30)
    fig, ax = plt.subplots(figsize=(11, 9.8), layout="constrained")
    ax.barh(range(len(worst)), worst.retained, color=cmap(worst.retained.to_numpy()))
    ax.set_yticks(range(len(worst)), [f"{names[c]} | {l}" for c, l in zip(worst.cell_id, worst.label)], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(.5, color="0.3", ls="--", lw=1)
    ax.set(xlim=(0, 1), xlabel="Robust fraction across tested values",
           title="Thirty most sensitive cell-parameter combinations")
    save(fig, directory / "most_sensitive_parameter_combinations")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.3), layout="constrained")
    for ax, source in zip(axes, ("familiar", "novel")):
        rows = baseline.loc[baseline.source.eq(source)]
        limit = np.ceil(1.05 * rows[["delta_NO", "delta_O"]].abs().to_numpy().max())
        ax.axhline(0, color="0.8", lw=1)
        ax.axvline(0, color="0.8", lw=1)
        for sign in (1, -1):
            ax.plot([-limit, limit], [-sign * limit, sign * limit], color="0.88", ls="--", lw=.8)
        for _, row in rows.iterrows():
            ax.arrow(0, 0, row.delta_NO, row.delta_O, width=.01 * limit, head_width=.06 * limit,
                     head_length=.1 * limit, length_includes_head=True, color=COLORS[row.sector])
            ax.text(row.delta_NO + .03 * limit, row.delta_O, row.sector.replace(" axis", ""), va="center")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", title=source.capitalize(),
               xlabel=r"$\Delta R_{NO}$", ylabel=r"$\Delta R_O$")
    fig.suptitle("Baseline transitions under the robustness protocol")
    save(fig, directory / "baseline_transition_vectors")
    for (kind, parameter, coordinate), sweep in results.groupby(["kind", "parameter", "coordinate"]):
        fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.9), layout="constrained")
        for ax, (cell_id, points) in zip(axes.flat, sweep.groupby("cell_id")):
            points = points.sort_values("value")
            failed = points.loc[~points.retained]
            for column, color, label in (("delta_NO", "k", r"$\Delta R_{NO}$"), ("delta_O", "red", r"$\Delta R_O$")):
                ax.plot(points.value, points[column], color=color, marker="o", ms=4, label=label)
                ax.scatter(failed.value, failed[column], marker="x", s=45, color="#8c3a0c", zorder=5)
            ax.axvline(points.reference.iloc[0], color=COLORS["+NO axis"], ls="--", lw=1)
            ax.axhline(0, color="0.8", lw=.8, zorder=0)
            ax.set(title=names[cell_id], xlabel="Parameter value", ylabel="Expert - naive response")
        axes.flat[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"One-at-a-time robustness: {sweep.label.iloc[0]}\n"
                     "blue dashed = YAML value; crosses = robustness failure")
        (directory / "curves" / kind).mkdir(parents=True, exist_ok=True)
        save(fig, directory / "curves" / kind / (parameter + (f"_{coordinate}" if coordinate >= 0 else "")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model_robustness.csv")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures" / "robustness")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--test-trials", type=int, default=10)
    parser.add_argument("--parameter", choices=(*WEIGHTS, *SCALARS))
    parser.add_argument("--factors", type=float, nargs="+", help="Multiples of the YAML value for scalars and weight families (default: 0.05 to 5).")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figures from a saved run.")
    args = parser.parse_args()
    if args.plot_only:
        return plot(args.output, args.configs, args.figures)
    cells, protocol = load_config(args.configs)
    protocol["test_trials"] = args.test_trials
    changes = [(cell, change) for cell in cells for change in design(cell, args.parameter, args.factors)]
    jobs = [(cell, change, cell["parameters"]["seed"] if change[0] == "baseline" else 1_000_000 + i)
            for i, (cell, change) in enumerate(changes)]
    print(f"Running {len(jobs)} points", flush=True)
    results = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(run_point)(cell, protocol, change, seed)
                                               for cell, change, seed in jobs))
    baseline = results.loc[results.kind.eq("baseline")].set_index("cell_id").magnitude
    results["relative_magnitude"] = results.magnitude / results.cell_id.map(baseline)
    results["retained"] = results.direction.eq(results.sector) & results.relative_magnitude.ge(.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    plot(args.output, args.configs, args.figures)

if __name__ == "__main__":
    main()
