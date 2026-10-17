"""Plot traces and transition vectors using only ground_truth.csv and model.csv."""

import argparse
from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
SECTORS = ("-NO axis", "+O axis", "+NO axis")
TRACE_ROWS = ("Overall", *SECTORS)
DATA_SECTORS = (*SECTORS, "-O axis")
COLORS = {"+NO axis": "#3159a7", "+O axis": "#ef202f",
          "-NO axis": "#e68600", "-O axis": "#16843e"}
SECTOR_SHORT = {sector: sector.replace(" axis", "") for sector in COLORS}
SOURCES = ("familiar", "novel")
LABELS = {"naive": "Naive", "expert": "Expert", "expert_no_fb": "FB silencing",
          "expert_no_lat": "PV silencing", "expert_no_fb_no_lat": "FB & PV silencing"}
MODEL_VERSIONS = {"core": ("naive", "expert"),
                  "ablations": tuple(LABELS)[2:], "all": tuple(LABELS)}


def use_myriad_pro():
    adobe_fonts = Path("/Library/Application Support/Adobe/Acrobat/DC/WebResources/Resource0/OWP/default/fonts")
    for name in ("MyriadPro-Regular.otf", "MyriadPro-Bold.otf", "MyriadPro-Light.otf"):
        path = adobe_fonts / name
        if path.exists():
            font_manager.fontManager.addfont(path)
    font_manager.findfont("Myriad Pro", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Myriad Pro", "font.size": 12,
                         "axes.titlesize": 18, "axes.labelsize": 18,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "legend.title_fontsize": 12,
                         "svg.fonttype": "none"})


def save(fig, path):
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def observation_columns(table):
    return ["observation_id"] + (["seed"] if "seed" in table and table.seed.notna().any() else [])


def summarize_traces(samples, keys):
    summary = samples.groupby(keys)["response"].agg(response="mean", sd="std", n="count").reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    return summary


def trace_observations(table):
    traces = table.loc[table.record_type.eq("trace")]
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    # Images are repeated measures, not independent neurons or seed replicates.
    return traces.groupby(keys + observation_columns(traces))["response"].mean().reset_index()


def trace_summary(table, overall_table=None, weights=None):
    observations = trace_observations(table.loc[table.sector.isin(SECTORS)])
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    sectors = summarize_traces(observations, keys)
    overall = average_traces(table if overall_table is None else overall_table, weights)
    overall["sector"] = "Overall"
    return pd.concat([overall, sectors], ignore_index=True)


def trace_limits(summary):
    limits = {}
    for sector in TRACE_ROWS:
        frame = summary.loc[summary.sector.eq(sector)]
        lower = (frame.response - frame.sd.fillna(0)).min()
        upper = (frame.response + frame.sd.fillna(0)).max()
        margin = max((upper - lower) * 0.08, 0.1)
        limits[sector] = (lower - margin, max(upper + margin, lower - margin + 1.5))
    return limits


def plot_traces(summary, source, path, uncertainty="sd", conditions=None, limits=None, rows=TRACE_ROWS):
    conditions = tuple(LABELS) if conditions is None else conditions
    limits = trace_limits(summary) if limits is None else limits
    trace_width = 1.2
    fig, axes = plt.subplots(len(rows), len(conditions),
                             figsize=(2.2 * len(conditions), 1.7 if len(rows) == 1 else 6.3),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    for row, sector in enumerate(rows):
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            ax.axvspan(0, 1, color="0.92")
            frame = summary.loc[summary.sector.eq(sector) & summary.source.eq(source) &
                                summary.condition_key.eq(condition)]
            for response, color in (("NO", "black"), ("O", "red")):
                line = frame.loc[frame.response_type.eq(response)]
                ax.plot(line.time_seconds, line.response, color=color, lw=trace_width, label=response)
                ax.fill_between(line.time_seconds, line.response - line[uncertainty],
                                line.response + line[uncertainty], color=color, alpha=0.18, linewidth=0)
            ax.set(xlim=(-1, 3), ylim=limits[sector])
            if row == 0:
                ax.set_title(LABELS[condition])
            ax.set_axis_off()
            if column == 0:
                ax.text(-0.05, 0.5, sector.replace(" axis", ""), transform=ax.transAxes,
                        ha="right", va="center", color=COLORS.get(sector, "black"))
                lower, upper = limits[sector]
                x, y = -0.8, lower + 0.02 * (upper - lower)
                ax.plot([x, x + 1], [y, y], color="black", lw=trace_width, clip_on=False)
                ax.plot([x, x], [y, y + 1], color="black", lw=trace_width, clip_on=False)
                ax.text(x + 0.5, y, "1 s", ha="center", va="top", fontsize=8)
                ax.text(x + 0.06, y + 1, "1 S.D.", ha="left", va="bottom", fontsize=8)
    axes[0, -1].legend(frameon=False)
    save(fig, path)


def empirical_sector_weights(data):
    vectors = data.loc[data.record_type.eq("transition") & data.sector.isin(SECTORS)]
    counts = vectors.groupby(["source", "sector"])["observation_id"].nunique()
    return counts / counts.groupby(level="source").transform("sum")


def average_traces(table, weights=None):
    observations = trace_observations(table)
    keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
    output_keys = [key for key in keys if key != "sector"]
    if weights is None:
        return summarize_traces(observations, output_keys)
    # Preserve covariance across templates sharing a replicate seed: weight first,
    # then summarize complete seed-level population traces, never pool templates.
    if "seed" not in observations:
        observations["seed"] = 0
    sectors = observations.groupby(keys + ["seed"])["response"].mean().reset_index()
    sectors["weight"] = [weights.loc[(source, sector)] for source, sector in
                         zip(sectors.source, sectors.sector)]
    sectors["response"] *= sectors["weight"]
    counts = sectors.groupby(output_keys + ["seed"])["sector"].nunique()
    expected = counts.index.get_level_values("source").map(weights.groupby(level="source").size())
    if not np.array_equal(counts.to_numpy(), expected.to_numpy()):
        raise ValueError("Weighted model traces require every sector for each seed/time/condition")
    samples = sectors.groupby(output_keys + ["seed"])["response"].sum().reset_index()
    return summarize_traces(samples, output_keys)


def model_transition_vectors(model, condition):
    traces = model.loc[model.record_type.eq("trace") & model.time_seconds.between(0, 1)]
    keys = ["source", "sector"] + observation_columns(traces) + ["condition_key", "response_type"]
    responses = traces.groupby(keys)["response"].mean().unstack(["condition_key", "response_type"])
    vectors = responses.index.to_frame(index=False)
    for response in ("NO", "O"):
        vectors[f"delta_{response}"] = (responses[(condition, response)] -
                                         responses[("naive", response)]).to_numpy()
    return vectors


def overall_transition(vectors, source, weights=None):
    frame = vectors.loc[vectors.source.eq(source)]
    if weights is None:
        return frame[["delta_NO", "delta_O"]].mean().to_numpy()
    means = frame.groupby("sector")[["delta_NO", "delta_O"]].mean()
    return sum(weights.loc[(source, sector)] * means.loc[sector].to_numpy() for sector in SECTORS)


def plot_vectors(data, model, source, path, uncertainty="sd", condition="expert"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), #layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    weights = empirical_sector_weights(data)
    data_vectors = data.loc[data.record_type.eq("transition")]
    model_vectors = model_transition_vectors(model, condition)
    for column, (vectors, label) in enumerate(((data_vectors, "Data"),
                                                (model_vectors, f"Model ({LABELS[condition]})"))):
        ax = axes[0, column]
        source_vectors = vectors.loc[vectors.source.eq(source)]
        for sector in DATA_SECTORS if label == "Data" else SECTORS:
            color = COLORS[sector]
            points = source_vectors.loc[source_vectors.sector.eq(sector),
                                         ["delta_NO", "delta_O"]].to_numpy()
            if not len(points):
                continue
            center = points.mean(axis=0)
            # Neuron-to-neuron (data) or seed-to-seed (model) covariance.
            if len(points) > 1:
                covariance = np.cov(points, rowvar=False)
                if uncertainty == "sem":
                    covariance /= len(points)
                values, directions = np.linalg.eigh(covariance)
                angle = np.degrees(np.arctan2(directions[1, -1], directions[0, -1]))
                width, height = 2 * np.sqrt(np.maximum(values[::-1], 0))
                ax.add_patch(Ellipse(center, width, height, angle=angle, color=color, alpha=0.18))
            ax.annotate("", xy=center, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=14))
            ax.plot([], [], color=color, label=sector.replace(" axis", ""))
        overall = overall_transition(vectors, source, None if label == "Data" else weights)
        ax.annotate("", xy=overall, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=3.2, mutation_scale=16))
        ax.plot([], [], color="black", lw=3.2, label="Overall transition")
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set(xlim=(-1.4, 1.4), ylim=(-.3, 1.0), #aspect=1.0,
               title=label, xlabel="Delta NO", ylabel="Delta O")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, path)


def transition_cosine_sim(transitions_A, transitions_B):
    dot = np.sum(transitions_A * transitions_B, axis=-1)
    norm = np.linalg.norm(transitions_A, axis=-1) * np.linalg.norm(transitions_B, axis=-1)
    return np.divide(dot, norm, out=np.full_like(dot, np.nan, dtype=float), where=norm != 0)


def transition_metric(observed, predicted, metric="cosine"):
    if metric == "cosine":
        return transition_cosine_sim(observed, predicted)
    if metric == "dot":
        return np.sum(observed * predicted, axis=-1)
    if metric in ("l2", "distance"):
        return np.linalg.norm(observed - predicted, axis=-1)
    raise ValueError("metric must be 'cosine', 'dot', or 'l2'")


def transition_similarities(data, model, metric="cosine"):
    data_vectors = data.loc[data.record_type.eq("transition")]
    weights = empirical_sector_weights(data)
    rows = []
    for condition in tuple(LABELS)[1:]:
        model_vectors = model_transition_vectors(model, condition)
        for source in SOURCES:
            for sector in SECTORS:
                observed = data_vectors.loc[data_vectors.source.eq(source) &
                                             data_vectors.sector.eq(sector),
                                             ["delta_NO", "delta_O"]].to_numpy()
                predicted = model_vectors.loc[model_vectors.source.eq(source) &
                                               model_vectors.sector.eq(sector),
                                               ["delta_NO", "delta_O"]].mean().to_numpy()
                for similarity in transition_metric(observed, predicted, metric):
                    rows.append((source, LABELS[condition], sector.replace(" axis", ""),
                                 float(similarity)))
            observed = overall_transition(data_vectors, source)
            predicted = overall_transition(model_vectors, source, weights)
            rows.append((source, LABELS[condition], "Overall",
                         float(transition_metric(observed, predicted, metric))))

    return pd.DataFrame(rows, columns=["source", "condition", "sector", "similarity"])


def cosine_similarities(data, model):
    return transition_similarities(data, model)


def plot_similarities(data, model, source, path, errorbars=False, 
                      metric="cosine", uncertainty = 'sd'):
    order = list(LABELS.values())[1:]
    hues = ["Overall"] + [sector.replace(" axis", "") for sector in SECTORS]
    palette = {sector.replace(" axis", ""): color for sector, color in COLORS.items()}
    palette["Overall"] = "black"
    similarities = transition_similarities(data, model, metric)
    fig, ax = plt.subplots(figsize=(4, 3.3), layout="constrained")
    for value in (-1, 0, 1) if metric == "cosine" else (0,):
        ax.axhline(value, color="0.7", lw=0.7, linestyle="--")
    sns.pointplot(data=similarities.loc[similarities.source.eq(source)], x="condition",
                  y="similarity", hue="sector", order=order, hue_order=hues,
                  palette=palette, dodge=0.45, errorbar=uncertainty if errorbars else None, ax=ax,
                  linestyle="none")
    labels = {"cosine": "Cosine similarity", "dot": "Dot product", "l2": "Euclidean distance"}
    ax.set(xlabel=None, ylabel=labels[metric])
    if metric == "cosine":
        ax.set_ylim(-1.2, 1.2)
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title=None)
    save(fig, path)


def plot_transition_proportions(data, path):
    transitions = data.loc[data.record_type.eq("transition")]
    counts = transitions.groupby(["source", "sector"])["observation_id"].nunique().unstack(fill_value=0)
    proportions = counts.reindex(index=SOURCES, columns=DATA_SECTORS, fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(4.6, 4), layout="constrained")
    bottom = np.zeros(len(SOURCES))
    for sector in reversed(DATA_SECTORS):
        values = proportions[sector].to_numpy()
        ax.bar(range(len(SOURCES)), values, bottom=bottom, color=COLORS[sector],
               label=sector.replace(" axis", ""), width=0.65)
        bottom += values
    ax.set(ylim=(0, 1), ylabel="Proportion of transitions")
    ax.set_xticks(range(len(SOURCES)), [source.capitalize() for source in SOURCES])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, title=None,
              loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, path)


def plot_empirical_overall_vectors(data, path):
    vectors = data.loc[data.record_type.eq("transition")]
    means = {source: overall_transition(vectors, source) for source in SOURCES}
    limit = 1.3 * max(abs(value).max() for value in means.values())
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(6, 3),
                             sharex=True, sharey=True, layout="constrained")
    for ax, source in zip(axes, SOURCES):
        value = means[source]
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)
        ax.annotate("", xy=value, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=3, mutation_scale=16))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
               title=source.capitalize(), xlabel="Delta NO", ylabel="Delta O")
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def _trace_axis(ax, summary, source, sector, condition, uncertainty, limits, fontsize):
    frame = summary.loc[summary.sector.eq(sector) & summary.source.eq(source) &
                        summary.condition_key.eq(condition)]
    ax.axvspan(0, 1, color="0.92")
    for response, color in (("NO", "black"), ("O", "red")):
        line = frame.loc[frame.response_type.eq(response)].sort_values("time_seconds")
        ax.plot(line.time_seconds, line.response, color=color, lw=1.2)
        ax.fill_between(line.time_seconds, line.response - line[uncertainty],
                        line.response + line[uncertainty], color=color, alpha=0.18, linewidth=0)
    ax.set(xlim=(-1, 3), ylim=limits[sector])
    ax.set_axis_off()
    return frame


def _plot_proportion_bar(ax, data, source, fontsize):
    transitions = data.loc[data.record_type.eq("transition")]
    counts = transitions.loc[transitions.source.eq(source)].groupby("sector").observation_id.nunique()
    total = counts.sum()
    bottom = 0.0
    for sector in reversed(DATA_SECTORS):
        proportion = counts.get(sector, 0) / total
        ax.bar(0, proportion, bottom=bottom, color=COLORS[sector], width=0.72)
        ax.text(0.43, bottom + proportion / 2, f"{sector.replace(' axis', '')}\n{proportion:.1%}",
                color=COLORS[sector], ha="left", va="center", fontsize=fontsize * 0.7)
        bottom += proportion
    ax.set(xlim=(-0.55, 1.35), ylim=(0, 1), ylabel="Proportion of transitions",
           title=source.capitalize())
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(labelsize=fontsize * 0.7)
    ax.yaxis.label.set_size(fontsize)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def panel_b_proportions(data, source, width=2, height=3, fontsize=12):
    """Export the labeled transition-proportion bar used beside one Panel B."""
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    _plot_proportion_bar(ax, data, source, fontsize)
    return fig, ax


def panel_b(data, model, source, width=8, height=2.8, fontsize=12,
            uncertainty="sd", show_proportions=True):
    """One familiar or novel B panel: proportion bar and four trace subpanels."""
    plotted_data = data.loc[~data.sector.eq("-O axis")]
    data_summary = trace_summary(data, plotted_data)
    model_summary = trace_summary(model, weights=empirical_sector_weights(data))
    core = model_summary.loc[model_summary.condition_key.isin(MODEL_VERSIONS["core"])]
    data_limits, model_limits = trace_limits(data_summary), trace_limits(core)
    columns = ((data_summary, "naive", "Data\nNaive", data_limits),
               (data_summary, "expert", "Data\nExpert", data_limits),
               (model_summary, "naive", "Model\nNaive", model_limits),
               (model_summary, "expert", "Model\nExpert", model_limits))
    if show_proportions:
        fig = plt.figure(figsize=(width, height), layout="constrained")
        grid = fig.add_gridspec(1, 5, width_ratios=(0.9, 1, 1, 1, 1))
        proportion_ax = fig.add_subplot(grid[0, 0])
        _plot_proportion_bar(proportion_ax, data, source, fontsize)
        trace_axes = [fig.add_subplot(grid[0, i]) for i in range(1, 5)]
    else:
        fig, trace_axes = plt.subplots(1, 4, figsize=(width, height), layout="constrained")
    for ax, (summary, condition, title, limits) in zip(trace_axes, columns):
        _trace_axis(ax, summary, source, "Overall", condition, uncertainty, limits, fontsize)
        ax.set_title(title, fontsize=fontsize * 1.5)
    trace_axes[0].text(-0.05, 0.5, source.capitalize(), transform=trace_axes[0].transAxes,
                       rotation=90, ha="right", va="center", fontsize=fontsize)
    trace_axes[-1].legend(handles=[plt.Line2D([], [], color="black", label="NO"),
                                   plt.Line2D([], [], color="red", label="O")],
                          frameon=False, fontsize=fontsize * 0.8, ncol=2, loc="upper right")
    _add_trace_scale(trace_axes[0], data_limits["Overall"], fontsize#, uncertainty
                     )
    return fig, trace_axes


def _add_trace_scale(ax, limits, fontsize, uncertainty="sd"):
    lower, upper = limits
    x, y = -0.8, lower + 0.02 * (upper - lower)
    ax.plot([x, x + 1], [y, y], color="black", lw=1.2, clip_on=False)
    ax.plot([x, x], [y, y + 1], color="black", lw=1.2, clip_on=False)
    ax.text(x + 0.5, y, "1 s", ha="center", va="top", fontsize=fontsize * 0.65)
    label = "1 S.D." if uncertainty == "sd" else "1 S.E.M."
    ax.text(x + 0.06, y + 1, label, ha="left", va="bottom", fontsize=fontsize * 0.65)


def _trace_grid(data, model, width, height, fontsize, uncertainty, model_conditions, 
                model_only = False, data_only = False):
    plotted_data = data.loc[~data.sector.eq("-O axis")]
    data_summary = trace_summary(data, plotted_data)
    model_summary = trace_summary(model, weights=empirical_sector_weights(data))
    data_limits = trace_limits(data_summary)
    model_plot = model_summary.loc[model_summary.condition_key.isin(("naive", *model_conditions))]
    model_limits = trace_limits(model_plot)
    columns = ((data_summary, "naive", "Data\nNaive", data_limits),
               (data_summary, "expert", "Data\nExpert", data_limits),
               (model_summary, "naive", "Model\nNaive", model_limits),
               *((model_summary, condition, f"Model\n{LABELS[condition]}", model_limits)
                 for condition in model_conditions))
    if model_only:
        columns = columns[2:]

    elif data_only:
        columns = columns[:2]
    # one y-scale per sector across every column shown, so data and model traces are directly comparable
    shown = [c[3] for c in columns]
    shared = {sector: (min(l[sector][0] for l in shown), max(l[sector][1] for l in shown)) for sector in TRACE_ROWS}
    columns = tuple((summary, condition, title, shared) for summary, condition, title, _ in columns)
    data_limits = shared

    fig, axes = plt.subplots(len(SECTORS) * len(SOURCES), len(columns),
                             figsize=(width, height), squeeze=False, layout="constrained")
    for row, (sector, source) in enumerate((sector, source) for sector in SECTORS for source in SOURCES):
        for col, (summary, condition, title, limits) in enumerate(columns):
            ax = axes[row, col]
            _trace_axis(ax, summary, source, sector, condition, uncertainty, limits, fontsize)
            if row == 0:
                ax.set_title(title, fontsize=fontsize * 1.5)
        axes[row, 0].text(-0.05, 0.5, f"{SECTOR_SHORT[sector]}\n{source.capitalize()}",
                          transform=axes[row, 0].transAxes, ha="right", va="center",
                          fontsize=fontsize, color=COLORS[sector])
    # axes[0, -1].legend(handles=[plt.Line2D([], [], color="black", label="NO"),
    #                             plt.Line2D([], [], color="red", label="O")],
    #                    frameon=False, fontsize=fontsize * 0.8, ncol=2, loc="upper right")
    _add_trace_scale(axes[0, 0], data_limits[SECTORS[0]], fontsize, #uncertainty
                     )
    return fig, axes


def panel_c(data, model, width=8, height=6.5, fontsize=10, uncertainty="sd"):
    """Panel C: empirical and core model traces, grouped by data sector."""
    return _trace_grid(data, model, width, height, fontsize, uncertainty, ("expert",))


def panel_c_appendix(data, model, width=12, height=6.5, fontsize=10, uncertainty="sd",
                     model_only = False, data_only = False):
    """Appendix C: Panel C plus model FB, PV, and combined silencing columns."""
    return _trace_grid(data, model, width, height, fontsize, uncertainty,
                       ("expert", "expert_no_fb", "expert_no_lat", "expert_no_fb_no_lat"),
                       model_only, data_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "ground_truth.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "model.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--include-minus-o", action="store_true",
                        help="Include empirical -O cells in overall vectors and traces")
    parser.add_argument("--similarity-errorbars", "--cosine-errorbars", action="store_true",
                        help="Plot SD across neuron-wise values in all three similarity panels")
    parser.add_argument("--uncertainty", choices=("sd", "sem", "both"), default="both",
                        help="Trace bands and vector covariance ellipses (default: both)")
    parser.add_argument("--overall-only", action="store_true",
                        help="Export only the empirical Overall traces and transition vectors")
    args = parser.parse_args()
    use_myriad_pro()
    data, model = pd.read_csv(args.data), pd.read_csv(args.model)
    plotted_data = data if args.include_minus_o else data.loc[~data.sector.eq("-O axis")]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.overall_only:
        summary = trace_summary(data, plotted_data)
        limits = trace_limits(summary)
        for source in SOURCES:
            for uncertainty in ("sd", "sem") if args.uncertainty == "both" else (args.uncertainty,):
                suffix = "" if uncertainty == "sd" else "_sem"
                plot_traces(summary, source, args.output / f"{source}_ground_truth_overall_traces{suffix}",
                            uncertainty, conditions=MODEL_VERSIONS["core"], limits=limits, rows=("Overall",))
        plot_empirical_overall_vectors(plotted_data, args.output / "ground_truth_overall_transition_vectors")
        return
    plot_transition_proportions(data, args.output / "transition_proportions")
    weights = empirical_sector_weights(data)
    data_summary = trace_summary(data, plotted_data)
    model_summary = trace_summary(model, weights=weights)
    core_conditions = MODEL_VERSIONS["core"]
    data_limits = trace_limits(data_summary)
    model_limits = {version: trace_limits(model_summary.loc[
        model_summary.condition_key.isin(conditions)]) for version, conditions in MODEL_VERSIONS.items()}
    for uncertainty in ("sd", "sem") if args.uncertainty == "both" else (args.uncertainty,):
        suffix = "" if uncertainty == "sd" else "_sem"
        for source in SOURCES:
            output = args.output / source
            traces_output = args.output / "traces" / source
            traces_output.mkdir(parents=True, exist_ok=True)
            vectors_output = output / "transition_vectors"
            vectors_output.mkdir(parents=True, exist_ok=True)
            plot_traces(data_summary, source, traces_output / f"ground_truth_traces{suffix}",
                        uncertainty, conditions=core_conditions, limits=data_limits)
            if uncertainty == "sem":
                for sector in TRACE_ROWS:
                    name = sector.lower().replace(" axis", "").replace("+", "plus_").replace("-", "minus_")
                    plot_traces(data_summary, source, traces_output / f"ground_truth_traces_sem_{name}",
                                uncertainty, conditions=core_conditions, limits=data_limits, rows=(sector,))
            for version, conditions in MODEL_VERSIONS.items():
                plot_traces(model_summary, source, traces_output / f"model_traces_{version}{suffix}",
                            uncertainty, conditions=conditions, limits=model_limits[version])
            if uncertainty == "sem":
                for sector in TRACE_ROWS:
                    name = sector.lower().replace(" axis", "").replace("+", "plus_").replace("-", "minus_")
                    plot_traces(model_summary, source, traces_output / f"model_traces_core_sem_{name}",
                                uncertainty, conditions=core_conditions, limits=model_limits["core"], rows=(sector,))
            for condition in tuple(LABELS)[1:]:
                prefix = "" if condition == "expert" else f"_{condition}"
                plot_vectors(plotted_data, model, source,
                             vectors_output / f"{prefix}_transition_vectors_{uncertainty}",
                             uncertainty=uncertainty, condition=condition)
    for source in SOURCES:
        similarity_output = args.output / "similarity"
        similarity_output.mkdir(parents=True, exist_ok=True)
        plot_similarities(plotted_data, model, source, similarity_output / f"{source}_similarity",
                          args.similarity_errorbars)


if __name__ == "__main__":
    main()
