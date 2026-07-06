from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


MATLAB_COLORS = (
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
    "#7E7C7C",
)

ROTATED_SECTOR_ORDER = ("+NO axis", "+O axis", "-NO axis", "-O axis", "small ∆")
ROTATED_SECTOR_PALETTE = {
    "+NO axis": MATLAB_COLORS[0],
    "+O axis": MATLAB_COLORS[1],
    "-NO axis": MATLAB_COLORS[2],
    "-O axis": MATLAB_COLORS[3],
    "small ∆": MATLAB_COLORS[-1]
}

ROTATED_SECTOR_PALETTE = {
    "+NO axis": 'blue',
    "+O axis": 'red',
    "-NO axis": 'darkorange',
    "-O axis": 'green',
    "small ∆": 'gray'
}


DEFAULT_PLOT_STYLE = {
    "pre_point_alpha": 0.6,
    "target_point_alpha": 0.22,
    "shift_point_alpha": 0.22,
    "sector_point_alpha": 0.24,
    "point_size": 28,
    "pre_vector_alpha": 0.42,
    "target_vector_alpha": 0.62,
    "alpha_min": 0.6,
    "alpha_max": 1.0,
    "individual_vector_width": 0.0046,
    "mean_arrow_width": 2.9,
    "mean_arrow_mutation_scale": 16.0,
}

# small epsilon to avoid log(0)
LOG_NORM_EPS = 1e-6


def resolve_plot_style(style: dict | None = None) -> dict:
    merged = DEFAULT_PLOT_STYLE.copy()
    if style is not None:
        merged.update(style)
    return merged


def build_transition_cmap(size: int = 256) -> np.ndarray:
    reds = np.linspace(0.0, 1.0, size)
    blues = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(reds, blues)

    cmap = np.zeros((size, size, 3))
    cmap[:, :, 0] = xx
    cmap[:, :, 2] = yy
    return cmap


def _rgb_from_values(
    no_values: pd.Series | np.ndarray,
    o_values: pd.Series | np.ndarray,
    *,
    no_min: float,
    no_max: float,
    o_min: float,
    o_max: float,
    cmap: np.ndarray,
) -> np.ndarray:
    no_values = np.asarray(no_values, dtype=float)
    o_values = np.asarray(o_values, dtype=float)

    no_span = no_max - no_min if no_max > no_min else 1.0
    o_span = o_max - o_min if o_max > o_min else 1.0

    no_idx = np.clip(np.rint((no_values - no_min) / no_span * 255).astype(int), 0, 255)
    o_idx = np.clip(np.rint((o_values - o_min) / o_span * 255).astype(int), 0, 255)
    return cmap[o_idx, no_idx]


def add_direction_columns(
    frame: pd.DataFrame,
    *,
    dx_col: str = "dNO",
    dy_col: str = "dO",
) -> pd.DataFrame:
    frame = frame.copy()
    frame["Angle"] = np.arctan2(frame[dy_col], frame[dx_col])
    frame["CosAngle"] = np.cos(frame["Angle"])
    frame["SinAngle"] = np.sin(frame["Angle"])
    return frame


def load_transition_table(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    stage_order = raw["stage"].drop_duplicates().tolist()

    wide = (
        raw.pivot(
            index=[
                "transition",
                "image_group",
                "image_idx_original",
                "image_idx_within_group",
                "neuron_idx",
                "stage",
            ],
            columns="image_type",
            values="response",
        )
        .reset_index()
        .rename(columns={"Full": "NO", "Occl": "O"})
    )

    wide["stage"] = pd.Categorical(wide["stage"], categories=stage_order, ordered=True)
    wide = wide.sort_values(
        ["transition", "image_group", "image_idx_original", "neuron_idx", "stage"]
    ).reset_index(drop=True)
    return wide


def _resolve_target_stage(
    frame: pd.DataFrame,
    *,
    pre_stage: str = "Pre",
    target_stage: str | None = None,
) -> str:
    if target_stage is not None:
        return target_stage

    candidates = [stage for stage in frame["stage"].astype(str).drop_duplicates().tolist() if stage != pre_stage]
    if not candidates:
        raise ValueError(f"Could not infer a target stage after {pre_stage}.")
    return candidates[0]


def assign_rotated_sectors(frame: pd.DataFrame, threshold:float = 0.0) -> pd.DataFrame:
    frame = frame.copy()
    angle = frame["Angle"]
    norm = frame['dNorm']

    sector = np.full(len(frame), "+NO axis", dtype=object)
    sector[(angle >= np.pi / 4.0) & (angle < 3.0 * np.pi / 4.0)] = "+O axis"
    sector[(angle >= 3.0 * np.pi / 4.0) | (angle < -3.0 * np.pi / 4.0)] = "-NO axis"
    sector[(angle >= -3.0 * np.pi / 4.0) & (angle < -np.pi / 4.0)] = "-O axis"
    sector[norm <= threshold] = "small ∆"


    frame["RotatedSector"] = pd.Categorical(
        sector,
        categories=ROTATED_SECTOR_ORDER,
        ordered=True,
    )
    return frame


def build_mean_summary(
    transition_table: pd.DataFrame,
    *,
    image_group: str | None = None,
    pre_stage: str = "Pre",
    target_stage: str | None = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    frame = transition_table.copy()
    if image_group is not None:
        frame = frame.loc[frame["image_group"] == image_group].copy()

    if frame.empty:
        raise ValueError("No rows left after filtering transition data.")

    target_stage = _resolve_target_stage(frame, pre_stage=pre_stage, target_stage=target_stage)

    stage_means = (
        frame.groupby(["neuron_idx", "stage"], as_index=False)[["NO", "O"]]
        .mean()
    )

    summary = stage_means.pivot(index="neuron_idx", columns="stage", values=["NO", "O"])
    summary.columns = [f"{metric}_{stage}" for metric, stage in summary.columns]
    summary = summary.reset_index()
    summary = summary.dropna(
        subset=[f"NO_{pre_stage}", f"O_{pre_stage}", f"NO_{target_stage}", f"O_{target_stage}"]
    )

    summary = summary.rename(
        columns={
            f"NO_{pre_stage}": "NO_Pre",
            f"O_{pre_stage}": "O_Pre",
            f"NO_{target_stage}": "NO_Target",
            f"O_{target_stage}": "O_Target",
        }
    )
    summary["dNO"] = summary["NO_Target"] - summary["NO_Pre"]
    summary["dO"] = summary["O_Target"] - summary["O_Pre"]
    # add displacement norm and its log for downstream analysis/plotting
    summary["dNorm"] = np.hypot(summary["dNO"].to_numpy(dtype=float), summary["dO"].to_numpy(dtype=float))
    summary["log_dNorm"] = np.log(summary["dNorm"] + LOG_NORM_EPS)
    summary = add_direction_columns(summary)
    summary = assign_rotated_sectors(summary, threshold=threshold)

    summary.attrs["pre_stage"] = pre_stage
    summary.attrs["target_stage"] = target_stage
    summary.attrs["image_group"] = image_group or frame["image_group"].mode().iloc[0]
    summary.attrs["transition"] = frame["transition"].mode().iloc[0]
    return summary


def attach_pre_colors(summary_df: pd.DataFrame, *, cmap: np.ndarray | None = None) -> pd.DataFrame:
    cmap = build_transition_cmap() if cmap is None else cmap
    summary_df = summary_df.copy()
    summary_df["PreColor"] = list(
        _rgb_from_values(
            summary_df["NO_Pre"],
            summary_df["O_Pre"],
            no_min=summary_df["NO_Pre"].min(),
            no_max=summary_df["NO_Pre"].max(),
            o_min=summary_df["O_Pre"].min(),
            o_max=summary_df["O_Pre"].max(),
            cmap=cmap,
        )
    )
    return summary_df


def compute_response_limits(*frames: pd.DataFrame, pad: float = 0.4) -> list[float]:
    values: list[np.ndarray] = []
    for frame in frames:
        values.extend(
            [
                frame["NO_Pre"].to_numpy(dtype=float),
                frame["O_Pre"].to_numpy(dtype=float),
                frame["NO_Target"].to_numpy(dtype=float),
                frame["O_Target"].to_numpy(dtype=float),
            ]
        )

    all_values = np.concatenate(values)
    return [float(np.nanmin(all_values) - pad), float(np.nanmax(all_values) + pad)]


def compute_shift_limits(
    *frames: pd.DataFrame,
    pad_ratio: float = 0.12,
    fallback_extent: float = 0.5,
) -> list[float]:
    values: list[np.ndarray] = []
    for frame in frames:
        values.extend([frame["dNO"].to_numpy(dtype=float), frame["dO"].to_numpy(dtype=float)])

    extent = np.nanmax(np.abs(np.concatenate(values)))
    if not np.isfinite(extent) or extent == 0:
        extent = fallback_extent
    else:
        extent *= 1.0 + pad_ratio
    return [-float(extent), float(extent)]


def sector_labels_with_counts(summary_df: pd.DataFrame) -> dict[str, str]:
    total = max(len(summary_df), 1)
    counts = (
        summary_df["RotatedSector"]
        .value_counts(sort=False)
        .reindex(ROTATED_SECTOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    return {
        sector: f"{sector} (n={count}, {100.0 * count / total:.1f}%)"
        for sector, count in counts.items()
    }


def sector_fraction_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    total = max(len(summary_df), 1)
    counts = (
        summary_df["RotatedSector"]
        .value_counts(sort=False)
        .reindex(ROTATED_SECTOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    means = summary_df.groupby("RotatedSector", observed=True)["dNorm"].mean().reindex(ROTATED_SECTOR_ORDER)
    medians = summary_df.groupby("RotatedSector", observed=True)["dNorm"].median().reindex(ROTATED_SECTOR_ORDER)
    
    fractions = counts / total
    table = pd.DataFrame(
        {
            "RotatedSector": list(ROTATED_SECTOR_ORDER),
            "Count": counts.to_numpy(),
            "Fraction": fractions.to_numpy(dtype=float),
            "Mean_dNorm": means.to_numpy(dtype=float),
            "Median_dNorm": medians.to_numpy(dtype=float),
        }
    )
    table["ExpectedFraction"] = 0.25
    table["DeltaFromExpected"] = table["Fraction"] - table["ExpectedFraction"]
    table["AboveExpected"] = table["DeltaFromExpected"] > 0.0
    return table


def sector_mean_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    sector_means = (
        summary_df.groupby("RotatedSector", observed=True, as_index=False)
        .agg(
            NO_Pre=("NO_Pre", "mean"),
            O_Pre=("O_Pre", "mean"),
            NO_Target=("NO_Target", "mean"),
            O_Target=("O_Target", "mean"),
            dNO=("dNO", "mean"),
            dO=("dO", "mean"),
            Count=("neuron_idx", "size"),
        )
    )
    sector_means["MeanAngle"] = np.arctan2(sector_means["dO"], sector_means["dNO"])
    return sector_means


def _sector_percentages(summary_df: pd.DataFrame) -> dict[str, float]:
    fractions = sector_fraction_table(summary_df).set_index("RotatedSector")["Fraction"]
    return {
        sector: float(fractions.get(sector, 0.0)) * 100.0
        for sector in ROTATED_SECTOR_ORDER
    }


def _sector_plot_order(*, small_delta_first: bool = False) -> tuple[str, ...]:
    if not small_delta_first:
        return ROTATED_SECTOR_ORDER
    return ("small ∆",) + tuple(sector for sector in ROTATED_SECTOR_ORDER if sector != "small ∆")


def _sector_scatter_zorder(sector: str) -> int:
    return 1 if sector == "small ∆" else 2


def _legend_label_anchor(
    vector_xy: tuple[float, float],
    *,
    gap: float,
) -> tuple[tuple[float, float], str, str]:
    vector = np.asarray(vector_xy, dtype=float)
    norm = float(np.hypot(vector[0], vector[1]))
    unit = vector / norm if norm > 0 else np.array([0.0, 0.0])
    label_xy = vector + unit * gap

    if abs(unit[0]) >= abs(unit[1]):
        ha = "left" if unit[0] >= 0 else "right"
        va = "center"
    else:
        ha = "center"
        va = "bottom" if unit[1] >= 0 else "top"

    return (float(label_xy[0]), float(label_xy[1])), ha, va


def _format_legend_percent(value: float) -> str:
    return f"{value:.1f}"


def plot_rotated_sector_unit_legend(
    summary_df: pd.DataFrame,
    *,
    title: str | None = None,
    center_radius: float = 0.3,
    vector_length: float = 0.46,
    label_color: str = "white",
    label_fontsize: float = 20.0,
    center_label_fontsize: float | None = None,
    label_gap: float = 0.0,
    label_bbox: dict | None = None,
) -> plt.Figure:
    """Draw a standalone transition-sector legend with sector percentages."""
    percentages = _sector_percentages(summary_df)
    center_label_fontsize = label_fontsize if center_label_fontsize is None else center_label_fontsize

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")

    sector_specs = {
        "+NO axis": (-45.0, 45.0, (vector_length, 0.0)),
        "+O axis": (45.0, 135.0, (0.0, vector_length)),
        "-NO axis": (135.0, 225.0, (-vector_length, 0.0)),
        "-O axis": (225.0, 315.0, (0.0, -vector_length)),
    }

    for sector, (theta1, theta2, vector_xy) in sector_specs.items():
        color = ROTATED_SECTOR_PALETTE[sector]
        text_xy, ha, va = _legend_label_anchor(
            vector_xy,
            gap=label_gap,
        )
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                1.0,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="none",
                linewidth=1.0,
                alpha=0.6,
            )
        )
        _draw_arrow(
            ax,
            (0.0, 0.0),
            vector_xy,
            color=color,
            linewidth=3.0,
            mutation_scale=16.0,
            zorder=4,
            alpha=1.0,
        )
        ax.text(
            text_xy[0],
            text_xy[1],
            _format_legend_percent(percentages[sector]),
            ha=ha,
            va=va,
            fontsize=label_fontsize,
            fontweight="bold",
            color=label_color,
            bbox=label_bbox,
            zorder=7,
        )

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            center_radius,
            facecolor=ROTATED_SECTOR_PALETTE["small ∆"],
            edgecolor="none",
            linewidth=1.0,
            alpha=1.0,
            zorder=5,
        )
    )
    ax.text(
        0.0,
        0.0,
        _format_legend_percent(percentages["small ∆"]),
        ha="center",
        va="center",
        fontsize=center_label_fontsize,
        fontweight="bold",
        color=label_color,
        bbox=label_bbox,
        zorder=6,
    )
    # ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="none", edgecolor="0.15", linewidth=1.5, zorder=6))

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout(pad=0.3)
    return fig


def save_rotated_sector_unit_legend(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 300,
    label_color: str = "white",
    label_fontsize: float = 20.0,
    center_label_fontsize: float | None = None,
    label_gap: float = 0.0,
    label_bbox: dict | None = None,
    formats: tuple[str, ...] | None = None,
) -> list[Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_rotated_sector_unit_legend(
        summary_df,
        title=title,
        label_color=label_color,
        label_fontsize=label_fontsize,
        center_label_fontsize=center_label_fontsize,
        label_gap=label_gap,
        label_bbox=label_bbox,
    )
    if formats is None:
        formats = (output_path.suffix.lstrip(".") or "png",)
    saved_paths = []
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def _stack_pre_colors(summary_df: pd.DataFrame) -> np.ndarray:
    return np.vstack(summary_df["PreColor"].to_numpy())


def _stack_pre_colors_with_alpha(summary_df: pd.DataFrame, alphas: np.ndarray | None = None) -> np.ndarray:
    """Return Nx3 (RGB) or Nx4 (RGBA) color array from `PreColor` and optional per-point alphas."""
    rgb = np.vstack(summary_df["PreColor"].to_numpy())
    if alphas is None:
        return rgb
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.shape[0] != rgb.shape[0]:
        raise ValueError("alphas must be a 1-D array with same length as summary_df")
    rgba = np.concatenate([rgb, alphas.reshape(-1, 1)], axis=1)
    return rgba


def _map_norms_to_alphas(norms: np.ndarray, *, min_alpha: float = 0.3, max_alpha: float = 1.0) -> np.ndarray:
    norms = np.asarray(norms, dtype=float)
    if norms.size == 0:
        return np.array([], dtype=float)
    mn = float(np.nanmin(norms))
    mx = float(np.nanmax(norms))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.full_like(norms, max_alpha, dtype=float)
    scaled = (norms - mn) / (mx - mn)
    return min_alpha + scaled * (max_alpha - min_alpha)


def _darken_color(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _sector_percentage_alphas(
    summary_df: pd.DataFrame,
    *,
    min_alpha: float = 0.5,
    max_alpha: float = 1.0,
) -> dict[str, float]:
    sectors = [sector for sector in ROTATED_SECTOR_ORDER if sector != "small ∆"]
    fractions = (
        sector_fraction_table(summary_df)
        .set_index("RotatedSector")
        .loc[sectors, "Fraction"]
        .astype(float)
    )
    mn = float(fractions.min())
    mx = float(fractions.max())
    if mx <= mn:
        return {sector: max_alpha for sector in sectors}

    scaled = (fractions - mn) / (mx - mn)
    alphas = min_alpha + scaled * (max_alpha - min_alpha)
    return {sector: float(alphas.loc[sector]) for sector in sectors}


def _draw_individual_vectors(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    alphas: np.ndarray | None,
    width: float,
) -> None:
    colors = _stack_pre_colors_with_alpha(summary_df, alphas=alphas)
    ax.quiver(
        summary_df["NO_Pre"],
        summary_df["O_Pre"],
        summary_df["dNO"],
        summary_df["dO"],
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=width,
        zorder=2,
    )


def _draw_arrow(
    ax: plt.Axes,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    mutation_scale: float,
    zorder: int = 4,
    alpha: float = 0.98,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _draw_diagonal(ax: plt.Axes, lims: list[float]) -> None:
    ax.plot(lims, lims, "--", color="0.75", linewidth=1.0, zorder=0)


def _draw_origin_guides(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="0.82", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=1.0, zorder=0)


def _draw_rotated_guides(ax: plt.Axes, lims: list[float]) -> None:
    extent = float(max(abs(lims[0]), abs(lims[1])))
    guide = np.array([-extent, extent], dtype=float)
    ax.plot(guide, guide, "--", color="0.82", linewidth=1.0, zorder=0)
    ax.plot(guide, -guide, "--", color="0.82", linewidth=1.0, zorder=0)


def _legend_handles(labels: dict[str, str], *, linewidth: float) -> list[Line2D]:
    handles = []
    for sector in ROTATED_SECTOR_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ROTATED_SECTOR_PALETTE[sector],
                linewidth=linewidth,
                marker="o",
                markersize=7,
                label=labels[sector],
            )
        )
    return handles


def export_figure_panels(
    fig: plt.Figure,
    output_dir: str | Path,
    basename: str,
    *,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
    pad_inches: float = 0.04,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved_paths: list[Path] = []
    for idx, ax in enumerate(fig.axes, start=1):
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            bbox = ax.get_window_extent(renderer)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox = Bbox.from_extents(
            bbox.x0 - pad_inches,
            bbox.y0 - pad_inches,
            bbox.x1 + pad_inches,
            bbox.y1 + pad_inches,
        )
        for fmt in formats:
            path = output_dir / f"{basename}_panel_{idx:02d}.{fmt}"
            fig.savefig(path, bbox_inches=bbox, dpi=dpi)
            saved_paths.append(path)
    return saved_paths


def plot_mean_transition_summary(
    summary_df: pd.DataFrame,
    *,
    title: str,
    start_label: str | None = None,
    end_label: str | None = None,
    response_lims: list[float] | None = None,
    shift_lims: list[float] | None = None,
    style: dict | None = None,
) -> plt.Figure:
    summary_df = attach_pre_colors(summary_df)
    style = resolve_plot_style(style)
    response_lims = compute_response_limits(summary_df) if response_lims is None else response_lims
    shift_lims = compute_shift_limits(summary_df) if shift_lims is None else shift_lims

    sector_means = sector_mean_table(summary_df)
    sector_labels = sector_labels_with_counts(summary_df)
    sector_arrow_alphas = _sector_percentage_alphas(summary_df)

    # use pre-computed norm/log from summary if available (preferred)
    if "dNorm" in summary_df.columns and "log_dNorm" in summary_df.columns:
        norms = summary_df["dNorm"].to_numpy(dtype=float)
        log_norms = summary_df["log_dNorm"].to_numpy(dtype=float)
    else:
        # fallback: compute locally
        norms = np.hypot(summary_df["dNO"].to_numpy(dtype=float), summary_df["dO"].to_numpy(dtype=float))
        log_norms = np.log(norms + LOG_NORM_EPS)
    # Use the same opacity range for scatter points across all rows.
    alphas_row01 = _map_norms_to_alphas(log_norms, min_alpha=style["alpha_min"], max_alpha=style["alpha_max"])
    alphas_row2 = _map_norms_to_alphas(log_norms, min_alpha=style["alpha_min"], max_alpha=style["alpha_max"])

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.8), sharex=False, sharey=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 1]
    colors_rgba = _stack_pre_colors_with_alpha(summary_df, alphas=alphas_row01)
    ax.scatter(
        summary_df["NO_Pre"],
        summary_df["O_Pre"],
        s=style["point_size"],
        c=colors_rgba,
        edgecolors="none",
        zorder=1,
    )
    _draw_individual_vectors(
        ax,
        summary_df,
        alphas=np.full(len(summary_df), 0.5),
        width=style["individual_vector_width"],
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{start_label} by pre color")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[0, 2]
    ax.scatter(
        summary_df["NO_Target"],
        summary_df["O_Target"],
        s=style["point_size"],
        c=colors_rgba,
        edgecolors="none",
        zorder=1,
    )
    _draw_individual_vectors(
        ax,
        summary_df,
        alphas=np.full(len(summary_df), 0.5),
        width=style["individual_vector_width"],
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} by pre color")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[0, 0]
    ax.scatter(
        summary_df["dNO"],
        summary_df["dO"],
        s=style["point_size"],
        c=colors_rgba,
        edgecolors="none",
        zorder=1,
    )
    _draw_origin_guides(ax)
    _draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} - {start_label} by pre color")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")

    ax = axes[1, 1]
    for sector in _sector_plot_order(small_delta_first=True):
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        # sector-colored points with per-point alpha inherited from global mapping
        pos_idx = np.flatnonzero(summary_df["RotatedSector"].to_numpy() == sector)
        sector_alphas = alphas_row01[pos_idx]
        rgb = np.array(mcolors.to_rgb(color)).reshape(1, 3)
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, sector_alphas.reshape(-1, 1)], axis=1)
        ax.scatter(
            sector_rows["NO_Pre"],
            sector_rows["O_Pre"],
            s=style["point_size"],
            c=rgba,
            edgecolors="none",
            zorder=_sector_scatter_zorder(sector),
        )
        # mean arrow removed from this subplot to reduce visual clutter
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{start_label} by rotated sector")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[1, 2]
    for sector in _sector_plot_order(small_delta_first=True):
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        pos_idx = np.flatnonzero(summary_df["RotatedSector"].to_numpy() == sector)
        sector_alphas = alphas_row01[pos_idx]
        rgb = np.array(mcolors.to_rgb(color)).reshape(1, 3)
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, sector_alphas.reshape(-1, 1)], axis=1)
        ax.scatter(
            sector_rows["NO_Target"],
            sector_rows["O_Target"],
            s=style["point_size"],
            c=rgba,
            edgecolors="none",
            zorder=_sector_scatter_zorder(sector),
        )
        # mean arrow removed from this subplot to reduce visual clutter
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} by rotated sector")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[1, 0]
    for sector in _sector_plot_order(small_delta_first=True):
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        color = ROTATED_SECTOR_PALETTE[sector]
        pos_idx = np.flatnonzero(summary_df["RotatedSector"].to_numpy() == sector)
        sector_alphas = alphas_row01[pos_idx]
        rgb = np.array(mcolors.to_rgb(color)).reshape(1, 3)
        rgba = np.repeat(rgb, len(sector_rows), axis=0)
        rgba = np.concatenate([rgba, sector_alphas.reshape(-1, 1)], axis=1)
        ax.scatter(
            sector_rows["dNO"],
            sector_rows["dO"],
            s=style["point_size"],
            c=rgba,
            edgecolors="none",
            zorder=_sector_scatter_zorder(sector),
        )
        if sector == "small ∆":
            continue
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector].iloc[0]
        _draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=_darken_color(ROTATED_SECTOR_PALETTE[sector]),
            linewidth=max(3.0, style["mean_arrow_width"] * 0.9),
            mutation_scale=style["mean_arrow_mutation_scale"],
            alpha=sector_arrow_alphas[sector],
            zorder=4,
        )
    _draw_origin_guides(ax)
    _draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} - {start_label} by rotated sector")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")
    ax.legend(
        handles=_legend_handles(sector_labels, linewidth=style["mean_arrow_width"]),
        frameon=False,
        loc="best",
    )

    # --- Third row: color + alpha mapped to displacement norm (coolwarm colormap) ---
    ax = axes[2, 1]
    # normalized scalars for colormap using log norms
    mn = float(np.nanmin(log_norms)) if log_norms.size else 0.0
    mx = float(np.nanmax(log_norms)) if log_norms.size else 0.0
    if mx <= mn:
        norm_scaled = np.zeros_like(log_norms, dtype=float)
    else:
        norm_scaled = (log_norms - mn) / (mx - mn)
    cmap = plt.cm.coolwarm
    colors_by_norm = cmap(norm_scaled)
    # replace alpha channel with mapped alphas for third row
    colors_by_norm[:, 3] = alphas_row2

    ax.scatter(
        summary_df["NO_Pre"],
        summary_df["O_Pre"],
        s=style["point_size"],
        c=colors_by_norm,
        edgecolors="none",
        zorder=1,
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{start_label} colored by |d| (coolwarm)")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[2, 2]
    ax.scatter(
        summary_df["NO_Target"],
        summary_df["O_Target"],
        s=style["point_size"],
        c=colors_by_norm,
        edgecolors="none",
        zorder=1,
    )
    _draw_diagonal(ax, response_lims)
    ax.set_xlim(response_lims)
    ax.set_ylim(response_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{end_label} colored by |d| (coolwarm)")
    ax.set_xlabel("NO")
    ax.set_ylabel("O")

    ax = axes[2, 0]
    for sector in _sector_plot_order(small_delta_first=True):
        sector_rows = summary_df.loc[summary_df["RotatedSector"] == sector]
        if sector_rows.empty:
            continue
        pos_idx = np.flatnonzero(summary_df["RotatedSector"].to_numpy() == sector)
        ax.scatter(
            sector_rows["dNO"],
            sector_rows["dO"],
            s=style["point_size"],
            c=colors_by_norm[pos_idx],
            edgecolors="none",
            zorder=_sector_scatter_zorder(sector),
        )
        if sector == "small ∆":
            continue
        mean_row = sector_means.loc[sector_means["RotatedSector"] == sector].iloc[0]
        _draw_arrow(
            ax,
            (0.0, 0.0),
            (float(mean_row["dNO"]), float(mean_row["dO"])),
            color=_darken_color(ROTATED_SECTOR_PALETTE[sector]),
            linewidth=max(3.2, style["mean_arrow_width"]),
            mutation_scale=style["mean_arrow_mutation_scale"],
            zorder=4,
            alpha=sector_arrow_alphas[sector],
        )
    _draw_origin_guides(ax)
    _draw_rotated_guides(ax, shift_lims)
    ax.set_xlim(shift_lims)
    ax.set_ylim(shift_lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("|d| colored by coolwarm, alpha scaled by |d|")
    ax.set_xlabel("dNO")
    ax.set_ylabel("dO")

    fig.text(
        0.5,
        0.03,
        "Rotated sectors are defined by dO = +/- dNO.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig
