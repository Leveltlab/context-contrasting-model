import hashlib
import os
import re
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from pandas import DataFrame

PLOTSDIR = os.path.join(os.path.dirname(__file__), "plotsexperiment_s")
PLOT_ALL_PANELS_DIR = os.path.join(PLOTSDIR, "all_panels")
PLOT_PANEL_A_DIR = os.path.join(PLOTSDIR, "panel_A")
PLOT_TRANSITION_PANELS_DIR = os.path.join(PLOTSDIR, "transition_panels")

PLOT_CONDITION_LABELS = {
    "full": "Nonoccluded",
    "occlusion": "Occluded",
    "no_context": "FB silencing",
    "nolat": "PV silencing",
    "no_context_nolat": "FB & PV silencing",
}
PLOT_CONDITION_ORDER = ["full", "occlusion", "no_context", "nolat", "no_context_nolat"]
PLOT_COLORS = {
    "Nonoccluded": "black",
    "Occluded": "red",
    "FB silencing": "blue",
    "PV silencing": "green",
    "FB & PV silencing": "darkorange",
}
TRANSITION_LABELS: dict[str, str] = {}
TRACE_COLORS = {
    "full": "black",
    "occlusion": "red",
    "no_context": "blue",
    "nolat": "green",
    "no_context_nolat": "darkorange",
}
TRACE_LINESTYLES = {
    "full": "-",
    "occlusion": "-",
    "no_context": ":",
    "nolat": "--",
    "no_context_nolat": "--",
}
TRACE_LABELS = {
    "full": "Nonoccluded",
    "occlusion": "Occluded",
    "no_context": "FB silencing",
    "occlusion_no_context": "Occluded FB silencing",
    "nolat": "PV silencing",
    "occlusion_nolat": "Occluded PV silencing",
    "no_context_nolat": "FB & PV silencing",
    "occlusion_no_context_nolat": "Occluded FB & PV silencing",
}
TRANSITION_RESPONSE_COLUMN_SPECS = (
    {
        "key": "naive",
        "label": "Naive",
        "phase": "naive",
        "no_trace": "full",
        "o_trace": "occlusion",
    },
    {
        "key": "expert",
        "label": "Expert",
        "phase": "expert",
        "no_trace": "full",
        "o_trace": "occlusion",
    },
    {
        "key": "expert_no_fb",
        "label": TRACE_LABELS["no_context"],
        "phase": "expert",
        "no_trace": "no_context",
        "o_trace": "occlusion_no_context",
    },
    {
        "key": "expert_no_lat",
        "label": TRACE_LABELS["nolat"],
        "phase": "expert",
        "no_trace": "nolat",
        "o_trace": "occlusion_nolat",
    },
    {
        "key": "expert_no_fb_no_lat",
        "label": TRACE_LABELS["no_context_nolat"],
        "phase": "expert",
        "no_trace": "no_context_nolat",
        "o_trace": "occlusion_no_context_nolat",
    },
)
IMAGE_LABELS = {"familiar": "Familiar Image", "novel": "Novel Image"}
AXIS_LABEL_FONTSIZE = 32
AXIS_TICK_FONTSIZE = 32
TIME_STEPS_PER_SECOND = 100.0
PHASE_DISPLAY_LABELS = {
    "naive": "Naive",
    "expert": "Expert",
}
PHASE_ORDER = ["naive", "expert"]


def format_transition_label(name: str) -> str:
    return str(name).replace("_", " ")


def resolve_transition_labels(
    transition_names: list[str],
    transition_labels: dict[str, str] | None = None,
) -> dict[str, str]:
    labels = {name: format_transition_label(name) for name in transition_names}
    if transition_labels is not None:
        labels.update(transition_labels)
    return labels


def resolve_transition_order(
    long_dfs_by_transition: dict[str, DataFrame],
    transition_order: list[str] | None = None,
) -> list[str]:
    if transition_order is None:
        return list(long_dfs_by_transition)

    ordered = [transition for transition in transition_order if transition in long_dfs_by_transition]
    return ordered or list(long_dfs_by_transition)


def _safe_filename_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return stem or "config"


def build_config_output_name_map(config_names: list[str]) -> dict[str, str]:
    base_by_name = {name: _safe_filename_stem(name) for name in config_names}
    names_by_casefolded_base: dict[str, list[str]] = {}
    for name, base in base_by_name.items():
        names_by_casefolded_base.setdefault(base.casefold(), []).append(name)

    output_names: dict[str, str] = {}
    for casefolded_base, names in names_by_casefolded_base.items():
        if len(names) == 1:
            output_names[names[0]] = base_by_name[names[0]]
            continue

        for name in names:
            digest = hashlib.blake2s(str(name).encode("utf-8"), digest_size=3).hexdigest()
            output_names[name] = f"{base_by_name[name]}__{digest}"

    return output_names


def _resolve_phase_sequence(long_df: DataFrame) -> list[str]:
    if "experiment_phase" not in long_df.columns:
        return []
    present = long_df["experiment_phase"].dropna().astype(str).unique().tolist()
    return [phase for phase in PHASE_ORDER if phase in present]


def _resolve_plot_dirs(base_dir: str) -> dict[str, str]:
    if os.path.abspath(base_dir) == os.path.abspath(PLOTSDIR):
        plot_dirs = {
            "all_panels": PLOT_ALL_PANELS_DIR,
            "panel_a": PLOT_PANEL_A_DIR,
            "transition_panels": PLOT_TRANSITION_PANELS_DIR,
        }
    else:
        plot_dirs = {
            "all_panels": os.path.join(base_dir, "all_panels"),
            "panel_a": os.path.join(base_dir, "panel_A"),
            "transition_panels": os.path.join(base_dir, "transition_panels"),
        }

    for path in plot_dirs.values():
        os.makedirs(path, exist_ok=True)
    return plot_dirs


def _natural_sort_key(text: str) -> list[str | int]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def _is_familiar_condition(condition: str) -> bool:
    return str(condition).startswith("familiar")


def _is_novel_condition(condition: str) -> bool:
    return str(condition).startswith("novel")


def _condition_sort_key(condition: str) -> tuple[int, int, list[str | int]]:
    text = str(condition)
    if _is_familiar_condition(text):
        return (0, 0 if text == "familiar" else 1, _natural_sort_key(text))
    if _is_novel_condition(text):
        return (1, 0 if text == "novel" else 1, _natural_sort_key(text))
    return (2, 0, _natural_sort_key(text))


def _resolve_condition_sequence(
    conditions: list[str],
    preferred: list[str] | None = None,
) -> list[str]:
    present = [str(condition) for condition in conditions if pd.notna(condition)]
    if not present:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    for condition in preferred or []:
        text = str(condition)
        if text in present and text not in seen:
            ordered.append(text)
            seen.add(text)

    for condition in sorted(set(present), key=_condition_sort_key):
        if condition not in seen:
            ordered.append(condition)
            seen.add(condition)

    return ordered


def _display_condition_label(condition: str) -> str:
    text = str(condition)
    if text in IMAGE_LABELS:
        return IMAGE_LABELS[text]
    return text.replace("_", " ").title()


def _add_row_scale_bar(
    axes_row: np.ndarray,
    *,
    length: float = 1.0,
    color: str = "0.15",
    linewidth: float = 2.2,
) -> None:
    visible_axes = [ax for ax in axes_row if ax.get_visible()]
    if not visible_axes:
        return
    ax = visible_axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.06 * (x1 - x0)
    y_start = y0 + 0.18 * (y1 - y0)
    y_end = min(y_start + length, y1 - 0.08 * (y1 - y0))
    if y_end <= y_start:
        return
    ax.plot([x, x], [y_start, y_end], color=color, lw=linewidth, solid_capstyle="butt", zorder=20)


def _matrix_header_fontsizes(n_cols: int) -> tuple[int, int]:
    condition_size = int(np.clip(220.0 / max(n_cols, 1), 18, 32))
    group_size = int(np.clip(240.0 / max(n_cols, 1), 20, 32))
    return condition_size, group_size


def _transition_response_layout_conditions(
    *,
    image_mode: Literal["familiar", "novel"],
    available_conditions: list[str],
    selected_conditions: list[str],
) -> list[str]:
    if image_mode != "novel":
        return selected_conditions

    familiar_conditions = [
        condition
        for condition in _resolve_condition_sequence(available_conditions)
        if _is_familiar_condition(condition)
    ]
    return familiar_conditions if len(familiar_conditions) > len(selected_conditions) else selected_conditions


def _span_axis_across(ax: plt.Axes, span_axes: list[plt.Axes]) -> None:
    if len(span_axes) <= 1:
        return
    position = ax.get_position()
    span_left = min(span_ax.get_position().x0 for span_ax in span_axes)
    span_right = max(span_ax.get_position().x1 for span_ax in span_axes)
    ax.set_position([span_left, position.y0, span_right - span_left, position.height])


def _add_plot_condition_labels(df: DataFrame) -> DataFrame:
    styled = df.copy()
    if "image_type" in styled.columns:
        styled["plot_condition"] = styled["image_type"].map(PLOT_CONDITION_LABELS).fillna(styled["image_type"])
    return styled


def _plot_condition_order(image_types: list[str]) -> list[str]:
    return [PLOT_CONDITION_LABELS[k] for k in PLOT_CONDITION_ORDER if k in image_types]


def _resolve_xlim(xlim: tuple[float, float] | None) -> tuple[float, float]:
    if xlim is None:
        return (1000.0, 1350.0)
    start, end = xlim
    if start >= end:
        raise ValueError("xlim must be an increasing (start, end) tuple.")
    return float(start), float(end)


def _style_axis_fonts(ax, *, tick_labelsize: int = AXIS_TICK_FONTSIZE) -> None:
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=tick_labelsize)


def _to_np_2d(ts: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(ts, torch.Tensor):
        arr = ts.detach().cpu().numpy()
    else:
        arr = np.asarray(ts)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _condition_token_to_image_type(image_type: str, condition_token: str) -> tuple[str, str]:
    if image_type == "nocontext":
        return "no_context", condition_token
    if image_type == "nolat":
        return "nolat", condition_token
    if image_type == "nocontextnolat":
        return "no_context_nolat", condition_token
    suffix_to_type = {
        "_nocontextnolat": "no_context_nolat",
        "_nocontext": "no_context",
        "_nolat": "nolat",
    }
    for suffix, ablation_type in suffix_to_type.items():
        if condition_token.endswith(suffix):
            base_condition = condition_token.removesuffix(suffix)
            if image_type == "full":
                return ablation_type, base_condition
            if image_type == "occlusion":
                return f"occlusion_{ablation_type}", base_condition
            return f"{image_type}_{ablation_type}", base_condition
    if condition_token.endswith("_nocontext"):
        base_condition = condition_token.removesuffix("_nocontext")
        return "no_context", base_condition
    return image_type, condition_token


def _resolve_trace_types(
    image_types: list[str] | set[str],
    *,
    include_no_response_ablations: bool,
    base_trace_types: tuple[str, ...] = ("full", "occlusion"),
) -> tuple[str, ...]:
    available = {str(image_type) for image_type in image_types if pd.notna(image_type)}
    trace_types = [trace_type for trace_type in base_trace_types if trace_type in available]

    if include_no_response_ablations:
        for trace_type in PLOT_CONDITION_ORDER:
            if trace_type in base_trace_types:
                continue
            if trace_type in available:
                trace_types.append(trace_type)

    return tuple(trace_types)


def _indexed_palette(indices: list[int], palette_name: str) -> dict[int, tuple[float, float, float]]:
    unique_indices = sorted({int(idx) for idx in indices})
    colors = sns.color_palette(palette_name, n_colors=max(1, len(unique_indices)))
    return {idx: colors[pos] for pos, idx in enumerate(unique_indices)}


def _resolve_image_mode(
    available_conditions: list[str],
    image_mode: Literal["familiar", "novel", "both"] | None,
    include_novel_image: bool | None,
) -> list[str]:
    ordered = _resolve_condition_sequence(available_conditions)
    familiar_conditions = [condition for condition in ordered if _is_familiar_condition(condition)]
    novel_conditions = [condition for condition in ordered if _is_novel_condition(condition)]
    other_conditions = [condition for condition in ordered if condition not in familiar_conditions and condition not in novel_conditions]

    if image_mode is None:
        image_mode = "both" if include_novel_image else "familiar"
    if image_mode not in {"familiar", "novel", "both"}:
        raise ValueError("image_mode must be one of 'familiar', 'novel', or 'both'.")

    if image_mode == "familiar":
        selected = familiar_conditions or other_conditions
    elif image_mode == "novel":
        selected = novel_conditions or other_conditions
    else:
        selected = familiar_conditions + novel_conditions + other_conditions

    return selected or ordered


def _extract_repeated_y(
    long_df: DataFrame,
    condition: str,
    phase: str,
    image_type: str,
) -> np.ndarray:
    cell = long_df.loc[
        long_df["condition"].eq(condition)
        & long_df["experiment_phase"].eq(phase)
        & long_df["image_type"].eq(image_type),
        ["step", "y"],
    ].drop_duplicates()
    if cell.empty:
        return np.asarray([], dtype=float)
    return cell.sort_values("step")["y"].to_numpy(dtype=float)


def _build_trace_series_lookup(long_df: DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    required = {"condition", "experiment_phase", "image_type", "step", "y"}
    if not required.issubset(long_df.columns):
        return {}

    trace_df = (
        long_df[["condition", "experiment_phase", "image_type", "step", "y"]]
        .drop_duplicates()
        .sort_values(["condition", "experiment_phase", "image_type", "step"])
    )
    return {
        (str(condition), str(phase), str(image_type)): group["y"].to_numpy(dtype=float)
        for (condition, phase, image_type), group in trace_df.groupby(
            ["condition", "experiment_phase", "image_type"],
            sort=False,
            observed=True,
        )
    }


def _lookup_repeated_y(
    long_df: DataFrame,
    *,
    condition: str,
    phase: str,
    image_type: str,
    series_lookup: dict[tuple[str, str, str], np.ndarray] | None = None,
) -> np.ndarray:
    key = (str(condition), str(phase), str(image_type))
    if series_lookup is not None:
        return series_lookup.get(key, np.asarray([], dtype=float))
    return _extract_repeated_y(long_df, condition=condition, phase=phase, image_type=image_type)


def _extract_stimulus_intervals(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
) -> list[tuple[int, int]]:
    x = _to_np_2d(stim_pair[0])
    c = _to_np_2d(stim_pair[1])
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return []

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return []

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    return [(int(onset), int(offset)) for onset, offset in zip(onsets, offsets, strict=False)]


def _expand_window_to_event_bounds(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[int, int],
) -> tuple[int, int]:
    x = _to_np_2d(stim_pair[0])
    c = _to_np_2d(stim_pair[1])
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return focus_window

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return focus_window

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    if onsets.size == 0 or offsets.size == 0:
        return focus_window

    start, end = focus_window
    best_idx = None
    best_overlap = -1
    for idx, (onset, offset) in enumerate(zip(onsets, offsets, strict=False)):
        overlap = min(end, offset) - max(start, onset)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is None or best_overlap <= 0:
        return focus_window

    current_onset = int(onsets[best_idx])
    current_offset = int(offsets[best_idx])
    total_steps = int(stimulus_strength.size)

    if onsets.size > 1:
        period = int(round(float(np.median(np.diff(onsets)))))
        stimulus_len = int(round(float(np.median(offsets - onsets))))
        pre_stimulus_len = stimulus_len
        post_stimulus_len = max(0, period - pre_stimulus_len - stimulus_len)
        expanded_start = max(0, current_onset - pre_stimulus_len)
        expanded_end = min(total_steps, current_offset + post_stimulus_len)
        if expanded_end < current_offset:
            expanded_end = min(total_steps, current_offset)
        return expanded_start, expanded_end

    if start < current_onset and end > current_offset:
        return max(0, int(start)), min(total_steps, int(end))

    if current_onset > 0:
        stimulus_len = current_offset - current_onset
        expanded_start = max(0, current_onset - stimulus_len)
        expanded_end = min(total_steps, current_offset + 2 * stimulus_len)
        return expanded_start, expanded_end

    if best_idx > 0:
        prev_offset = int(offsets[best_idx - 1])
        expanded_start = int(round(0.5 * (prev_offset + current_onset)))
    else:
        expanded_start = 0

    if best_idx < len(onsets) - 1:
        next_onset = int(onsets[best_idx + 1])
        expanded_end = int(round(0.5 * (current_offset + next_onset)))
    else:
        expanded_end = int(stimulus_strength.size)

    return expanded_start, expanded_end


def _relative_seconds_ticks(xlim_seconds: tuple[float, float]) -> tuple[np.ndarray, list[str]]:
    tick_start = float(np.ceil(xlim_seconds[0]))
    tick_end = float(np.floor(xlim_seconds[1]))
    if tick_end < tick_start:
        ticks = np.asarray([xlim_seconds[0], xlim_seconds[1]], dtype=float)
    else:
        ticks = np.arange(tick_start, tick_end + 1.0, 1.0, dtype=float)

    labels = [str(int(round(tick))) if np.isclose(tick, round(tick)) else f"{tick:g}" for tick in ticks]
    return ticks, labels


def _window_repeated_trace(
    series: np.ndarray,
    *,
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[float, float],
) -> dict[str, np.ndarray | float | int] | None:
    if series.size == 0:
        return None

    intervals = _extract_stimulus_intervals(stim_pair)
    if not intervals:
        return None

    window_start, window_end = focus_window
    focus_interval = None
    for onset, offset in intervals:
        if onset < window_end and offset > window_start:
            focus_interval = (onset, offset)
            break
    if focus_interval is None:
        focus_interval = intervals[0]

    rel_start = int(round(window_start - focus_interval[0]))
    rel_end = int(round(window_end - focus_interval[0]))
    if rel_end <= rel_start:
        return None

    windows: list[np.ndarray] = []
    for onset, _ in intervals:
        start = onset + rel_start
        end = onset + rel_end
        if start < 0 or end > series.size:
            continue
        windows.append(series[start:end])

    if not windows:
        return None

    stacked = np.vstack(windows).astype(float)
    baseline_stop = max(0, min(-rel_start, stacked.shape[1]))
    if baseline_stop > 0:
        baseline_values = stacked[:, :baseline_stop].reshape(-1)
    else:
        baseline_values = stacked.reshape(-1)

    relative_step = np.arange(rel_start, rel_end, dtype=float)
    return {
        "stacked": stacked,
        "baseline_values": baseline_values,
        "stim_seconds": (
            0.0,
            float(focus_interval[1] - focus_interval[0]) / TIME_STEPS_PER_SECOND,
        ),
        "xlim_seconds": (
            float(rel_start) / TIME_STEPS_PER_SECOND,
            float(rel_end) / TIME_STEPS_PER_SECOND,
        ),
        "x_seconds": relative_step / TIME_STEPS_PER_SECOND,
    }


def _compute_baseline_stats(
    baseline_values: np.ndarray,
    *,
    zscore_std_floor: float | None = None,
) -> dict[str, float | int]:
    baseline_values = np.asarray(baseline_values, dtype=float).reshape(-1)
    baseline_n = int(baseline_values.size)
    baseline_mean = float(baseline_values.mean()) if baseline_n else 0.0
    baseline_std = float(baseline_values.std(ddof=1)) if baseline_n > 1 else 0.0
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_n": baseline_n,
        "zscore_std_floor": float(zscore_std_floor or 0.0),
    }


def _collect_shared_baseline_stats(
    long_df: DataFrame,
    *,
    trace_specs: list[tuple[str, str, str]],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    focus_window: tuple[float, float],
    series_lookup: dict[tuple[str, str, str], np.ndarray] | None = None,
    zscore_std_floor: float | None = None,
) -> dict[str, float | int] | None:
    baseline_chunks: list[np.ndarray] = []
    for condition, phase, image_type in trace_specs:
        stim_pair = stimuli.get(condition)
        if stim_pair is None:
            continue
        series = _lookup_repeated_y(
            long_df,
            condition=condition,
            phase=phase,
            image_type=image_type,
            series_lookup=series_lookup,
        )
        if series.size == 0:
            continue
        windowed = _window_repeated_trace(series, stim_pair=stim_pair, focus_window=focus_window)
        if windowed is None:
            continue
        baseline_values = np.asarray(windowed["baseline_values"], dtype=float).reshape(-1)
        if baseline_values.size:
            baseline_chunks.append(baseline_values)

    if not baseline_chunks:
        return None
    return _compute_baseline_stats(np.concatenate(baseline_chunks), zscore_std_floor=zscore_std_floor)


def _collect_naive_row_baseline_stats(
    long_df: DataFrame,
    *,
    selected_conditions: list[str],
    trace_types: tuple[str, ...],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    focus_window: tuple[float, float],
    series_lookup: dict[tuple[str, str, str], np.ndarray] | None = None,
    zscore_std_floor: float | None = None,
) -> dict[str, float | int] | None:
    return _collect_shared_baseline_stats(
        long_df,
        trace_specs=[
            (condition, "naive", trace_type)
            for condition in selected_conditions
            for trace_type in trace_types
        ],
        stimuli=stimuli,
        focus_window=focus_window,
        series_lookup=series_lookup,
        zscore_std_floor=zscore_std_floor,
    )


def _require_naive_row_baseline(
    baseline_stats: dict[str, float | int] | None,
    *,
    transition_name: str,
) -> dict[str, float | int]:
    if baseline_stats is None:
        raise ValueError(
            f"Cannot z-score transition '{transition_name}': no naive pre-stimulus baseline samples were found."
        )
    return baseline_stats


def _infer_row_zscore_std_floor(long_df: DataFrame, fallback: float | None = None) -> float:
    if "_zscore_std_floor" in long_df.columns:
        values = pd.to_numeric(long_df["_zscore_std_floor"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float(fallback or 0.0)


def _summarize_windowed_repeated_trace(
    long_df: DataFrame,
    *,
    condition: str,
    phase: str,
    image_type: str,
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[float, float],
    zscore: bool = True,
    baseline_stats: dict[str, float | int] | None = None,
    series_lookup: dict[tuple[str, str, str], np.ndarray] | None = None,
) -> dict[str, np.ndarray | float | int] | None:
    series = _lookup_repeated_y(
        long_df,
        condition=condition,
        phase=phase,
        image_type=image_type,
        series_lookup=series_lookup,
    )
    windowed = _window_repeated_trace(series, stim_pair=stim_pair, focus_window=focus_window)
    if windowed is None:
        return None

    stacked = np.asarray(windowed["stacked"], dtype=float)
    if baseline_stats is None:
        baseline_stats = _compute_baseline_stats(np.asarray(windowed["baseline_values"], dtype=float))

    baseline_mean = float(baseline_stats["baseline_mean"])
    baseline_std = float(baseline_stats["baseline_std"])
    baseline_n = int(baseline_stats["baseline_n"])
    if zscore:
        zscore_std_floor = float(baseline_stats.get("zscore_std_floor", 0.0))
        scale = max(baseline_std, zscore_std_floor) if np.isfinite(baseline_std) and baseline_std > 1e-12 else max(1.0, zscore_std_floor)
        summarized = (stacked - baseline_mean) / scale
    else:
        summarized = stacked

    y_mean = summarized.mean(axis=0)
    if summarized.shape[0] > 1:
        y_sem = summarized.std(axis=0, ddof=1) / np.sqrt(summarized.shape[0])
    else:
        y_sem = np.zeros_like(y_mean)

    return {
        "x_seconds": np.asarray(windowed["x_seconds"], dtype=float),
        "y_mean": y_mean,
        "y_sem": y_sem,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "zscore_scale": scale if zscore else 1.0,
        "zscore_std_floor": float(baseline_stats.get("zscore_std_floor", 0.0)),
        "baseline_n": baseline_n,
        "n_trials": int(stacked.shape[0]),
        "stim_seconds": tuple(windowed["stim_seconds"]),
        "xlim_seconds": tuple(windowed["xlim_seconds"]),
    }


def _build_windowed_transition_export(
    long_dfs_by_transition: dict[str, DataFrame],
    *,
    ordered_transitions: list[str],
    labels: dict[str, str],
    phases: list[str],
    selected_conditions: list[str],
    baseline_conditions: list[str] | None = None,
    trace_types: tuple[str, ...],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    plot_window: tuple[float, float],
    zscore_activity: bool,
    zscore_std_floor: float | None = None,
) -> DataFrame:
    export_frames: list[DataFrame] = []

    for transition_row, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        series_lookup = _build_trace_series_lookup(long_df)
        row_baseline_stats = _collect_naive_row_baseline_stats(
            long_df,
            selected_conditions=baseline_conditions or selected_conditions,
            trace_types=trace_types,
            stimuli=stimuli,
            focus_window=plot_window,
            series_lookup=series_lookup,
            zscore_std_floor=_infer_row_zscore_std_floor(long_df, zscore_std_floor),
        )
        if zscore_activity:
            row_baseline_stats = _require_naive_row_baseline(
                row_baseline_stats,
                transition_name=transition_name,
            )
        for phase_index, phase in enumerate(phases):
            for condition_index, condition in enumerate(selected_conditions):
                stim_pair = stimuli.get(condition)
                if stim_pair is None:
                    continue
                for trace_index, trace_type in enumerate(trace_types):
                    summary = _summarize_windowed_repeated_trace(
                        long_df,
                        condition=condition,
                        phase=phase,
                        image_type=trace_type,
                        stim_pair=stim_pair,
                        focus_window=plot_window,
                        zscore=zscore_activity,
                        baseline_stats=row_baseline_stats,
                        series_lookup=series_lookup,
                    )
                    if summary is None:
                        continue
                    y_mean = np.asarray(summary["y_mean"], dtype=float)
                    y_sem = np.asarray(summary["y_sem"], dtype=float)
                    x_seconds = np.asarray(summary["x_seconds"], dtype=float)
                    stim_start, stim_end = summary["stim_seconds"]

                    export_frames.append(
                        pd.DataFrame(
                            {
                                "time_seconds": x_seconds,
                                "y": y_mean,
                                "y_sem": y_sem,
                            }
                        ).assign(
                            transition=transition_name,
                            transition_label=labels.get(transition_name, transition_name),
                            transition_row=transition_row,
                            experiment_phase=phase,
                            phase_label=PHASE_DISPLAY_LABELS.get(phase, phase.title()),
                            phase_index=phase_index,
                            condition=condition,
                            condition_label=_display_condition_label(condition),
                            condition_index=condition_index,
                            image_type=trace_type,
                            trace_label=TRACE_LABELS.get(trace_type, trace_type),
                            trace_index=trace_index,
                            baseline_mean=float(summary["baseline_mean"]),
                            baseline_std=float(summary["baseline_std"]),
                            zscore_scale=float(summary["zscore_scale"]),
                            zscore_std_floor=float(summary["zscore_std_floor"]),
                            baseline_n=int(summary["baseline_n"]),
                            n_trials=int(summary["n_trials"]),
                            stim_start_seconds=float(stim_start),
                            stim_end_seconds=float(stim_end),
                        )
                    )

    if not export_frames:
        return pd.DataFrame(
            columns=[
                "transition",
                "transition_label",
                "transition_row",
                "experiment_phase",
                "phase_label",
                "phase_index",
                "condition",
                "condition_label",
                "condition_index",
                "image_type",
                "trace_label",
                "trace_index",
                "time_seconds",
                "y",
                "y_sem",
                "baseline_mean",
                "baseline_std",
                "zscore_scale",
                "zscore_std_floor",
                "baseline_n",
                "n_trials",
                "experiment_series",
                "seed",
                "stim_start_seconds",
                "stim_end_seconds",
            ]
        )

    export_df = pd.concat(export_frames, ignore_index=True)
    ordered_columns = [
        "transition",
        "transition_label",
        "transition_row",
        "experiment_phase",
        "phase_label",
        "phase_index",
        "condition",
        "condition_label",
        "condition_index",
        "image_type",
        "trace_label",
        "trace_index",
        "time_seconds",
        "y",
        "y_sem",
        "baseline_mean",
        "baseline_std",
        "zscore_scale",
        "zscore_std_floor",
        "baseline_n",
        "n_trials",
        "stim_start_seconds",
        "stim_end_seconds",
    ]
    optional_columns = [column for column in ("experiment_series", "seed") if column in export_df.columns]
    return export_df.loc[:, ordered_columns + optional_columns]


def _build_response_transition_export(
    long_dfs_by_transition: dict[str, DataFrame],
    *,
    ordered_transitions: list[str],
    labels: dict[str, str],
    selected_conditions: list[str],
    baseline_conditions: list[str] | None = None,
    column_specs: tuple[dict[str, str], ...],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    plot_window: tuple[float, float],
    zscore_activity: bool,
    zscore_std_floor: float | None = None,
) -> DataFrame:
    export_frames: list[DataFrame] = []

    for transition_row, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        series_lookup = _build_trace_series_lookup(long_df)
        response_trace_types = tuple(
            dict.fromkeys(
                trace_type
                for column_spec in column_specs
                for trace_type in (column_spec["o_trace"], column_spec["no_trace"])
            )
        )
        row_baseline_stats = _collect_naive_row_baseline_stats(
            long_df,
            selected_conditions=baseline_conditions or selected_conditions,
            trace_types=response_trace_types,
            stimuli=stimuli,
            focus_window=plot_window,
            series_lookup=series_lookup,
            zscore_std_floor=_infer_row_zscore_std_floor(long_df, zscore_std_floor),
        )
        if zscore_activity:
            row_baseline_stats = _require_naive_row_baseline(
                row_baseline_stats,
                transition_name=transition_name,
            )
        for column_index, column_spec in enumerate(column_specs):
            phase = column_spec["phase"]
            trace_specs = [
                ("O", column_spec["o_trace"], "O"),
                ("NO", column_spec["no_trace"], "NO"),
            ]
            for condition_index, condition in enumerate(selected_conditions):
                stim_pair = stimuli.get(condition)
                if stim_pair is None:
                    continue
                for response_index, (response_type, trace_type, trace_label) in enumerate(trace_specs):
                    summary = _summarize_windowed_repeated_trace(
                        long_df,
                        condition=condition,
                        phase=phase,
                        image_type=trace_type,
                        stim_pair=stim_pair,
                        focus_window=plot_window,
                        zscore=zscore_activity,
                        baseline_stats=row_baseline_stats,
                        series_lookup=series_lookup,
                    )
                    if summary is None:
                        continue
                    stim_start, stim_end = summary["stim_seconds"]
                    export_frames.append(
                        pd.DataFrame(
                            {
                                "time_seconds": np.asarray(summary["x_seconds"], dtype=float),
                                "y": np.asarray(summary["y_mean"], dtype=float),
                                "y_sem": np.asarray(summary["y_sem"], dtype=float),
                            }
                        ).assign(
                            transition=transition_name,
                            transition_label=labels.get(transition_name, transition_name),
                            transition_row=transition_row,
                            column_key=column_spec["key"],
                            column_label=column_spec["label"],
                            column_index=column_index,
                            experiment_phase=phase,
                            condition=condition,
                            condition_label=_display_condition_label(condition),
                            condition_index=condition_index,
                            response_type=response_type,
                            response_label=trace_label,
                            response_index=response_index,
                            image_type=trace_type,
                            baseline_mean=float(summary["baseline_mean"]),
                            baseline_std=float(summary["baseline_std"]),
                            zscore_scale=float(summary["zscore_scale"]),
                            zscore_std_floor=float(summary["zscore_std_floor"]),
                            baseline_n=int(summary["baseline_n"]),
                            n_trials=int(summary["n_trials"]),
                            stim_start_seconds=float(stim_start),
                            stim_end_seconds=float(stim_end),
                        )
                    )

    if not export_frames:
        return pd.DataFrame(
            columns=[
                "transition",
                "transition_label",
                "transition_row",
                "column_key",
                "column_label",
                "column_index",
                "experiment_phase",
                "condition",
                "condition_label",
                "condition_index",
                "response_type",
                "response_label",
                "response_index",
                "image_type",
                "time_seconds",
                "y",
                "y_sem",
                "baseline_mean",
                "baseline_std",
                "zscore_scale",
                "zscore_std_floor",
                "baseline_n",
                "n_trials",
                "stim_start_seconds",
                "stim_end_seconds",
            ]
        )

    ordered_columns = [
        "transition",
        "transition_label",
        "transition_row",
        "column_key",
        "column_label",
        "column_index",
        "experiment_phase",
        "condition",
        "condition_label",
        "condition_index",
        "response_type",
        "response_label",
        "response_index",
        "image_type",
        "time_seconds",
        "y",
        "y_sem",
        "baseline_mean",
        "baseline_std",
        "zscore_scale",
        "zscore_std_floor",
        "baseline_n",
        "n_trials",
        "stim_start_seconds",
        "stim_end_seconds",
    ]
    export_df = pd.concat(export_frames, ignore_index=True)
    return export_df.loc[:, ordered_columns]


def visualize_transition_response_matrix(
    long_dfs_by_transition: dict[str, DataFrame],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    save_path: str = PLOTSDIR,
    name: str,
    image_mode: Literal["familiar", "novel"],
    transition_order: list[str] | None = None,
    transition_labels: dict[str, str] | None = None,
    step_window: tuple[int, int] = (1000, 1350),
    save_in_transition_subdir: bool = True,
    save_csv: bool = True,
    zscore_activity: bool = True,
    image_format: str = "png",
    figure_size_inches: tuple[float, float] | None = None,
    zscore_std_floor: float | None = None,
) -> list[str]:
    if not long_dfs_by_transition:
        raise ValueError("long_dfs_by_transition must contain at least one transition result.")
    if image_mode not in {"familiar", "novel"}:
        raise ValueError("image_mode must be 'familiar' or 'novel'.")

    ordered_transitions = resolve_transition_order(long_dfs_by_transition, transition_order)
    labels = resolve_transition_labels(ordered_transitions, transition_labels)

    sample_df = long_dfs_by_transition[ordered_transitions[0]]
    phases = set(_resolve_phase_sequence(sample_df))
    column_specs = tuple(
        column_spec
        for column_spec in TRANSITION_RESPONSE_COLUMN_SPECS
        if column_spec["phase"] in phases
    )
    if not column_specs:
        raise ValueError("Transition response matrix requires naive/expert experiment phases.")

    phase_filtered_df = sample_df.loc[sample_df["experiment_phase"].isin(phases)].copy()
    available_conditions = _resolve_condition_sequence(
        phase_filtered_df["condition"].dropna().astype(str).unique().tolist()
        if "condition" in phase_filtered_df.columns
        else [],
        preferred=list(STIMULI),
    )
    selected_conditions = _resolve_image_mode(
        available_conditions=available_conditions or list(STIMULI),
        image_mode=image_mode,
        include_novel_image=(image_mode == "novel"),
    )
    baseline_conditions = available_conditions or list(STIMULI)
    layout_conditions = _transition_response_layout_conditions(
        image_mode=image_mode,
        available_conditions=available_conditions or list(STIMULI),
        selected_conditions=selected_conditions,
    )

    display_windows = [
        _expand_window_to_event_bounds(STIMULI[condition], focus_window=step_window)
        for condition in selected_conditions
        if condition in STIMULI
    ]
    if display_windows:
        plot_window = (
            min(window[0] for window in display_windows),
            max(window[1] for window in display_windows),
        )
    else:
        plot_window = step_window

    condition_summaries: dict[str, dict[str, np.ndarray | float | int]] = {}
    sample_lookup = _build_trace_series_lookup(sample_df)
    for condition in selected_conditions:
        stim_pair = STIMULI.get(condition)
        if stim_pair is None:
            continue
        summary = _summarize_windowed_repeated_trace(
            sample_df,
            condition=condition,
            phase=column_specs[0]["phase"],
            image_type=column_specs[0]["no_trace"],
            stim_pair=stim_pair,
            focus_window=plot_window,
            zscore=zscore_activity,
            series_lookup=sample_lookup,
        )
        if summary is not None:
            condition_summaries[condition] = summary

    if not condition_summaries:
        raise ValueError("STIMULI must contain at least one of the requested conditions.")

    stim_windows = {
        condition: tuple(summary["stim_seconds"])
        for condition, summary in condition_summaries.items()
    }
    xlim_transition = (
        min(float(summary["xlim_seconds"][0]) for summary in condition_summaries.values()),
        max(float(summary["xlim_seconds"][1]) for summary in condition_summaries.values()),
    )

    center_condition_by_group = {
        group_idx: selected_conditions[0]
        for group_idx in range(len(column_specs))
        if image_mode == "novel" and len(layout_conditions) > len(selected_conditions) and selected_conditions
    }
    center_single_condition_slots = bool(center_condition_by_group)
    center_slot = len(layout_conditions) // 2 - 1 if len(layout_conditions) % 2 == 0 else len(layout_conditions) // 2
    column_condition_specs = [
        (
            column_spec,
            layout_condition,
            center_condition_by_group.get(group_idx) if slot_idx == center_slot else None,
        )
        if group_idx in center_condition_by_group
        else (column_spec, layout_condition, layout_condition)
        for group_idx, column_spec in enumerate(column_specs)
        for slot_idx, layout_condition in enumerate(layout_conditions)
    ]
    n_rows = len(ordered_transitions)
    n_cols = len(column_condition_specs)
    n_slots_per_group = len(layout_conditions)
    visible_slot_indices = [
        idx
        for idx, (_, _, plot_condition) in enumerate(column_condition_specs)
        if plot_condition is not None
    ]
    title_specs = [
        (idx, _display_condition_label(plot_condition))
        for idx, (_, _, plot_condition) in enumerate(column_condition_specs)
        if plot_condition is not None
    ]
    hidden_slot_indices = [
        idx
        for idx, (_, _, plot_condition) in enumerate(column_condition_specs)
        if plot_condition is None
    ]
    group_spans = [
        (group_idx * n_slots_per_group, group_idx * n_slots_per_group + n_slots_per_group - 1)
        for group_idx in range(len(column_specs))
    ]
    condition_title_size, column_group_size = _matrix_header_fontsizes(n_cols)
    fig_width = max(12.0, 3.4 * n_cols + 3.2)
    fig_height = max(6.0, 2.15 * n_rows + 1.9)
    figsize = figure_size_inches or (fig_width, fig_height)
    size_scale = min(figsize[0] / fig_width, figsize[1] / fig_height)
    compact = figure_size_inches is not None
    condition_title_size = max(condition_title_size * size_scale, 3.5) if compact else condition_title_size
    column_group_size = max(column_group_size * size_scale, 3.5) if compact else column_group_size
    legend_fontsize = max(18 * size_scale, 3.5) if compact else 18
    legend_linewidth = max(4.0 * size_scale, 0.7) if compact else 4.0
    trace_linewidth = max(4.0 * size_scale, 0.7) if compact else 4.0
    guide_linewidth = max(0.6 * size_scale, 0.2) if compact else 0.6
    row_label_fontsize = max(20 * size_scale, 6.0) if compact else 20
    scale_bar_linewidth = max(2.2 * size_scale, 0.5) if compact else 2.2
    title_pad = max(10 * size_scale, 1.0) if compact else 10
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.8, bottom=0.055, wspace=0.12, hspace=0.2)

    legend_handles = [
        Line2D([0], [0], color="red", lw=legend_linewidth, label="O"),
        Line2D([0], [0], color="black", lw=legend_linewidth, label="NO"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        frameon=False,
        ncol=2,
        handlelength=2.0 * size_scale,
        columnspacing=1.2 * size_scale,
        fontsize=legend_fontsize,
    )

    for col_idx, title in title_specs:
        axes[0, col_idx].set_title(title, fontsize=condition_title_size, pad=title_pad)
    for col_idx in hidden_slot_indices:
        axes[0, col_idx].set_title("")
    for row_idx in range(n_rows):
        for col_idx in hidden_slot_indices:
            axes[row_idx, col_idx].set_visible(False)

    # Previously, when image_mode="novel" was rendered onto a 2-slot familiar
    # layout, the single novel axis was stretched across both Fam1 and Fam2 slots
    # via `_span_axis_across`. We now leave the novel response in its slot (Fam1)
    # and the Fam2 slot stays empty rather than being absorbed into a wide axis.
    pass

    for group_idx, column_spec in enumerate(column_specs):
        start_col, end_col = group_spans[group_idx]
        x_center = 0.5 * (axes[0, start_col].get_position().x0 + axes[0, end_col].get_position().x1)
        fig.text(
            x_center,
            0.955,
            column_spec["label"],
            ha="center",
            va="center",
            fontsize=column_group_size,
        )

    for row_idx, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        series_lookup = _build_trace_series_lookup(long_df)
        row_bounds: list[tuple[float, float]] = []
        response_trace_types = tuple(
            dict.fromkeys(
                trace_type
                for column_spec in column_specs
                for trace_type in (column_spec["o_trace"], column_spec["no_trace"])
            )
        )
        row_baseline_stats = _collect_naive_row_baseline_stats(
            long_df,
            selected_conditions=baseline_conditions,
            trace_types=response_trace_types,
            stimuli=STIMULI,
            focus_window=plot_window,
            series_lookup=series_lookup,
            zscore_std_floor=_infer_row_zscore_std_floor(long_df, zscore_std_floor),
        )
        if zscore_activity:
            row_baseline_stats = _require_naive_row_baseline(
                row_baseline_stats,
                transition_name=transition_name,
            )

        for col_idx, (column_spec, _, condition) in enumerate(column_condition_specs):
            ax = axes[row_idx, col_idx]
            if condition is None:
                continue
            # Stretching of the novel axis across the Fam1+Fam2 slots intentionally
            # removed: novel sits in slot 0 (Fam1) and slot 1 (Fam2) stays empty.
            stim_interval = stim_windows.get(condition)
            if condition not in STIMULI:
                ax.set_visible(False)
                continue

            if stim_interval is not None:
                ax.axvspan(stim_interval[0], stim_interval[1], color="0.92", zorder=0)
            ax.axhline(0.0, color="0.85", lw=guide_linewidth, zorder=0)

            for response_type, trace_type, color in (
                ("O", column_spec["o_trace"], "red"),
                ("NO", column_spec["no_trace"], "black"),
            ):
                summary = _summarize_windowed_repeated_trace(
                    long_df,
                    condition=condition,
                    phase=column_spec["phase"],
                    image_type=trace_type,
                    stim_pair=STIMULI[condition],
                    focus_window=plot_window,
                    zscore=zscore_activity,
                    baseline_stats=row_baseline_stats,
                    series_lookup=series_lookup,
                )
                if summary is None:
                    continue
                y_mean = np.asarray(summary["y_mean"], dtype=float)
                ax.plot(
                    np.asarray(summary["x_seconds"], dtype=float),
                    y_mean,
                    color=color,
                    lw=trace_linewidth,
                    label=response_type,
                )
                row_bounds.append((float(np.min(y_mean)), float(np.max(y_mean))))

            ax.set_xlim(*xlim_transition)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        label_ax = axes[row_idx, 0]
        label_ax.text(
            -0.13,
            0.5,
            labels.get(transition_name, transition_name),
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=row_label_fontsize,
        )
        if row_bounds:
            row_min = min(bound[0] for bound in row_bounds)
            row_max = max(bound[1] for bound in row_bounds)
            span = row_max - row_min
            if span <= 0:
                span = max(abs(row_min), abs(row_max), 0.1)
            pad = 0.12 * span
            for ax in axes[row_idx, :]:
                if ax.get_visible():
                    ax.set_ylim(row_min - pad, row_max + pad)
            if zscore_activity:
                _add_row_scale_bar(axes[row_idx, :], linewidth=scale_bar_linewidth)

    if save_in_transition_subdir:
        plot_dirs = _resolve_plot_dirs(save_path)
        output_dir = plot_dirs["transition_panels"]
    else:
        output_dir = save_path
        os.makedirs(output_dir, exist_ok=True)

    base_path = os.path.join(output_dir, name)
    saved_paths: list[str] = []
    for ext in (image_format,):
        out_path = f"{base_path}.{ext}"
        fig.savefig(out_path)
        saved_paths.append(out_path)
    plt.close(fig)

    if save_csv:
        export_df = _build_response_transition_export(
            long_dfs_by_transition,
            ordered_transitions=ordered_transitions,
            labels=labels,
            selected_conditions=selected_conditions,
            baseline_conditions=baseline_conditions,
            column_specs=column_specs,
            stimuli=STIMULI,
            plot_window=plot_window,
            zscore_activity=zscore_activity,
            zscore_std_floor=zscore_std_floor,
        )
        export_df.to_csv(f"{base_path}.csv", index=False)
        saved_paths.append(f"{base_path}.csv")

    return saved_paths


def _plot_panel_a_activity(
    ax_grid: np.ndarray,
    y_df: DataFrame,
    activity_layout: list[tuple[str, str]],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    xlim: tuple[float, float],
    *,
    zscore_activity: bool = True,
    include_novel_no_context: bool = False,
    image_types: list[str] | None = None,
) -> None:
    flat_axes = np.asarray(ax_grid).reshape(-1)
    if flat_axes.size == 0 or not activity_layout:
        return

    for ax in flat_axes[len(activity_layout):]:
        ax.set_visible(False)

    ref_ax = flat_axes[0]
    for ax in flat_axes[1:len(activity_layout)]:
        ax.sharex(ref_ax)
        ax.sharey(ref_ax)

    available_image_types = set(image_types or y_df["image_type"].dropna().unique().tolist())
    trace_types = _resolve_trace_types(
        available_image_types,
        include_no_response_ablations=include_novel_no_context,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRACE_COLORS[trace_type],
            linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
            lw=5.0,
            label=TRACE_LABELS[trace_type],
        )
        for trace_type in trace_types
        if trace_type in TRACE_COLORS
    ]

    selected_conditions = list(
        dict.fromkeys(condition for condition, _ in activity_layout if condition in STIMULI)
    )
    row_baseline_stats = _collect_naive_row_baseline_stats(
        y_df,
        selected_conditions=selected_conditions,
        trace_types=trace_types,
        stimuli=STIMULI,
        focus_window=xlim,
    )
    if zscore_activity:
        row_baseline_stats = _require_naive_row_baseline(
            row_baseline_stats,
            transition_name="panel A activity",
        )

    xlims_seconds: list[tuple[float, float]] = []
    stim_windows_seconds: dict[str, tuple[float, float] | None] = {}
    for condition, _ in activity_layout:
        if condition not in STIMULI:
            continue
        summary = _summarize_windowed_repeated_trace(
            y_df,
            condition=condition,
            phase=activity_layout[0][1],
            image_type="full",
            stim_pair=STIMULI[condition],
            focus_window=xlim,
            zscore=zscore_activity,
            baseline_stats=row_baseline_stats,
        )
        if summary is None:
            continue
        xlims_seconds.append(summary["xlim_seconds"])
        stim_windows_seconds[condition] = summary["stim_seconds"]

    if xlims_seconds:
        xlim_seconds = (
            min(bounds[0] for bounds in xlims_seconds),
            max(bounds[1] for bounds in xlims_seconds),
        )
    else:
        xlim_seconds = (
            float(xlim[0]) / TIME_STEPS_PER_SECOND,
            float(xlim[1]) / TIME_STEPS_PER_SECOND,
        )
    xticks, xticklabels = _relative_seconds_ticks(xlim_seconds)

    global_y_bounds: list[tuple[float, float]] = []
    for idx, (condition, phase) in enumerate(activity_layout):
        ax = flat_axes[idx]
        trace_specs = [
            (trace_type, TRACE_COLORS.get(trace_type, "black"))
            for trace_type in trace_types
        ]

        has_trace = False
        for image_type, color in trace_specs:
            if condition not in STIMULI:
                continue
            summary = _summarize_windowed_repeated_trace(
                y_df,
                condition=condition,
                phase=phase,
                image_type=image_type,
                stim_pair=STIMULI[condition],
                focus_window=xlim,
                zscore=zscore_activity,
                baseline_stats=row_baseline_stats,
            )
            if summary is None:
                continue
            has_trace = True
            ax.plot(
                np.asarray(summary["x_seconds"], dtype=float),
                np.asarray(summary["y_mean"], dtype=float),
                color=color,
                linestyle=TRACE_LINESTYLES.get(image_type, "-"),
                lw=5.0,
            )
            global_y_bounds.append(
                (
                    float(np.min(np.asarray(summary["y_mean"], dtype=float))),
                    float(np.max(np.asarray(summary["y_mean"], dtype=float))),
                )
            )

        if not has_trace:
            ax.set_visible(False)
            continue

        stim_interval = stim_windows_seconds.get(condition)
        if stim_interval is not None:
            ax.axvspan(
                stim_interval[0],
                stim_interval[1],
                ymin=0.02,
                ymax=0.055,
                color="#8c5a2b",
                clip_on=False,
                zorder=3,
            )

        ax.set_xlim(xlim_seconds)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_title(
            f"{_display_condition_label(condition)} | {PHASE_DISPLAY_LABELS.get(phase, phase.title())}",
            fontsize=19,
            pad=10,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(5.0)
        ax.spines["bottom"].set_linewidth(5.0)
        ax.tick_params(axis="both", width=1.6, length=5, labelsize=AXIS_TICK_FONTSIZE)
        ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.set_ylabel("Z-scored activity" if zscore_activity else "Activation", fontsize=AXIS_LABEL_FONTSIZE)
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                handlelength=2.0,
                borderaxespad=0.2,
                fontsize=15,
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        _style_axis_fonts(ax)

    if global_y_bounds:
        y_min = min(bound[0] for bound in global_y_bounds)
        y_max = max(bound[1] for bound in global_y_bounds)
        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_min), abs(y_max), 0.1)
        pad = 0.08 * span
        for ax in flat_axes[:len(activity_layout)]:
            if ax.get_visible():
                ax.set_ylim(y_min - pad, y_max + pad)


def visualize_transition_panel(
    long_dfs_by_transition: dict[str, DataFrame],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str = PLOTSDIR,
    name: str = "transition_panel",
    image_mode: Literal["familiar", "novel", "both"] | None = None,
    include_novel_image: bool | None = None,
    transition_order: list[str] | None = None,
    transition_labels: dict[str, str] | None = None,
    trace_types: tuple[str, ...] | None = None,
    step_window: tuple[int, int] = (1000, 1350),
    save_in_transition_subdir: bool = True,
    save_csv: bool = True,
    zscore_activity: bool = True,
    image_format: str = "png",
) -> str:
    if not long_dfs_by_transition:
        raise ValueError("long_dfs_by_transition must contain at least one transition result.")

    ordered_transitions = resolve_transition_order(long_dfs_by_transition, transition_order)
    labels = resolve_transition_labels(ordered_transitions, transition_labels)

    sample_df = long_dfs_by_transition[ordered_transitions[0]]
    phases = _resolve_phase_sequence(sample_df)
    if not phases:
        raise ValueError("Transition panel requires experiment_phase values.")

    phase_filtered_df = sample_df.loc[sample_df["experiment_phase"].isin(phases)].copy()
    available_conditions = _resolve_condition_sequence(
        phase_filtered_df["condition"].dropna().astype(str).unique().tolist() if "condition" in phase_filtered_df.columns else [],
        preferred=list(STIMULI),
    )
    available_image_types = phase_filtered_df["image_type"].dropna().astype(str).unique().tolist() if "image_type" in phase_filtered_df.columns else []
    resolved_trace_types = trace_types or _resolve_trace_types(
        available_image_types,
        include_no_response_ablations=True,
    )
    selected_conditions = _resolve_image_mode(
        available_conditions=available_conditions or list(STIMULI),
        image_mode=image_mode,
        include_novel_image=include_novel_image,
    )
    baseline_conditions = available_conditions or list(STIMULI)

    display_windows = [
        _expand_window_to_event_bounds(STIMULI[condition], focus_window=step_window)
        for condition in selected_conditions
        if condition in STIMULI
    ]
    if display_windows:
        plot_window = (
            min(window[0] for window in display_windows),
            max(window[1] for window in display_windows),
        )
    else:
        plot_window = step_window

    condition_summaries: dict[str, dict[str, np.ndarray | float | int]] = {}
    sample_lookup = _build_trace_series_lookup(sample_df)
    for condition in selected_conditions:
        stim_pair = STIMULI.get(condition)
        if stim_pair is None:
            continue
        summary = _summarize_windowed_repeated_trace(
            sample_df,
            condition=condition,
            phase=phases[0],
            image_type=resolved_trace_types[0],
            stim_pair=stim_pair,
            focus_window=plot_window,
            zscore=zscore_activity,
            series_lookup=sample_lookup,
        )
        if summary is not None:
            condition_summaries[condition] = summary

    if not condition_summaries:
        raise ValueError("STIMULI must contain at least one of the requested conditions.")

    stim_windows = {
        condition: tuple(summary["stim_seconds"])
        for condition, summary in condition_summaries.items()
    }
    xlim_transition = (
        min(float(summary["xlim_seconds"][0]) for summary in condition_summaries.values()),
        max(float(summary["xlim_seconds"][1]) for summary in condition_summaries.values()),
    )

    column_specs = [(phase, condition) for phase in phases for condition in selected_conditions]
    n_rows = len(ordered_transitions)
    n_cols = len(column_specs)
    transition_export_df = None
    if save_csv:
        transition_export_df = _build_windowed_transition_export(
            long_dfs_by_transition,
            ordered_transitions=ordered_transitions,
            labels=labels,
            phases=phases,
            selected_conditions=selected_conditions,
            baseline_conditions=baseline_conditions,
            trace_types=resolved_trace_types,
            stimuli=STIMULI,
            plot_window=plot_window,
            zscore_activity=zscore_activity,
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols + 1.8, 3 * n_rows + 1.9),
        squeeze=False,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.89, bottom=0.05, wspace=0.12, hspace=0.18)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRACE_COLORS[trace_type],
            linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
            lw=1.6,
            label=TRACE_LABELS.get(trace_type, trace_type),
        )
        for trace_type in resolved_trace_types
        if trace_type in TRACE_COLORS
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.985),
            frameon=False,
            ncol=len(legend_handles),
            handlelength=2.0,
            columnspacing=1.2,
        )

    for col_idx, (phase, condition) in enumerate(column_specs):
        axes[0, col_idx].set_title(_display_condition_label(condition), fontsize=32, pad=12)

    for phase_idx, phase in enumerate(phases):
        start_col = phase_idx * len(selected_conditions)
        end_col = start_col + len(selected_conditions) - 1
        x_center = 0.5 * (axes[0, start_col].get_position().x0 + axes[0, end_col].get_position().x1)
        fig.text(
            x_center,
            0.945,
            PHASE_DISPLAY_LABELS.get(phase, phase.title()),
            ha="center",
            va="center",
            fontsize=32,
        )

    for row_idx, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        series_lookup = _build_trace_series_lookup(long_df)
        row_bounds: list[tuple[float, float]] = []
        row_baseline_stats = _collect_naive_row_baseline_stats(
            long_df,
            selected_conditions=selected_conditions,
            trace_types=resolved_trace_types,
            stimuli=STIMULI,
            focus_window=plot_window,
            series_lookup=series_lookup,
        )
        if zscore_activity:
            row_baseline_stats = _require_naive_row_baseline(
                row_baseline_stats,
                transition_name=transition_name,
            )

        for col_idx, (phase, condition) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            stim_interval = stim_windows.get(condition)
            if condition not in STIMULI:
                ax.set_visible(False)
                continue

            if stim_interval is not None:
                ax.axvspan(stim_interval[0], stim_interval[1], color="0.9", zorder=0)
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)

            for trace_type in resolved_trace_types:
                summary = _summarize_windowed_repeated_trace(
                    long_df,
                    condition=condition,
                    phase=phase,
                    image_type=trace_type,
                    stim_pair=STIMULI[condition],
                    focus_window=plot_window,
                    zscore=zscore_activity,
                    baseline_stats=row_baseline_stats,
                    series_lookup=series_lookup,
                )
                if summary is None:
                    continue
                ax.plot(
                    np.asarray(summary["x_seconds"], dtype=float),
                    np.asarray(summary["y_mean"], dtype=float),
                    color=TRACE_COLORS.get(trace_type, "black"),
                    linestyle=TRACE_LINESTYLES.get(trace_type, "-"),
                    lw=5,
                )
                row_bounds.append(
                    (
                        float(np.min(np.asarray(summary["y_mean"], dtype=float))),
                        float(np.max(np.asarray(summary["y_mean"], dtype=float))),
                    )
                )

            ax.set_xlim(*xlim_transition)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        label_ax = axes[row_idx, 0]
        label_ax.text(
            -0.12,
            0.5,
            labels.get(transition_name, transition_name),
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=32,
        )
        if row_bounds:
            row_min = min(bound[0] for bound in row_bounds)
            row_max = max(bound[1] for bound in row_bounds)
            span = row_max - row_min
            if span <= 0:
                span = max(abs(row_min), abs(row_max), 0.1)
            pad = 0.12 * span
            for ax in axes[row_idx, :]:
                if ax.get_visible():
                    ax.set_ylim(row_min - pad, row_max + pad)
            if zscore_activity:
                _add_row_scale_bar(axes[row_idx, :])

    if save_in_transition_subdir:
        plot_dirs = _resolve_plot_dirs(save_path)
        out_path = os.path.join(plot_dirs["transition_panels"], f"{name}_{'_'.join(selected_conditions)}.{image_format}")
    else:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, f"{name}_{'_'.join(selected_conditions)}.{image_format}")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    if transition_export_df is not None:
        transition_export_df.to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    return out_path


def save_grouped_transition_panels(
    long_dfs_by_transition: dict[str, DataFrame],
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str,
    transition_order: list[str],
    transition_labels: dict[str, str] | None = None,
    save_in_transition_subdir: bool = True,
    step_window: tuple[int, int] = (1000, 1350),
    zscore_activity: bool = True,
    image_format: str = "png",
    zscore_std_floor: float | None = None,
) -> None:
    sample_df = next(iter(long_dfs_by_transition.values()), None)
    if sample_df is None:
        return

    combined_transitions: dict[str, DataFrame] = {}
    for transition_name in transition_order:
        long_df = long_dfs_by_transition.get(transition_name)
        if long_df is None:
            continue

        combined = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
        if combined.empty:
            continue

        combined_transitions[transition_name] = combined

    if combined_transitions:
        ordered_combined = [name for name in transition_order if name in combined_transitions]
        combined_labels = resolve_transition_labels(ordered_combined, transition_labels)
        visualize_transition_panel(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transition_panel_naive_expert",
            image_mode="both",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            step_window=step_window,
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
            zscore_activity=zscore_activity,
            image_format=image_format,
        )
        visualize_transition_response_matrix(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transitions_FAM",
            image_mode="familiar",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            step_window=step_window,
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
            zscore_activity=zscore_activity,
            image_format=image_format,
            zscore_std_floor=zscore_std_floor,
        )
        visualize_transition_response_matrix(
            combined_transitions,
            STIMULI=stimuli,
            save_path=save_path,
            name="transitions_NOV",
            image_mode="novel",
            transition_order=ordered_combined,
            transition_labels=combined_labels,
            step_window=step_window,
            save_in_transition_subdir=save_in_transition_subdir,
            save_csv=True,
            zscore_activity=zscore_activity,
            image_format=image_format,
            zscore_std_floor=zscore_std_floor,
        )


def wide_to_long(DF: DataFrame) -> DataFrame:
    if "step" not in DF.columns:
        raise ValueError("Input DataFrame must contain a 'step' column.")
    n = len(DF)

    x_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^x_(\d+)$", column)]
        if match
    )
    c_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^c_(\d+)$", column)]
        if match
    )
    pv_idx = sorted(
        int(match.group(1))
        for column in DF.columns
        for match in [re.match(r"^p_(\d+)$", column)]
        if match
    )
    if not x_idx or not c_idx or not pv_idx:
        return pd.DataFrame(columns=[
            "step", "y", "x_index", "x_value", "w_ff",
            "c_index", "c_value", "w_fb", "pv_index", "pv_value",
            "w_lat", "w_pv_lat", "W_pv", "image_type", "condition", "experiment_phase", "experiment_series", "seed",
        ])

    nx = len(x_idx)
    nc = len(c_idx)
    npv = len(pv_idx)
    rep = nx * nc * npv

    x_vals = DF[[f"x_{idx}" for idx in x_idx]].to_numpy(dtype=float)
    c_vals = DF[[f"c_{idx}" for idx in c_idx]].to_numpy(dtype=float)
    wff_vals = DF[[f"w_ff_{idx}" for idx in x_idx]].to_numpy(dtype=float)
    wfb_vals = DF[[f"w_fb_{idx}" for idx in c_idx]].to_numpy(dtype=float)
    p_vals = DF[[f"p_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
    wlat_vals = DF[[f"w_lat_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
    wpvlat_vals = (
        DF[[f"w_pv_lat_{idx}" for idx in pv_idx]].to_numpy(dtype=float)
        if all(f"w_pv_lat_{idx}" in DF.columns for idx in pv_idx)
        else np.full((n, npv), np.nan, dtype=float)
    )

    wpv_vals = np.full((n, npv, nx), np.nan, dtype=float)
    for pv_position, pv_value in enumerate(pv_idx):
        for x_position, x_value in enumerate(x_idx):
            column_name = f"W_pv_{pv_value}_{x_value}"
            if column_name in DF.columns:
                wpv_vals[:, pv_position, x_position] = DF[column_name].to_numpy(dtype=float)

    step = np.repeat(DF["step"].to_numpy(dtype=float), rep)
    y = np.repeat(DF["y"].to_numpy(dtype=float), rep)

    x_index_grid, c_index_grid, pv_index_grid = np.meshgrid(
        np.array(x_idx, dtype=int),
        np.array(c_idx, dtype=int),
        np.array(pv_idx, dtype=int),
        indexing="ij",
    )
    combo_x_index = x_index_grid.reshape(-1)
    combo_c_index = c_index_grid.reshape(-1)
    combo_pv_index = pv_index_grid.reshape(-1)

    x_value = np.broadcast_to(x_vals[:, :, None, None], (n, nx, nc, npv)).reshape(-1)
    w_ff = np.broadcast_to(wff_vals[:, :, None, None], (n, nx, nc, npv)).reshape(-1)
    c_value = np.broadcast_to(c_vals[:, None, :, None], (n, nx, nc, npv)).reshape(-1)
    w_fb = np.broadcast_to(wfb_vals[:, None, :, None], (n, nx, nc, npv)).reshape(-1)
    pv_value = np.broadcast_to(p_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    w_lat = np.broadcast_to(wlat_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    w_pv_lat = np.broadcast_to(wpvlat_vals[:, None, None, :], (n, nx, nc, npv)).reshape(-1)
    W_pv = np.broadcast_to(np.transpose(wpv_vals, (0, 2, 1))[:, :, None, :], (n, nx, nc, npv)).reshape(-1)

    long_df = pd.DataFrame({
        "step": step,
        "y": y,
        "x_index": np.tile(combo_x_index, n),
        "x_value": x_value,
        "w_ff": w_ff,
        "c_index": np.tile(combo_c_index, n),
        "c_value": c_value,
        "w_fb": w_fb,
        "pv_index": np.tile(combo_pv_index, n),
        "pv_value": pv_value,
        "w_lat": w_lat,
        "w_pv_lat": w_pv_lat,
        "W_pv": W_pv,
    })

    if "seed" in DF.columns:
        long_df["seed"] = np.repeat(DF["seed"].to_numpy(), rep)
    if "experiment_series" in DF.columns:
        long_df["experiment_series"] = np.repeat(DF["experiment_series"].astype(str).to_numpy(), rep)

    if "condition" in DF.columns:
        cond = DF["condition"].astype(str).to_numpy()
        cond_rep = np.repeat(cond, rep)
        parts = pd.Series(cond_rep).str.rsplit("_", n=1, expand=True)
        if parts.shape[1] == 2:
            prefix = parts[0].astype(str)
            image_type_token = prefix.str.split("_", n=1).str[0]
            condition_token = prefix.str.split("_", n=1).str[1]
            converted = [
                _condition_token_to_image_type(image_type=image_type, condition_token=condition)
                for image_type, condition in zip(image_type_token, condition_token, strict=False)
            ]
            long_df["experiment_phase"] = parts[1].to_numpy()
            long_df["image_type"] = [image_type for image_type, _ in converted]
            long_df["condition"] = [condition for _, condition in converted]
        else:
            long_df["condition"] = cond_rep

    result_cols = [
        "step", "y", "x_index", "x_value", "w_ff",
        "c_index", "c_value", "w_fb", "pv_index", "pv_value",
        "w_lat", "w_pv_lat", "W_pv", "image_type", "condition", "experiment_phase", "experiment_series", "seed",
    ]
    return long_df[[column for column in result_cols if column in long_df.columns]]
