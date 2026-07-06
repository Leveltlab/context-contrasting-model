from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import transitions_helpers as th


DIRECTION_SECTORS = {
    "+NO": "+NO axis",
    "+O": "+O axis",
    "-NO": "-NO axis",
}
DIAGONAL_ANGLES = {
    frozenset(("+NO", "+O")): np.pi / 4.0,
    frozenset(("-NO", "+O")): 3.0 * np.pi / 4.0,
}
MAGNITUDE_BANDS = {
    "s": "small",
    "m": "medium",
    "h": "high",
}
NEAR_DIAGONAL_MAX_DISTANCE = np.pi / 8.0


def _split_magnitude_suffix(token: str) -> tuple[str, str | None]:
    if len(token) > 1 and token[-1].lower() in MAGNITUDE_BANDS:
        return token[:-1], MAGNITUDE_BANDS[token[-1].lower()]
    return token, None


def _axis_token_to_sector(token: str) -> str:
    if token == "-O":
        raise ValueError("-O highlighted examples are not supported.")
    try:
        return DIRECTION_SECTORS[token]
    except KeyError as exc:
        raise ValueError(f"unsupported highlight direction {token!r}; use +NO, +O, or -NO.") from exc


def _parse_diagonal_token(token: str, magnitude_band: str | None) -> dict[str, Any] | None:
    if "/" not in token:
        return None
    first_raw, second_raw = token.upper().split("/", maxsplit=1)
    if not first_raw or not second_raw:
        raise ValueError(f"diagonal highlight token {token!r} must look like +NO/+O.")
    if first_raw == "-O" or second_raw == "-O":
        raise ValueError("-O diagonal highlighted examples are not supported.")
    diagonal_key = frozenset((first_raw, second_raw))
    if diagonal_key not in DIAGONAL_ANGLES:
        raise ValueError(
            f"diagonal highlight token {token!r} is not supported; only +NO/+O and -NO/+O diagonal pairs are supported."
        )
    return {
        "number": None,
        "requested_sector": _axis_token_to_sector(first_raw),
        "diagonal_sectors": [_axis_token_to_sector(first_raw), _axis_token_to_sector(second_raw)],
        "diagonal_angle": DIAGONAL_ANGLES[diagonal_key],
        "diagonal_token": token,
        "magnitude_band": magnitude_band,
    }


def parse_highlight_tokens(tokens: list[Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    aliases = {
        "__DIRECTION_NEG_NO__": "-NO",
        "__DIRECTION_NEG_O__": "-O",
    }
    for raw_token in tokens:
        raw_text = str(raw_token)
        token = next(
            (replacement + raw_text.removeprefix(alias) for alias, replacement in aliases.items() if raw_text.startswith(alias)),
            raw_text,
        )
        token, magnitude_band = _split_magnitude_suffix(token)
        diagonal_spec = _parse_diagonal_token(token, magnitude_band)
        if diagonal_spec is not None:
            specs.append(diagonal_spec)
            continue

        upper_token = token.upper()
        direction = None
        suffix = ""
        for candidate in ("+NO", "+O", "-NO", "-O"):
            if upper_token.startswith(candidate):
                direction = candidate
                suffix = token[len(candidate):]
                break
        if direction == "-O":
            raise ValueError("-O highlighted examples are not supported.")
        if direction in DIRECTION_SECTORS and suffix == "":
            specs.append(
                {
                    "number": None,
                    "requested_sector": DIRECTION_SECTORS[direction],
                    "diagonal_angle": None,
                    "diagonal_token": None,
                    "magnitude_band": magnitude_band,
                }
            )
            continue
        if direction in DIRECTION_SECTORS:
            try:
                number = int(suffix)
            except ValueError as exc:
                raise ValueError(f"highlight example token {token!r} has an invalid template suffix.") from exc
            specs.append(
                {
                    "number": number,
                    "requested_sector": DIRECTION_SECTORS[direction],
                    "diagonal_angle": None,
                    "diagonal_token": None,
                    "magnitude_band": magnitude_band,
                }
            )
            continue
        try:
            number = int(token)
        except ValueError as exc:
            raise ValueError(
                f"highlight example token {token!r} must be a template number, a direction (+NO, +O, -NO), "
                "a combined token such as +NO13, or a diagonal token such as +O/+NOh."
            ) from exc
        specs.append(
            {
                "number": number,
                "requested_sector": None,
                "diagonal_angle": None,
                "diagonal_token": None,
                "magnitude_band": magnitude_band,
            }
        )
    return specs


def parse_highlight_numbers(args: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        "familiar": parse_highlight_tokens(list(getattr(args, "fam_examples", []) or [])),
        "novel": parse_highlight_tokens(list(getattr(args, "nov_examples", []) or [])),
    }


def validate_highlight_numbers(highlights: dict[str, list[dict[str, Any]]], transition_order: list[str]) -> None:
    valid = set(range(1, len(transition_order) + 1))
    requested = [int(spec["number"]) for specs in highlights.values() for spec in specs if spec["number"] is not None]
    invalid = sorted(set(requested) - valid)
    if invalid:
        raise ValueError(f"highlight template numbers must be between 1 and {len(transition_order)}; invalid={invalid}")


def highlight_requested(highlights: dict[str, list[dict[str, Any]]]) -> bool:
    return any(bool(specs) for specs in highlights.values())


def summary_with_transition(summary: pd.DataFrame, samples: list[dict[str, Any]]) -> pd.DataFrame:
    transition_by_neuron = {
        int(sample["_sample_global_idx"]): sample["_canonical_transition"]
        for sample in samples
    }
    order_by_neuron = {
        int(sample["_sample_global_idx"]): int(sample.get("_highlight_candidate_order", idx))
        for idx, sample in enumerate(samples, start=1)
    }
    enriched = summary.copy()
    enriched["transition"] = enriched["neuron_idx"].map(transition_by_neuron)
    enriched["sample_order"] = enriched["neuron_idx"].map(order_by_neuron)
    return enriched


def _directional_highlight_requirements_met(
    transition_table: pd.DataFrame,
    *,
    samples: list[dict[str, Any]],
    highlights: dict[str, list[dict[str, Any]]],
    template_numbers: dict[int, str],
    threshold: float,
    wide_table: Callable[[pd.DataFrame], pd.DataFrame],
) -> bool:
    directional_specs = [
        (group, spec)
        for group, specs in highlights.items()
        for spec in specs
        if spec.get("requested_sector") is not None and spec.get("diagonal_angle") is None
    ]
    if not directional_specs:
        return True
    if transition_table.empty:
        return False

    wide = wide_table(transition_table)
    summaries = {
        group: summary_with_transition(
            th.build_mean_summary(wide, image_group=group, pre_stage="Naive", target_stage="Expert", threshold=threshold),
            samples,
        )
        for group, _ in directional_specs
    }
    for group, spec in directional_specs:
        transition_mask = True
        if spec["number"] is not None:
            transition_mask = summaries[group]["transition"].eq(template_numbers[int(spec["number"])])
        rows = summaries[group].loc[
            transition_mask
            & summaries[group]["RotatedSector"].astype(str).eq(str(spec["requested_sector"]))
            & (summaries[group]["dNorm"].astype(float) > threshold)
        ]
        if rows.empty:
            return False
    return True


def sample_uniform_highlight_pool(
    args: Any,
    *,
    transition_order: list[str],
    highlights: dict[str, list[dict[str, Any]]],
    template_numbers: dict[int, str],
    minimal_configs: dict[str, dict[str, Any]],
    perturb_config: Callable[..., dict[str, Any]],
    run_sample: Callable[..., pd.DataFrame],
    transition_table: Callable[[pd.DataFrame], pd.DataFrame],
    wide_table: Callable[[pd.DataFrame], pd.DataFrame],
    test_stimuli: dict[str, Any],
    training_stimuli: Any,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    requested_transitions = {
        template_numbers[int(spec["number"])]
        for specs in highlights.values()
        for spec in specs
        if spec["number"] is not None and spec.get("diagonal_angle") is None
    }
    has_direction_only = any(
        spec["number"] is None and spec.get("requested_sector") is not None and spec.get("diagonal_angle") is None
        for specs in highlights.values()
        for spec in specs
    )
    if not requested_transitions and not has_direction_only:
        return [], pd.DataFrame()

    max_samples = max_samples or max(args.n_samples, 100 * len(transition_order))
    samples: list[dict[str, Any]] = []
    response_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 500_000]))
    child_seed_sequence = np.random.SeedSequence([args.seed, 600_000])
    child_seeds = iter(child_seed_sequence.spawn(max_samples))
    seen = dict.fromkeys(transition_order, 0)

    def build_sample(global_order: int, transition: str, child_seed: np.random.SeedSequence) -> dict[str, Any]:
        seen[transition] += 1
        global_idx = -(500_000 + global_order)
        sample = perturb_config(
            transition,
            minimal_configs[transition],
            sample_idx=seen[transition],
            global_idx=global_idx,
            seed=args.seed + 500_000 + global_order,
            rng=np.random.default_rng(child_seed),
            scalar_noise_multiplier=args.scalar_noise_multiplier,
        )
        sample["_highlight_candidate_order"] = int(global_order)
        return sample

    while len(samples) < max_samples:
        remaining = max_samples - len(samples)
        batch_size = min(max(4 * len(transition_order), 64), remaining)
        transitions = rng.choice(transition_order, size=batch_size, replace=True).tolist()
        batch_samples = [
            build_sample(len(samples) + offset, transition, next(child_seeds))
            for offset, transition in enumerate(transitions, start=1)
        ]
        batch_frames = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(run_sample)(
                sample,
                n_steps_per_phase=args.n_steps_per_phase,
                response_tail_fraction=args.response_tail_fraction,
                test_stimuli=test_stimuli,
                training_stimuli=training_stimuli,
                zscore_std_floor=args.zscore_std_floor,
            )
            for sample in batch_samples
        )
        samples.extend(batch_samples)
        response_frames.extend(batch_frames)
        if requested_transitions.issubset({sample["_canonical_transition"] for sample in samples}):
            table = transition_table(pd.concat(response_frames, ignore_index=True))
            if _directional_highlight_requirements_met(
                table,
                samples=samples,
                highlights=highlights,
                template_numbers=template_numbers,
                threshold=args.threshold,
                wide_table=wide_table,
            ):
                return samples, table

    if response_frames:
        return samples, transition_table(pd.concat(response_frames, ignore_index=True))
    raise ValueError("Could not sample any highlighted-example candidates.")


def _angle_distance(values: pd.Series | np.ndarray, target: float) -> np.ndarray:
    angles = np.asarray(values, dtype=float)
    return np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))


def _magnitude_band_rows(rows: pd.DataFrame, band: str | None) -> pd.DataFrame:
    if band is None or rows.empty:
        return rows
    sorted_indices = rows.sort_values(["dNorm_float", "sample_order"]).index.to_numpy()
    chunks = np.array_split(sorted_indices, 3)
    selected = {
        "small": chunks[0],
        "medium": chunks[1],
        "high": chunks[2],
    }[band]
    if len(selected) == 0:
        return rows
    band_rows = rows.loc[selected]
    return band_rows if not band_rows.empty else rows


def select_highlight_examples(
    summaries: dict[str, pd.DataFrame],
    *,
    samples: list[dict[str, Any]],
    aggregate_summaries: dict[str, pd.DataFrame],
    aggregate_samples: list[dict[str, Any]],
    highlights: dict[str, list[dict[str, Any]]],
    template_numbers: dict[int, str],
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    sample_by_neuron = {int(sample["_sample_global_idx"]): sample for sample in samples}
    aggregate_sample_by_neuron = {int(sample["_sample_global_idx"]): sample for sample in aggregate_samples}
    selected: dict[str, list[dict[str, Any]]] = {"familiar": [], "novel": []}

    for group, specs in highlights.items():
        if not specs:
            continue
        for display_idx, spec in enumerate(specs, start=1):
            number = int(spec["number"]) if spec["number"] is not None else None
            requested_sector = spec.get("requested_sector")
            diagonal_angle = spec.get("diagonal_angle")
            diagonal_token = spec.get("diagonal_token")
            magnitude_band = spec.get("magnitude_band")
            use_aggregate_source = diagonal_angle is not None
            summary = aggregate_summaries[group] if use_aggregate_source else summaries[group]
            active_sample_by_neuron = aggregate_sample_by_neuron if use_aggregate_source else sample_by_neuron
            transition = template_numbers[number] if number is not None else None
            rows = summary.copy() if number is None else summary.loc[summary["transition"] == transition].copy()
            if rows.empty:
                target = f"template {number} ({transition})" if number is not None else f"direction {requested_sector!r}"
                raise ValueError(
                    f"No sampled cell for {target} is present in the {group} highlight pool. "
                    "Increase the highlighted-example sampling budget or choose a different example template."
                )
            rows["dNorm_float"] = rows["dNorm"].astype(float)
            rows["distance_to_threshold"] = np.abs(rows["dNorm_float"] - threshold)

            if diagonal_angle is not None:
                rows["diagonal_distance"] = _angle_distance(rows["Angle"], float(diagonal_angle))
                diagonal_sectors = [str(sector) for sector in spec.get("diagonal_sectors", [])]
                if not diagonal_sectors:
                    diagonal_sectors = [str(requested_sector)]
                diagonal_pool_all = rows.loc[
                    rows["RotatedSector"].astype(str).isin(diagonal_sectors)
                    & (rows["dNorm_float"] > threshold)
                ]
                diagonal_pool = diagonal_pool_all.loc[
                    diagonal_pool_all["diagonal_distance"] <= NEAR_DIAGONAL_MAX_DISTANCE
                ]
                if diagonal_pool.empty:
                    diagonal_pool = diagonal_pool_all.sort_values(["diagonal_distance", "sample_order"]).head(3)
                diagonal_pool = _magnitude_band_rows(diagonal_pool, magnitude_band)
                diagonal_rows = diagonal_pool.loc[
                    diagonal_pool["RotatedSector"].astype(str).eq(str(requested_sector))
                ].sort_values(["diagonal_distance", "sample_order"])
                if not diagonal_rows.empty:
                    row = diagonal_rows.iloc[0]
                    selection_rule = "nearest_diagonal_above_threshold"
                else:
                    unbanded_diagonal_rows = diagonal_pool_all.loc[
                        diagonal_pool_all["RotatedSector"].astype(str).eq(str(requested_sector))
                        & (diagonal_pool_all["diagonal_distance"] <= NEAR_DIAGONAL_MAX_DISTANCE)
                    ].sort_values(["diagonal_distance", "sample_order"])
                    if unbanded_diagonal_rows.empty:
                        unbanded_diagonal_rows = diagonal_pool_all.loc[
                            diagonal_pool_all["RotatedSector"].astype(str).eq(str(requested_sector))
                        ].sort_values(["diagonal_distance", "sample_order"])
                    if not unbanded_diagonal_rows.empty:
                        row = unbanded_diagonal_rows.iloc[0]
                        selection_rule = "nearest_diagonal_above_threshold_magnitude_fallback"
                    else:
                        fallback_rows = rows.loc[rows["RotatedSector"].astype(str).eq("small ∆")]
                        fallback_rows = _magnitude_band_rows(fallback_rows, magnitude_band).sort_values(["diagonal_distance", "sample_order"])
                        if fallback_rows.empty:
                            raise ValueError(
                                f"No sampled cell in the {group} aggregate sample could satisfy or fallback for "
                                f"diagonal request {diagonal_token!r}."
                            )
                        row = fallback_rows.iloc[0]
                        selection_rule = "nearest_diagonal_small_delta_fallback"
            elif requested_sector is not None:
                direction_rows = rows.loc[
                    rows["RotatedSector"].astype(str).eq(str(requested_sector))
                    & (rows["dNorm_float"] > threshold)
                ]
                direction_rows = _magnitude_band_rows(direction_rows, magnitude_band).sort_values("sample_order")
                if direction_rows.empty:
                    target = f"template {number} ({transition})" if number is not None else "any template"
                    raise ValueError(
                        f"No sampled cell from {target} in the {group} highlight pool "
                        f"matched requested direction {requested_sector!r} above threshold={threshold:g}."
                    )
                row = direction_rows.iloc[0]
                selection_rule = f"first_{requested_sector}_above_threshold"
            else:
                above_threshold = rows.loc[rows["dNorm_float"] > threshold]
                if not above_threshold.empty:
                    candidate_rows = _magnitude_band_rows(above_threshold, magnitude_band).sort_values("sample_order")
                    row = candidate_rows.iloc[0]
                    selection_rule = "first_above_threshold"
                else:
                    candidate_rows = _magnitude_band_rows(rows, magnitude_band).sort_values(["distance_to_threshold", "sample_order"])
                    row = candidate_rows.iloc[0]
                    selection_rule = "closest_to_threshold"

            sector = str(row["RotatedSector"])
            selected[group].append(
                {
                    "number": int(number) if number is not None else None,
                    "display_number": int(display_idx),
                    "example_key": (
                        f"example_{group}_{display_idx:02d}_{int(number):02d}"
                        if number is not None
                        else f"example_{group}_{display_idx:02d}_{str(diagonal_token or requested_sector).replace('/', '_').replace(' ', '_').replace('+', 'pos').replace('-', 'neg')}"
                    ),
                    "selection_rule": selection_rule,
                    "requested_sector": requested_sector,
                    "requested_diagonal": diagonal_token,
                    "magnitude_band": magnitude_band,
                    "transition": transition if transition is not None else str(row["transition"]),
                    "neuron_idx": int(row["neuron_idx"]),
                    "sector": sector,
                    "color": th.ROTATED_SECTOR_PALETTE.get(sector, "0.35"),
                    "NO_Pre": float(row["NO_Pre"]),
                    "O_Pre": float(row["O_Pre"]),
                    "NO_Target": float(row["NO_Target"]),
                    "O_Target": float(row["O_Target"]),
                    "dNO": float(row["dNO"]),
                    "dO": float(row["dO"]),
                    "diagonal_distance": float(row["diagonal_distance"]) if "diagonal_distance" in row.index else np.nan,
                    "sample": active_sample_by_neuron[int(row["neuron_idx"])],
                }
            )
    return selected
