from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from pandas import DataFrame


PRIMARY_EXPERIMENT_SERIES = "training_familiar"

STIMULUS_SPECS = {
    "familiar_1": ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    "familiar_2": ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    "novel": ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
}

NO_RESPONSE_ABLATION_SPECS = {
    "no_context": {"condition_prefix": "nocontext", "zero_context": True},
    "nolat": {"condition_prefix": "nolat", "model_overrides": {"use_lat_connection": False}},
    "no_context_nolat": {
        "condition_prefix": "nocontextnolat",
        "zero_context": True,
        "model_overrides": {"use_lat_connection": False},
    },
}


def _collect_outputs(step: int, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, c: torch.Tensor, model: Any, rows: list[dict]) -> None:
    row = {"step": step, "y": y.item()}
    for i, value in enumerate(x.detach().cpu().numpy().copy()):
        row[f"x_{i}"] = value
        row[f"w_ff_{i}"] = model.w_ff.detach().cpu().numpy().copy()[i]
    for i, value in enumerate(c.detach().cpu().numpy().copy()):
        row[f"c_{i}"] = value
        row[f"w_fb_{i}"] = model.w_fb.detach().cpu().numpy().copy()[i]
    p_values = p.detach().cpu().numpy().copy()
    w_lat = model.w_lat.detach().cpu().numpy().copy()
    w_pv_lat = model.w_pv_lat.detach().cpu().numpy().copy()
    W_pv = model.W_pv.detach().cpu().numpy().copy()
    for i, value in enumerate(p_values):
        row[f"p_{i}"] = value
        row[f"w_lat_{i}"] = w_lat[i]
        row[f"w_pv_lat_{i}"] = w_pv_lat[i]
        for j in range(model.n_features):
            row[f"W_pv_{i}_{j}"] = W_pv[i, j]
    rows.append(row)


@contextmanager
def _temporary_model_overrides(model: Any, **overrides: bool):
    originals = {name: getattr(model, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(model, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(model, name, value)


def run_experimental_phase(
    model: Any,
    X: torch.Tensor,
    C: torch.Tensor,
    condition_name: str = "default",
    update: bool = False,
    reset_rates: bool = True,
) -> DataFrame:
    if reset_rates:
        model._reset_state()

    rows: list[dict] = []
    for step in range(X.shape[0]):
        x, y_t, y_next, p, c = model(X[step], C[step])
        if update:
            model.update(x, y_t, y_next, p, c)
        _collect_outputs(step, x, y_next, p, c, model, rows)

    df = DataFrame(rows)
    df["condition"] = condition_name
    return df


def _run_test_phase_variants(
    model: Any,
    stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase_label: str,
) -> list[DataFrame]:
    frames: list[DataFrame] = []

    for condition_name, (X, C) in stimuli.items():
        occluded_X = torch.zeros_like(X)
        no_context_C = torch.zeros_like(C)

        frames.append(run_experimental_phase(model, X, C, condition_name=f"full_{condition_name}_{phase_label}", update=False))
        frames.append(run_experimental_phase(model, occluded_X, C, condition_name=f"occlusion_{condition_name}_{phase_label}", update=False))

        for ablation_label, ablation_spec in NO_RESPONSE_ABLATION_SPECS.items():
            ablated_C = no_context_C if ablation_spec.get("zero_context", False) else C
            model_overrides = ablation_spec.get("model_overrides", {})
            condition_prefix = ablation_spec.get("condition_prefix", ablation_label)
            with _temporary_model_overrides(model, **model_overrides):
                frames.append(run_experimental_phase(model, X, ablated_C, condition_name=f"{condition_prefix}_{condition_name}_{phase_label}", update=False))
                frames.append(
                    run_experimental_phase(
                        model,
                        occluded_X,
                        ablated_C,
                        condition_name=f"occlusion_{condition_name}_{condition_prefix}_{phase_label}",
                        update=False,
                    )
                )

    return frames
