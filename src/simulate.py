"""Run the six YAML models and save traces and trial responses. Run from the repository root."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed

from .model import CCNeuron

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("naive", "expert", "expert_no_fb", "expert_no_lat", "expert_no_fb_no_lat")


def load_config(directory=ROOT / "configs"):
    protocol = yaml.safe_load((directory / "protocol.yaml").read_text())
    cells = [yaml.safe_load(path.read_text()) for path in sorted(directory.glob("*_*.yaml"))]
    cells.sort(key=lambda cell: cell["id"])
    return cells, protocol


def stimuli(images, protocol, post=True):
    """Three seconds without input, one second of input, repeated per image."""
    n = protocol["steps_per_trial"]
    x = torch.zeros(len(images) * n + (n // 2 if post else 0), 3)
    c = torch.zeros_like(x)
    breadth = protocol["signal_other_value"]
    for trial, image in enumerate(images):
        start, end = trial * n + 3 * n // 4, (trial + 1) * n
        x[start:end, image] = 1
        c[start:end] = breadth
        c[start:end, image] = 1
    c /= 1 + 2 * breadth
    return x, c


def run_phase(model, x, c, learn=False, silence_pv=False):
    model.reset()
    return np.array([model.step(xt, ct, learn, silence_pv) for xt, ct in zip(x, c)])


def train(model, protocol):
    images = [0, 1] * protocol["training_trials"]
    # images = np.random.default_rng(protocol["training_order_seed"]).permutation(images)
    x, c = stimuli(images, protocol, post=False)
    run_phase(model, x, c, learn=True)


def simulate(cell, protocol, include_trial_responses=False):
    torch.set_num_threads(1)
    model = CCNeuron(cell["parameters"])
    initial_rng = torch.random.get_rng_state()
    n, trials = protocol["steps_per_trial"], protocol["test_trials"]
    probes = [stimuli([image] * trials, protocol) for image in range(3)]
    traces = {}
    for phase in ("naive", "expert"):
        if phase == "expert":
            # Replay the short original probe so changing measurement averaging
            # cannot change the noise seen during learning.
            torch.random.set_rng_state(initial_rng)
            for image in range(3):
                x, c = stimuli([image] * protocol["rng_reference_trials"], protocol)
                run_phase(model, x, c)
                run_phase(model, torch.zeros_like(x), c)
            train(model, protocol)
        for image, (x, c) in enumerate(probes):
            conditions = CONDITIONS[1:] if phase == "expert" else ("naive",)
            for condition in conditions:
                context = torch.zeros_like(c) if "no_fb" in condition else c
                for response in ("NO", "O"):
                    bottom_up = x if response == "NO" else torch.zeros_like(x)
                    values = run_phase(model, bottom_up, context, silence_pv="no_lat" in condition)
                    # Average aligned trial windows: -1 to +3 seconds.
                    traces[condition, image, response] = np.stack([
                        values[t * n + n // 2:t * n + 3 * n // 2] for t in range(trials)
                    ])

    baseline = np.concatenate([
        values[:, :n // 4].ravel() for (condition, _, _), values in traces.items()
        if condition == "naive"
    ])
    mean, std = baseline.mean(), baseline.std(ddof=1)
    native_images = (0, 1) if cell["source"] == "familiar" else (2,)
    common = dict(source=cell["source"], sector=cell["sector"], observation_id=cell["id"],
                  seed=cell["parameters"]["seed"])
    rows = []
    trial_rows = []
    responses = {}
    for (condition, image, response), values in traces.items():
        if image not in native_images:
            continue
        normalized = (values - mean) / std
        y = normalized.mean(axis=0)
        time = (np.arange(n) - n // 4) / protocol["steps_per_second"]
        # The saved final vectors include the first sample at stimulus offset.
        responses[condition, image, response] = y[n // 4:n // 2 + 1].mean()
        if include_trial_responses:
            for trial, value in enumerate(normalized[:, n // 4:n // 2 + 1].mean(axis=1), 1):
                trial_rows.append(dict(**common, condition_key=condition, image_id=image + 1,
                                       test_trial=trial, response_type=response, response=value))
        rows.append(pd.DataFrame(dict(
            **common, record_type="trace", condition_key=condition,
            image_id=image + 1, response_type=response, time_seconds=time, response=y,
            n_trials=trials, naive_baseline_mean=mean, naive_baseline_std=std,
        )))
    vector = dict(**common, record_type="transition", condition_key="transition")
    for response in ("NO", "O"):
        for phase, label in (("naive", "naive"), ("expert", "target")):
            vector[f"{label}_{response}"] = np.mean([responses[phase, i, response] for i in native_images])
        vector[f"delta_{response}"] = vector[f"target_{response}"] - vector[f"naive_{response}"]
    rows.append(pd.DataFrame([vector]))
    print(f"{cell['source']} {cell['sector']} seed={common['seed']}: "
          f"delta NO={vector['delta_NO']:.4f}, O={vector['delta_O']:.4f}", flush=True)
    traces_and_vectors = pd.concat(rows, ignore_index=True)
    if not include_trial_responses:
        return traces_and_vectors
    trial_responses = pd.DataFrame(trial_rows).pivot(
        index=["source", "sector", "observation_id", "seed", "condition_key", "image_id", "test_trial"],
        columns="response_type", values="response").reset_index()
    trial_responses.columns.name = None
    return traces_and_vectors, trial_responses.rename(columns={"NO": "response_NO", "O": "response_O"})


def seeded_cell(cell, seed):
    cell = copy.deepcopy(cell)
    cell["parameters"]["seed"] = seed
    return cell


def simulate_ensemble(cells, protocol, seeds=tuple(range(10)), jobs=-1,
                      include_trial_responses=False):
    # Separate processes isolate PyTorch's global RNG; each task uses one thread.
    frames = Parallel(n_jobs=jobs, backend="loky")(
        delayed(simulate)(seeded_cell(cell, seed), protocol, include_trial_responses)
        for seed in seeds for cell in cells)
    if include_trial_responses:
        return tuple(pd.concat(parts, ignore_index=True) for parts in zip(*frames))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model.csv")
    parser.add_argument("--trial-output", type=Path,
                        help="Trial-level response CSV (default: model_trial_responses.csv beside --output)")
    parser.add_argument("--test-trials", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Replicate seeds shared across configs (default: 0 through 9); overrides YAML seeds")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Independent worker processes (-1: all available CPU cores)")
    args = parser.parse_args()
    if args.test_trials < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("test-trials must be positive and seeds must be unique")
    if args.jobs == 0:
        parser.error("jobs cannot be zero")
    cells, protocol = load_config(args.configs)
    if args.test_trials is not None:
        protocol["test_trials"] = args.test_trials
    frame, trial_responses = simulate_ensemble(cells, protocol, args.seeds, args.jobs,
                                               include_trial_responses=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    trial_output = args.trial_output or args.output.with_name(args.output.stem + "_trial_responses.csv")
    trial_output.parent.mkdir(parents=True, exist_ok=True)
    trial_responses.to_csv(trial_output, index=False)


if __name__ == "__main__":
    main()
