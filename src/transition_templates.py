"""Pruned, principled mini model-scatter variant.

Keeps the mini model assumptions -- only the PyC synapses ``w_ff``, ``w_fb`` and
``w_lat`` learn (PV tuning ``W_pv`` / ``w_pv_lat`` is fixed), and per-cell noise is
injected only into the FF/FB initial weights, ``apical_drive_threshold`` and
``apical_gain_strength`` -- but replaces hand-assigned transition identities with
a small set of templates keyed by **tuning width** and PyC/PV feedforward strength:

  * ``silent``: near-silent FF drive. With no feedback it stays near the origin
    (small dNO/dO); with strong generalized feedback it is a feedback-driven
    occluded ("O") responder.
  * ``narrow`` (width 1): weak FF (anti-Hebbian) plasticity + a high apical drive
    threshold, so feedback can only *gain-modulate*. The non-adapted FF drive is
    amplified into a ``+NO`` response. Each narrow cell's single preferred
    stimulus is drawn at random per cell, so the familiar-vs-novel ``+NO``
    asymmetry **emerges** from the protocol (only the familiar images are shown
    during the plastic phase, so their FF adapts while the novel image's does not)
    rather than being hand-assigned to separate familiar/novel templates.
  * ``broad`` (width 2): strong FF plasticity + a low apical drive threshold, so
    familiar FF adapts away (``-NO``) and the strengthened, generalized feedback
    drives the occluded response (``+O``). Each sampled broad cell is tuned to a
    random pair of the three images rather than all inputs.

Two principled choices distinguish this from the legacy templates: feedback is
always **generalized over the context channels a cell receives** (no hand-painted
per-channel weight asymmetry), and narrow/broad tuning is the *only* thing that
sets the FF-plasticity scale and the feedback drive-vs-gain regime. Surround
weights (``w_lat``, ``W_pv``, ``w_pv_lat``) are coupled per template; only
``w_lat`` is plastic. PV feedforward tuning is sampled independently of PyC
tuning; every sampled PV is tuned to two images and less tuned to the third.

The width-class scalars and the mixture weights at the top of this file are the
tuning levers; everything downstream reuses ``run_model_scatter.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import model_scatter as base
from .neuron_utils import ThresholdReLU


DEFAULT_OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR
N_FEATURES = 3

# Only PyC synapses learn; PV feedforward learning is off (lr_pv = 0). Feedback
# strengthening and lateral surround learning stay shared across templates. These
# rates are calibrated at 200 steps/phase; longer phase durations scale them down
# so the accumulated plasticity stays close to the clearer 200-step scatter.
LEARNING_RATE_REFERENCE_STEPS = 200
SHARED_LEARNING_RATES = {"lr_ff": 0.0155, "lr_fb": 0.00065, "lr_lat": 0.0300, "lr_pv": 0.0}
SCALAR_NOISE_KEYS = ("apical_gain_strength", "apical_drive_threshold")
SOMA_ACTIVATION_THRESHOLD = 0.08
APICAL_DRIVE_SUBTRACTIVE = True
PV_NOISE_SIGMA = 0.075
DIVISIVE_GAIN = 10.0
BASELINE_STD_SCALE = 0.27
# Shared apical gain-sigmoid threshold for every neuron. The clipped GainSigmoid
# (see neuron_utils.GainSigmoid) returns max(1, ...) regardless of (gain, k,
# threshold), so a positive threshold here only DELAYS the onset of
# amplification (gain factor sits at 1.0 below threshold and rises >1 above)
# and never suppresses the soma below the FF-only baseline -- the apical
# compartment stays strictly amplify-only.
SHARED_GAIN_THRESHOLD = 0.05
SCALAR_NOISE = {
    "apical_gain_strength": ("log", 0.18, 0.1, 50.0, 0.0),
    "apical_drive_threshold": ("add", 0.12, 0.0, 3.0, 0.05),
}
# `hi` bounds must sit well above the largest template-centre value so the noise
# distribution isn't truncated. FF: strong tuned=0.190 with rel=0.35 noise -> 99th
# percentile ~ 0.32, so hi=0.40. FB: very_strong=0.300 with rel=0.50 noise -> 99th
# percentile ~ 0.65, so hi=0.80.
UNIFORM_FF_NOISE = dict(rel=0.28, floor=0.016, lo=0.0, hi=0.35)
UNIFORM_FB_NOISE = dict(rel=0.40, floor=0.016, lo=0.0, hi=0.65)
UNIFORM_GAIN_CLIP = (1.5, 6.4)
UNIFORM_DRIVE_CLIP = (0.0, 1.5)

BASE_CONFIG = {
    "n_features": 3,
    "n_pv": 1,
    "n_context": 3,
    "activation": ThresholdReLU(threshold=0.1, subtractive=False, hasMax=True, maxValue=1.0),
    "lr_ff": 0.032,
    "lr_fb": 0.0035,
    "lr_lat": 0.002,
    "lr_pv": 0.005,
    "w_ff_init": {"mu": [0.5, 0.5, 0.01], "sigma": 0},
    "w_fb_init": {"mu": [0.03, 0.03, 0.03], "sigma": 0},
    "w_lat_init": {"mu": [0.1], "sigma": 0},
    "W_pv_init": {"mu": [0.1, 0.1, 0.05], "sigma": 0},
    "pyc_decay": 0.05,
    "pv_decay": 0.5,
    "apical_drive_threshold": 0.3,
    "apical_drive_subtractive": False,
    "apical_gain_strength": 8.0,
    "apical_gain_k": 5.0,
    "apical_gain_threshold": 0.0,
    "baseline_drive_sigma": 0.08,
    "pv_noise_sigma": PV_NOISE_SIGMA,
    "alpha": 1.0,
    "weight_decay": 0.0,
    "seed": 42,
    "receives_context": (True, True, True),
    "FBrule": "dampened-anti-Hebbian",
    "w_pv_lat_init": {"mu": [0.4], "sigma": 0},
}


def _effective_learning_rates(n_steps_per_phase: int) -> dict[str, float]:
    if n_steps_per_phase <= 0:
        raise ValueError("n_steps_per_phase must be positive.")
    scale = LEARNING_RATE_REFERENCE_STEPS / float(n_steps_per_phase)
    return {
        name: (float(rate) * scale if name != "lr_pv" else float(rate))
        for name, rate in SHARED_LEARNING_RATES.items()
    }

# Two tuning-width classes only. BROAD: strong FF (anti-Hebbian) plasticity + a low
# apical drive threshold, so feedback DRIVES the soma. NARROW: weak FF plasticity +
# a high drive threshold, so feedback only GAIN-modulates. "Untuned" cells are
# modeled as weak / already-adapted BROAD cells (a broad cell whose FF was depressed
# by past adaptation), so they share the broad width class and merely carry a
# near-silent feedforward weight -- there is no separate untuned regime. Each entry
# gives fixed centers and post-noise clips for the two jittered scalar parameters
# (apical_gain_strength, apical_drive_threshold). Baseline sigma is a fixed center.
WIDTH_CLASSES: dict[str, dict[str, Any]] = {
    "broad": dict(
        ff_plasticity_scale=8.0,
        # Broad cells need enough gain that intact novel FF drive can be amplified
        # alongside strengthened generalized FB; otherwise novel expert NO gets
        # suppressed even for mixed FF+FB responders.
        gain=3.8, gain_clip=UNIFORM_GAIN_CLIP,
        drive=0.16, drive_clip=UNIFORM_DRIVE_CLIP,
        baseline=0.16,
    ),
    "narrow": dict(
        # Weak (not zero) FF plasticity: a narrow cell tuned to a *familiar* image
        # partially adapts its FF during training. With GainSigmoid now clipped at
        # >=1, naive narrows are no longer FB-suppressed (gain factor = 1 at
        # y_fb < threshold); the familiar +NO mover therefore requires the
        # adapted-FF * expert-gain product to remain ABOVE the un-amplified naive
        # NO. So the scale must be MILD enough that FF doesn't drain past
        # ~gain_expert^-1 of its initial value -- otherwise familiar narrows
        # collapse into small/-NO en masse. Novel-tuned narrows (FF intact) still
        # ride the smaller expert-gain growth into +NO.
        ff_plasticity_scale=2.0,
        gain=6.4, gain_clip=UNIFORM_GAIN_CLIP,
        # Sharper sigmoid (gain_k=9 vs the broad default 5) so the small naive->
        # expert w_fb growth produces a visible amplification step around the
        # shared SHARED_GAIN_THRESHOLD; below the threshold the clipped
        # GainSigmoid pins the factor at 1.0 (no suppression).
        gain_k=9.0,
        drive=1.08, drive_clip=UNIFORM_DRIVE_CLIP,
        # Low baseline -> low adaptation current, so the occluded basal stays near 0
        # and the rising gain barely drags the occluded response negative: the +NO
        # cells stay centred at O~0 (their NO shift is FF-driven, baseline-independent).
        baseline=0.10,
    ),
}

# Shared feedforward initial-weight levels. These are independent of tuning width:
# the tuned value is assigned to preferred stimulus channels and the silent value
# to non-preferred channels. Broad PyCs tune two randomly sampled image channels,
# while narrow PyCs tune one channel.
FF_STRENGTHS: dict[str, dict[str, Any]] = {
    "silent": dict(tuned=0.010, silent=0.010, **UNIFORM_FF_NOISE),
    "super_weak": dict(tuned=0.065, silent=0.004, **UNIFORM_FF_NOISE),
    "very_weak": dict(tuned=0.080, silent=0.005, **UNIFORM_FF_NOISE),
    "weak": dict(tuned=0.105, silent=0.006, **UNIFORM_FF_NOISE),
    "mid": dict(tuned=0.145, silent=0.006, **UNIFORM_FF_NOISE),
    "strong": dict(tuned=0.190, silent=0.008, **UNIFORM_FF_NOISE),
}

# PV feedforward levels. PV tuning is sampled independently of PyC tuning, so the
# same template can create different full-image surround regimes depending on the
# overlap state
PV_STRENGTHS: dict[str, dict[str, float]] = {
    "very_weak": dict(tuned=0.045, silent=0.010),
    "weak": dict(tuned=0.24, silent=0.050),
    "mid": dict(tuned=0.31, silent=0.130),
    "strong": dict(tuned=0.66, silent=0.170),
    "very_strong": dict(tuned=0.88, silent=0.280),
}

# Generalized feedback levels (equal across all three context channels):
# (receives_context, center, rel_noise, noise_floor, lo, hi).
FB_LEVELS: dict[str, dict[str, Any]] = {
    "none": dict(receives=False, center=0.004, **UNIFORM_FB_NOISE),
    "very_weak": dict(receives=True, center=0.015, **UNIFORM_FB_NOISE),
    "weak": dict(receives=True, center=0.050, **UNIFORM_FB_NOISE),
    "mid": dict(receives=True, center=0.075, **UNIFORM_FB_NOISE),
    "strong": dict(receives=True, center=0.110, **UNIFORM_FB_NOISE),
    # High initial FB: creates naive O responders
    # without making their post-training O shift dominate the novel scatter. Tight
    # rel-noise + raised lo keep the O spread (hence the NO spread = O - gain*Dy_lat)
    # narrow so the expert O cloud sits squarely at NO=0 rather than smearing into a
    # negative tail / the diagonal.
    "very_strong": dict(receives=True, center=0.300, **UNIFORM_FB_NOISE),
}

# Which PyC feedforward channels each cell prefers (drawn per cell):
#   "all"       -> two random preferred images for broad cells
#   "permuted1" -> one random preferred image for narrow cells
NOVEL_INDEX = 2
BROAD_TUNING_WIDTH = 2
PV_TUNING_WIDTH = 2
CANONICAL_BROAD_TUNED_INDICES = (0, NOVEL_INDEX)
CANONICAL_PV_TUNED_INDICES = (0, 1)

# Feedback context is deliberately curtailed for this variant: feedforward PyC/PV
# overlap already creates the main variability, so templates either receive no FB
# context or all context channels.
CONTEXT_MODES = ("none", "all")

# The population mixture. Templates set strength levels for PyC tuned/untuned FF
# and PV tuned/untuned FF plus feedback level; the sampled PyC/PV tuning overlap
# determines which image gets which member of each tuned/untuned pair.
TEMPLATES: dict[str, dict[str, Any]] = {
    "silent_broad_FFonly": dict(width="broad", ff="silent", pv="weak", fb="none", tuning="all", context="none", weight=0.012, baseline=0.18),
    "silent_broad_FB_mid": dict(width="broad", ff="silent", pv="very_strong", fb="mid", tuning="all", context="all", weight=0.035, drive=0.040, gain=2.8, baseline=0.18),
    "silent_broad_FB_strong": dict(width="broad", ff="silent", pv="very_strong", fb="very_strong", tuning="all", context="all", weight=0.028, drive=0.160, gain=2.0, baseline=0.26),

    # Broad no-FB cells are the main pure -NO source. Independent PV tuning makes
    # the same template sometimes more strongly shunted on familiar or novel.
    "weak_broad_FFonly": dict(width="broad", ff="weak", pv="weak", fb="none", tuning="all", context="none", weight=0.060, baseline=0.32),
    "mid_broad_FFonly": dict(width="broad", ff="mid", pv="strong", fb="none", tuning="all", context="none", weight=0.180, baseline=0.32),
    "strong_broad_FFonly": dict(width="broad", ff="strong", pv="strong", fb="none", tuning="all", context="none", weight=0.380, baseline=0.32),

    # Broad FB cells are the desired overlap-dependent movers. In the canonical
    # center example PyC={familiar_1, novel} and PV={familiar_1, familiar_2}, so
    # familiar_1 can adapt toward -NO/+O while novel keeps FF drive with weak PV
    # surround and can move +NO/+O after generalized FB growth. Three of these
    # templates (weak/mid/strong_broad_FB_all) share the same low drive_threshold
    # and mid initial FB so the drive crosses threshold as w_fb grows during
    # training -- giving the NO->O up-and-left transition at THREE FF strengths
    # (weak, mid, strong) instead of only strong, and a real +NO+O mixed
    # population at (~0.5, 0.5) on novel where FF is intact and FB-drive grows
    # together. mixed_broad_FB_all keeps a higher drive_threshold so it stays a
    # pure +NO mover on novel (drive never activates), seeding the dense +NO
    # cloud at O~0.
    "weak_broad_FB_all": dict(width="broad", ff="mid", pv="strong", fb="mid", tuning="all", context="all", weight=0.080, drive=0.080, gain=6.0, gain_clip=(1.5, 7.5), gain_k=7.0, baseline=0.22),
    "mid_broad_FB_all": dict(width="broad", ff="strong", pv="strong", fb="mid", tuning="all", context="all", weight=0.110, drive=0.080, gain=6.4, gain_clip=(1.5, 8.0), gain_k=7.0, baseline=0.22),
    "strong_broad_FB_all": dict(width="broad", ff="strong", pv="very_strong", fb="mid", tuning="all", context="all", weight=0.110, drive=0.080, gain=7.2, gain_clip=(1.5, 9.0), baseline=0.22),
    "mixed_broad_FB_all": dict(width="broad", ff="strong", pv="mid", fb="weak", tuning="all", context="all", weight=0.280, drive=0.240, gain=7.8, gain_clip=(1.5, 9.0), gain_k=8.0, baseline=0.22),

    # Narrow cells sample their one preferred image uniformly; no template is
    # explicitly novel-tuned. Their share controls the +NO source size.
    "narrow_super_weak_FB_all": dict(width="narrow", ff="super_weak", pv="very_weak", fb="weak", tuning="permuted1", context="all", weight=0.220, gain=8.0, gain_clip=(1.5, 9.5)),
    "narrow_very_weak_FB_all": dict(width="narrow", ff="weak", pv="very_weak", fb="weak", tuning="permuted1", context="all", weight=0.540, gain=8.4, gain_clip=(1.5, 10.0)),
    "narrow_mid_FFonly": dict(width="narrow", ff="mid", pv="weak", fb="none", tuning="permuted1", context="none", weight=0.01, gain=7.5, gain_clip=(1.5, 8.0)),
}

SURROUND_SETTINGS: dict[str, dict[str, float]] = {
    "silent_broad_FFonly": dict(lat=0.04, pvlat=0.08),
    "silent_broad_FB_mid": dict(lat=0.9, pvlat=0.2),
    "silent_broad_FB_strong": dict(lat=0.85, pvlat=0.05),
    "weak_broad_FFonly": dict(lat=0.035, pvlat=0.10),
    "mid_broad_FFonly": dict(lat=0.050, pvlat=0.10),
    "strong_broad_FFonly": dict(lat=0.050, pvlat=0.10),
    "weak_broad_FB_all": dict(lat=0.24, pvlat=0.05),
    "mid_broad_FB_all": dict(lat=0.24, pvlat=0.05),
    "strong_broad_FB_all": dict(lat=0.24, pvlat=0.05),
    "mixed_broad_FB_all": dict(lat=0.36, pvlat=0.05),
    "narrow_super_weak_FB_all": dict(lat=0.03, pvlat=0.03),
    "narrow_very_weak_FB_all": dict(lat=0.03, pvlat=0.03),
    "narrow_mid_FFonly": dict(lat=0.03, pvlat=0.03),
}


# Per-template scalar overrides of the width-class defaults.
_SCALAR_OVERRIDES = {
    "gain": "apical_gain_strength",
    "drive": "apical_drive_threshold",
    "ff_plasticity_scale": "ff_plasticity_scale",
    "gain_threshold": "apical_gain_threshold",
    "gain_k": "apical_gain_k",
}


def _random_indices(width: int, rng: np.random.Generator) -> tuple[int, ...]:
    if width <= 0 or width > N_FEATURES:
        raise ValueError(f"width must be in [1, {N_FEATURES}], got {width}.")
    return tuple(sorted(int(idx) for idx in rng.choice(N_FEATURES, size=width, replace=False)))


def _draw_tuned_indices(tuning: str, width: str, rng: np.random.Generator) -> tuple[int, ...]:
    if width == "broad":
        return _random_indices(BROAD_TUNING_WIDTH, rng)
    if tuning == "all":
        return tuple(range(N_FEATURES))
    if tuning == "permuted1":
        return (int(rng.integers(0, N_FEATURES)),)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _draw_pv_tuned_indices(rng: np.random.Generator) -> tuple[int, ...]:
    return _random_indices(PV_TUNING_WIDTH, rng)


def _draw_context_indices(mode: str, rng: np.random.Generator) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    raise ValueError(f"unknown context mode: {mode}")


def _canonical_context_indices(mode: str) -> tuple[int, ...]:
    if mode == "none":
        return ()
    if mode == "all":
        return tuple(range(N_FEATURES))
    raise ValueError(f"unknown context mode: {mode}")


def _vector(tuned_indices: tuple[int, ...], tuned: float, silent: float) -> list[float]:
    tuned_set = set(tuned_indices)
    return [float(tuned if idx in tuned_set else silent) for idx in range(N_FEATURES)]


def _general_vector(value: float) -> list[float]:
    return [float(value)] * N_FEATURES


def _bool_vector(indices: tuple[int, ...]) -> tuple[bool, ...]:
    index_set = set(indices)
    return tuple(idx in index_set for idx in range(N_FEATURES))


def _build_transition_specs() -> dict[str, dict[str, Any]]:
    """Translate the TEMPLATES table into base ``transition`` specs (only their
    ``weight`` is read downstream; the real config is built in the perturb hook)."""
    specs: dict[str, dict[str, Any]] = {}
    for name, template in TEMPLATES.items():
        specs[name] = base.transition(template["weight"])
    return specs


def _build_config(
    name: str,
    tuned_indices: tuple[int, ...],
    pv_tuned_indices: tuple[int, ...],
    context_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Assemble the (noise-free) config for one cell of a template, given the drawn
    preferred-stimulus set ``tuned_indices``."""
    template = TEMPLATES[name]
    width_class = WIDTH_CLASSES[template["width"]]
    ff = FF_STRENGTHS[template["ff"]]
    pv = PV_STRENGTHS[template["pv"]]
    fb = FB_LEVELS[template["fb"]]
    surround = SURROUND_SETTINGS[name]

    pv_vec = _vector(pv_tuned_indices, pv["tuned"], pv["silent"])
    config: dict[str, Any] = {
        "w_ff_init": base.weight_init(
            _vector(tuned_indices, ff["tuned"], ff["silent"]),
            ff["rel"], ff["floor"],
            _vector(tuned_indices, ff["lo"], ff["lo"]),
            _vector(tuned_indices, ff["hi"], ff["hi"]),
        ),
        "w_fb_init": base.weight_init(
            _general_vector(fb["center"]), fb["rel"], fb["floor"],
            _general_vector(fb["lo"]), _general_vector(fb["hi"]),
        ),
        "w_lat_init": base.weight_init([surround["lat"]]),
        "w_pv_lat_init": base.weight_init([surround["pvlat"]]),
        "W_pv_init": base.weight_init(pv_vec),
    }
    fixed = {
        "receives_context": _bool_vector(context_indices) if fb["receives"] else (False,) * N_FEATURES,
        "baseline_drive_sigma": template.get("baseline", width_class["baseline"]),
        "pv_noise_sigma": PV_NOISE_SIGMA,
        "pv_plasticity": False,
        "pv_lat_plasticity": False,
    }
    # Width-class scalars, with optional per-template overrides.
    for key, model_key in _SCALAR_OVERRIDES.items():
        value = template.get(key, width_class.get(key))
        if value is not None:
            fixed[model_key] = value
    clip = {
        "apical_gain_strength": template.get("gain_clip", width_class["gain_clip"]),
        "apical_drive_threshold": template.get("drive_clip", width_class["drive_clip"]),
    }
    return {
        "init": config,
        "fix": fixed,
        "clip": clip,
        "width": len(tuned_indices),
        "pv_width": len(pv_tuned_indices),
        "context_width": len(context_indices),
    }


def _perturb_config_factory():
    """Replacement for ``base._perturb_config`` that draws each cell's preferred
    stimulus (FF tuning) at random, then applies the usual FF/FB weight noise and
    scalar jitter and clipping."""

    def _perturb_config(
        transition: str,
        base_config: dict[str, Any],
        *,
        sample_idx: int,
        global_idx: int,
        seed: int,
        rng: np.random.Generator,
        scalar_noise_multiplier: float,
    ) -> dict[str, Any]:
        template = TEMPLATES[transition]
        tuned_indices = _draw_tuned_indices(template["tuning"], template["width"], rng)
        pv_tuned_indices = _draw_pv_tuned_indices(rng)
        context_indices = _draw_context_indices(template.get("context", "all"), rng)
        spec = _build_config(transition, tuned_indices, pv_tuned_indices, context_indices)
        config = copy.deepcopy(base_config)
        for key, init_spec in spec["init"].items():
            if key in {"w_ff_init", "w_fb_init"}:
                values = base._draw_init(init_spec, rng)
            else:
                values = base._center_init_values(init_spec)
            base._set_init(config, key, values)

        config.update(spec["fix"])
        for key, scalar_spec in base.SCALAR_NOISE.items():
            if key in config and base._is_num(config[key]):
                config[key] = base._draw_scalar(float(config[key]), scalar_spec, rng, scalar_noise_multiplier)
        for key, (lo, hi) in (base.GLOBAL_SCALAR_CLIP | spec["clip"]).items():
            if key in config and base._is_num(config[key]):
                config[key] = base._clip(float(config[key]), lo, hi)
        base._apply_shared_learning_rates(config)

        config.update(
            seed=int(seed),
            _canonical_transition=transition,
            _sample_idx=int(sample_idx),
            _sample_global_idx=int(global_idx),
            _ff_tuning_width=int(spec["width"]),
            _ff_strength=template["ff"],
            _pv_strength=template["pv"],
            _fb_level=template["fb"],
            _tuned_indices=list(tuned_indices),
            _pv_tuned_indices=list(pv_tuned_indices),
            _context_indices=list(context_indices),
        )
        return config

    return _perturb_config


def _canonical_tuned_indices(tuning: str) -> tuple[int, ...]:
    """A fixed, representative preferred-stimulus set for a tuning mode, used for the
    noise-free center-panel traces (permuted cells are shown tuned to familiar_1)."""
    if tuning == "all":
        return tuple(range(N_FEATURES))
    if tuning == "permuted1":
        return (0,)
    raise ValueError(f"unknown tuning mode: {tuning}")


def _canonical_pyc_tuned_indices(template: dict[str, Any]) -> tuple[int, ...]:
    if template["width"] == "broad":
        return CANONICAL_BROAD_TUNED_INDICES
    return _canonical_tuned_indices(template["tuning"])


def _canonical_pv_tuned_indices() -> tuple[int, ...]:
    return CANONICAL_PV_TUNED_INDICES


def _center_config(
    name: str,
    *,
    tuned_indices: tuple[int, ...] | None = None,
    pv_tuned_indices: tuple[int, ...] | None = None,
    context_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Noise-free config at the template centers (no weight/scalar jitter), for the
    center-panel trace plots. Replaces ``base._center_config`` so each pruned
    template renders correctly."""
    template = TEMPLATES[name]
    tuned_indices = tuned_indices or _canonical_pyc_tuned_indices(template)
    pv_tuned_indices = pv_tuned_indices or _canonical_pv_tuned_indices()
    context_indices = context_indices or _canonical_context_indices(template.get("context", "all"))
    spec = _build_config(name, tuned_indices, pv_tuned_indices, context_indices)
    config = copy.deepcopy(base.minimal_configs3[name])
    for key, init_spec in spec["init"].items():
        base._set_init(config, key, base._center_init_values(init_spec))
    config.update(spec["fix"])
    for key, (lo, hi) in (base.GLOBAL_SCALAR_CLIP | spec["clip"]).items():
        if key in config and base._is_num(config[key]):
            config[key] = base._clip(float(config[key]), lo, hi)
    base._apply_shared_learning_rates(config)
    config.update(
        _canonical_transition=name,
        _sample_idx=0,
        _sample_global_idx=0,
        _ff_tuning_width=int(spec["width"]),
        _ff_strength=template["ff"],
        _pv_strength=template["pv"],
        _fb_level=template["fb"],
        _tuned_indices=list(tuned_indices),
        _pv_tuned_indices=list(pv_tuned_indices),
        _context_indices=list(context_indices),
    )
    return config


def _flatten_config_factory(original_flatten):
    def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
        flat = original_flatten(config)
        flat["ff_tuning_width"] = config.get("_ff_tuning_width")
        flat["pv_tuning_width"] = len(config.get("_pv_tuned_indices", []))
        flat["ff_strength"] = config.get("_ff_strength")
        flat["pv_strength"] = config.get("_pv_strength")
        flat["fb_level"] = config.get("_fb_level")
        tuned = config.get("_tuned_indices", [])
        pv_tuned = config.get("_pv_tuned_indices", [])
        context = config.get("_context_indices", [])
        for idx in range(N_FEATURES):
            flat[f"tuned_index_{idx}"] = int(idx in tuned)
            flat[f"pv_tuned_index_{idx}"] = int(idx in pv_tuned)
            flat[f"receives_context_{idx}"] = int(idx in context)
        return flat

    return _flatten_config


def configure_model_scatter(n_steps_per_phase: int = LEARNING_RATE_REFERENCE_STEPS) -> None:
    if set(SURROUND_SETTINGS) != set(TEMPLATES):
        missing = sorted(set(TEMPLATES) - set(SURROUND_SETTINGS))
        extra = sorted(set(SURROUND_SETTINGS) - set(TEMPLATES))
        raise ValueError(f"SURROUND_SETTINGS must match TEMPLATES exactly; missing={missing}, extra={extra}")
    transitions = _build_transition_specs()
    generic_config = copy.deepcopy(BASE_CONFIG)
    generic_config["activation"] = ThresholdReLU(threshold=SOMA_ACTIVATION_THRESHOLD, subtractive=False, hasMax=True, maxValue=1.0)
    generic_config["apical_drive_subtractive"] = APICAL_DRIVE_SUBTRACTIVE
    generic_config["pv_noise_sigma"] = PV_NOISE_SIGMA
    generic_config["divisive_gain"] = DIVISIVE_GAIN
    generic_config["apical_gain_threshold"] = SHARED_GAIN_THRESHOLD
    base.TRANSITIONS = transitions
    base.minimal_configs3 = {name: copy.deepcopy(generic_config) for name in transitions}
    base.SHARED_LEARNING_RATES = _effective_learning_rates(n_steps_per_phase)
    base.SCALAR_NOISE = {key: SCALAR_NOISE[key] for key in SCALAR_NOISE_KEYS}
    base.GLOBAL_SCALAR_CLIP = {}
    base.BASELINE_STD_SCALE = BASELINE_STD_SCALE
    base._perturb_config = _perturb_config_factory()
    base._flatten_config = _flatten_config_factory(base._flatten_config)
    # Center-panel traces use the noise-free template centers.
    base._center_config = _center_config
    # base._canonical_config = _center_config


def write_metadata(args) -> None:
    path = args.output_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata["pruned_mini_variant"] = {
        "description": "Mini model (only PyC plasticity) with a small width-keyed principled template set, generalized feedback, and emergent FF tuning via random per-cell permutation.",
        "disabled_plasticity": ["W_pv", "w_pv_lat"],
        "enabled_plasticity": ["w_ff", "w_fb", "w_lat"],
        "model": "minimal_divisive.CCNeuron",
        "divisive_gain": DIVISIVE_GAIN,
        "shared_learning_rates": _effective_learning_rates(args.n_steps_per_phase),
        "base_shared_learning_rates": SHARED_LEARNING_RATES,
        "learning_rate_reference_steps": LEARNING_RATE_REFERENCE_STEPS,
        "soma_activation_threshold": SOMA_ACTIVATION_THRESHOLD,
        "apical_drive_subtractive": APICAL_DRIVE_SUBTRACTIVE,
        "pv_noise_sigma": PV_NOISE_SIGMA,
        "uniform_ff_noise": UNIFORM_FF_NOISE,
        "uniform_fb_noise": UNIFORM_FB_NOISE,
        "uniform_gain_clip": UNIFORM_GAIN_CLIP,
        "uniform_drive_clip": UNIFORM_DRIVE_CLIP,
        "scalar_noise_keys": list(SCALAR_NOISE_KEYS),
        "width_classes": WIDTH_CLASSES,
        "ff_strengths": FF_STRENGTHS,
        "pv_strengths": PV_STRENGTHS,
        "fb_levels": FB_LEVELS,
        "context_modes": CONTEXT_MODES,
        "broad_tuning_width": BROAD_TUNING_WIDTH,
        "pv_tuning_width": PV_TUNING_WIDTH,
        "canonical_broad_tuned_indices": list(CANONICAL_BROAD_TUNED_INDICES),
        "canonical_pv_tuned_indices": list(CANONICAL_PV_TUNED_INDICES),
        "surround_settings": SURROUND_SETTINGS,
        "templates": {name: {k: v for k, v in t.items()} for name, t in TEMPLATES.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, default=repr))
