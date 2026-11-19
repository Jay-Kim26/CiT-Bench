"""Seven base families under one renderer: a one-cycle template on u in [0, 1), a phase track that
accumulates at 1/P per step (optionally with per-cycle jitter and rate changes from frequency
events), and a linear drift for trend_sine. IMP renders in raw units, AD in standardized units."""
import sys

import numpy as np

from spec import TS


from synth_ad import families as F  # noqa: E402

PERIODIC_SHAPE = {"sine", "trend_sine"}


def sample_shape(family, rng):
    if family in PERIODIC_SHAPE:
        return {}
    if family == "harmonic":
        n = int(rng.choice([2, 3]))
        return {"amps": [1.0] + [float(rng.uniform(.25, .6) / k) for k in range(2, n + 1)],
                "phis": [float(rng.uniform(0, 2 * np.pi)) for _ in range(n)]}
    return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in F.sample_shape_params(family, rng).items()}


def template(family, u, shape):
    u = np.mod(np.asarray(u, dtype=float), 1.0)
    if family in PERIODIC_SHAPE:
        return np.sin(2 * np.pi * u)
    if family == "harmonic":
        return sum(a * np.sin(2 * np.pi * k * u + ph) for k, (a, ph) in enumerate(zip(shape["amps"], shape["phis"]), 1))
    return F.WAVE_FN[family](u, {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in shape.items()}).astype(float)


def normalizer(family, shape):
    g = np.linspace(0, 1, 4000, endpoint=False)
    v = template(family, g, shape)
    return float((v.min() + v.max()) / 2), float(np.ptp(v))


def unit_shape(family, u, shape, norm):
    """Template rescaled to [-0.5, 0.5]: the usual signal is centre + swing * unit_shape."""
    mid, ptp = norm
    return (template(family, u, shape) - mid) / ptp


def phase_track(L, P, phase0, rate_mult=None, jitter=0.0, rng=None):
    """u[t]: phase in cycles. The step INTO t advances the phase by rate_mult[t] / P, so the samples of a
    frequency event (rate_mult != 1 on [s, e]) are exactly the steps s..e stated in the prompt and labelled
    in AD; with jitter every new cycle draws its own length P (1 + U(-j, j))."""
    u = np.empty(L)
    u[0] = phase0
    mult = np.ones(L) if rate_mult is None else np.asarray(rate_mult, dtype=float)
    cycle = P * (1 + rng.uniform(-jitter, jitter)) if jitter else float(P)
    for t in range(1, L):
        u[t] = u[t - 1] + mult[t] / cycle
        if jitter and np.floor(u[t]) != np.floor(u[t - 1]):
            cycle = P * (1 + rng.uniform(-jitter, jitter))
    return u


def sampled_extremes(base):
    """Lowest and highest unit-shape value at the phases one cycle is actually sampled at (phase0 + t / P)."""
    norm = normalizer(base["family"], base["shape"])
    u = base["phase0"] + np.arange(base["P"]) / base["P"]
    v = unit_shape(base["family"], u, base["shape"], norm)
    return float(v.min()), float(v.max())


def observed_stats(base, scale):
    """(range, centre offset) of the sampled usual signal: scale x (max - min) and scale x (max + min) / 2.
    The usual range, the envelopes and the centre line stated in the prompts are these observable values,
    not the extremes of the continuous shape between samples."""
    lo, hi = sampled_extremes(base)
    return scale * (hi - lo), scale * (hi + lo) / 2


def sample_imp_base(family, P, L, rng):
    """Raw-unit parameters in the style of generate_citb_imp (range R = 10^U(1,2), offset U(-50,150))."""
    shape = sample_shape(family, rng)
    R = float(10 ** rng.uniform(1, 2))
    b0 = float(rng.uniform(-50, 150))
    phase0 = float(rng.uniform(0, 1))
    slope = float(rng.choice([-1, 1]) * rng.uniform(.5, 1.0) * R / (L - 1)) if family == "trend_sine" else 0.0
    return {"family": family, "P": int(P), "shape": shape, "R": R, "b0": b0, "phase0": phase0, "slope": slope, "L": int(L)}


def render_imp(base, rate_mult=None):
    """Usual signal in raw units for a phase track. Returns (signal, centre line); swing = R."""
    L = base["L"]
    t = np.arange(L)
    u = phase_track(L, base["P"], base["phase0"], rate_mult)
    norm = normalizer(base["family"], base["shape"])
    centre = base["b0"] + base["slope"] * (t - (L - 1) / 2)
    return centre + base["R"] * unit_shape(base["family"], u, base["shape"], norm), centre


def sample_ad_base(family, P, L, rng):
    shape = sample_shape(family, rng)
    jitter = 0.0                                                     # final v1: fixed sampling phases (like IMP); the stated range / envelopes are then exactly the sampled extremes
    slope = float(rng.choice([-1, 1]) * rng.uniform(.6, 1.2)) if family == "trend_sine" else float(rng.uniform(-.3, .3))
    return {"family": family, "P": int(P), "shape": shape, "phase0": float(rng.uniform(0, 1)), "jitter": jitter,
            "slope": slope, "noise_sigma": float(rng.uniform(.02, .15)), "L": int(L)}


def render_ad(base, rate_mult=None, jitter_seed=None):
    """Unit-variance zero-mean periodic part + linear trend, in pre-standardization units.
    Returns (signal, centre line, swing). The jitter draws come from `jitter_seed` so that the usual
    track and the event track share cycle lengths."""
    L = base["L"]
    t = np.arange(L)
    rng = np.random.default_rng(jitter_seed)
    u = phase_track(L, base["P"], base["phase0"], rate_mult, base["jitter"], rng)
    norm = normalizer(base["family"], base["shape"])
    us = unit_shape(base["family"], u, base["shape"], norm)          # in [-0.5, 0.5]
    g = np.linspace(0, 1, 4000, endpoint=False)
    ref = unit_shape(base["family"], g, base["shape"], norm)
    scale = 1.0 / float(ref.std())                                    # unit variance over a full cycle
    trend = base["slope"] * (t - (L - 1) / 2) / L
    return scale * us + trend, trend, scale                           # centre line = trend (midpoint of the envelope is 0)
