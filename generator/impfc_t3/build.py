#!/usr/bin/env python3
"""CiT-Bench 2027 IMP and FC with three Temporal types ("t3"): 200 scenes x (L1, L2) = 400 queries per task.

Design agreed on 2026-09-25 (examples: ../temporal_redesign/TEMPORAL_EXAMPLES.md):
  * No A/B pairs. Every scene yields an L1 query (explicit numbers) and an L2 query (relative to the usual
    pattern or to what was observed); both share the observations and the ground truth.
  * Magnitude: the v15.1 IMP and roi_trend_v3 FC families, mechanics and wording, with the scene counts doubled.
  * Temporal:
      frequency        v15.1 IMP / roi_trend_v3 FC.
      self_repetition  a quiet level with irregular bursts; the interval repeats the pattern of an earlier burst
                       (L1: its steps, L2: "its first / second burst").
      delay            two series on an irregular wave; Y follows X with a usual delay that is visible in the
                       observations and changes during the interval (L1: the steps of X, L2: half / twice the
                       usual delay). Joints are placed where the two delays meet smoothly.
  * FC: every change runs from its first step to the end of the forecast (no return); for delay scenes series X
    is observed through the last forecast step, so the answer is readable for either direction of the change.
  * Prompts: the Final global answer-only format, without the words "RoI" and "event".
No model inference is performed.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_generator"))
import bases  # noqa: E402  periodic templates, observed statistics, raw-unit rendering
import util  # noqa: E402  decimals_for

VERSION = {"imp": "cit-bench-2027-imp-t3-v1", "fc": "cit-bench-2027-fc-t3-v1"}
QUOTAS = {
    "imp": {"shift": 24, "amplitude": 28, "flat": 24, "bounding": 24,
            "frequency": 34, "self_repetition": 33, "delay": 33},
    "fc": {"shift": 20, "amplitude": 20, "flat": 20, "bounding": 20, "trend_rate": 20,
           "frequency": 34, "self_repetition": 33, "delay": 33},
}
TEMPORAL = ("frequency", "self_repetition", "delay")
BASES = ("sine", "harmonic", "trend_sine", "rectified_sine", "exp_rise_decay", "pulse_train", "asym_bimodal")
SMOOTH = {"sine", "harmonic", "trend_sine"}
GAPS = (24, 32, 48)
IMP_L = (128, 256)
HISTORY = (128, 160)
HORIZONS = (24, 32, 48)
MAX_ATTEMPTS = 4000
MIN_EFFECT_SIGMA = 2.0          # mean |change - usual| over the interval, in noise sigmas
DELAY_EFFECT = 0.15             # delay scenes: mean |new delay - usual delay| over the interval, in Y's observed range
SYSTEM = {"imp": "You are a useful time series imputation assistant.",
          "fc": "You are a useful time series forecasting assistant."}


def axis_of(family):
    return "temporal" if family in TEMPORAL else "magnitude"


def rng_for(task, sid, attempt):
    h = hashlib.sha256(f"{VERSION[task]}|{sid}|{attempt}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def q(x, grid):
    """Round to the display grid, never to zero (v15.1 events.q)."""
    v = float(np.round(x / grid) * grid)
    return v if abs(v) >= grid else float(np.sign(x) * grid or grid)


def word(table, f):
    for k, v in table.items():
        if abs(k - f) < 1e-9:
            return v
    raise KeyError(f)


def number(value):
    value = round(float(value), 2)
    return f"{value:.2f}".rstrip("0").rstrip(".") if value else "0"


def places(d):
    return f"{d} decimal place{'s' if d != 1 else ''}"


# ------------------------------------------------------------------------------------------------ wording
# IMP (v15.1, past tense)
IMP_DEF = {
    "range": "The usual range is the usual signal's highest value minus its lowest within one cycle.",
    "interval": "The usual interval is the number of steps after which the usual signal repeats.",
    "centre": ("The upper and lower envelopes are the usual signal's highest and lowest values within one cycle, "
               "and the centre line is their midpoint; all three follow any gradual drift."),
}
IMP_AMP_DEF = IMP_DEF["centre"] + " The usual range is the distance between the upper and lower envelopes, so it excludes that drift."
RANGE_FRAC = {.25: "a quarter of the usual range", .5: "half the usual range", .75: "three quarters of the usual range"}
AMP_RANGE = {.5: "half the usual range", 1.5: "one and a half times the usual range"}
PERIOD_RATIO = {.5: "half the usual interval", 1 / 3: "a third of the usual interval", .25: "a quarter of the usual interval",
                2.: "twice the usual interval"}
LEVEL_FRAC = {.5: "midway between the centre line and the upper envelope",
              .25: "a quarter of the way from the centre line to the upper envelope",
              -.25: "a quarter of the way from the centre line to the lower envelope",
              -.5: "midway between the centre line and the lower envelope",
              1.: "at the upper envelope", -1.: "at the lower envelope"}
# FC (roi_trend_v3, future tense)
FC_ENVELOPES = ("The upper and lower envelopes are the usual signal's highest and lowest values within one cycle, "
                "and the centre line is their midpoint; all three follow any gradual drift.")
FC_RANGE = "The usual range is the usual signal's highest value minus its lowest within one cycle, excluding gradual drift."
FC_INTERVAL = "The usual interval is the number of steps between successive repetitions of the usual pattern."
FC_RATIO = {.25: "a quarter of its usual interval", 1 / 3: "a third of its usual interval", .5: "half its usual interval",
            2.: "twice its usual interval"}
DELAY_HEADER = ("Series Y is a delayed copy of series X: apart from noise, each value of Y repeats the value of X "
                "a fixed number of steps earlier, called the delay.")


def fc_relative_position(sign, fraction, s):
    direction = sign * fraction
    envelope = "upper" if direction > 0 else "lower"
    amount = abs(fraction)
    if amount == 1:
        return f"at the {envelope} envelope as of step {s}"
    part = {.25: "a quarter of the way", .5: "halfway", .75: "three quarters of the way"}[amount]
    return f"{part} from the centre line to the {envelope} envelope as of step {s}"


# ------------------------------------------------------------------------------------------------ shared signal helpers
def align_centre_sample(base, rng):
    """Choose phase0 so that the sampled usual signal meets its centre line (roi_trend_v3 / v15.1 amplitude)."""
    P = base["P"]

    def residual(phase):
        v = bases.template(base["family"], np.asarray(phase)[..., None] + np.arange(P) / P, base["shape"])
        return v[..., 0] - (v.min(axis=-1) + v.max(axis=-1)) / 2
    grid = np.linspace(0, 1, 513)
    r = residual(grid)
    brackets = np.flatnonzero(r[:-1] * r[1:] <= 0)
    if not len(brackets):
        return False
    k = int(rng.choice(brackets))
    lo, hi = grid[k:k + 2]
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(residual(lo)) * float(residual(mid)) <= 0:
            hi = mid
        else:
            lo = mid
    base["phase0"] = float(((lo + hi) / 2 - int(rng.integers(P)) / P) % 1)
    return True


SINGLE = ("hump", "skewed", "ramp_up", "ramp_down", "plateau")
DOUBLE_P = .35          # share of two-peaked bursts among the bursts that are not the referenced one
DOUBLE_MIN_W = 12       # a two-peaked burst needs at least this many steps to show both peaks


def peaks(s):
    """Indices of the local maxima of a noise-free burst (a flat top counts once)."""
    out, k = [], 1
    while k < len(s) - 1:
        m = k
        while m + 1 < len(s) - 1 and abs(s[m + 1] - s[k]) < 1e-12:
            m += 1
        if s[k] > s[k - 1] + 1e-12 and s[m] > s[m + 1] + 1e-12:
            out.append((k + m) // 2)
        k = m + 1
    return out


def burst_ok(s, kind):
    """Single-peaked: rises, then falls. Two-peaked: two peaks >= 3 steps apart, the dip between them visible
    (<= 85% of the lower peak) but high (>= 40% of the burst height), so the two peaks read as one burst."""
    pk = peaks(s)
    if kind != "double":
        return len(pk) == 1
    if len(pk) != 2 or pk[1] - pk[0] < 3:
        return False
    saddle = float(s[pk[0]:pk[1] + 1].min())
    return .4 * s.max() <= saddle <= .85 * min(s[pk[0]], s[pk[1]])


def burst_shape(w, rng, kind=None):
    """A smooth burst of w steps with height 1 whose first and last steps sit exactly on the quiet level.
    A burst leaves the quiet level and returns to it; it has one peak, or two peaks with a high dip between them.
    kind: None (random: two-peaked with probability DOUBLE_P when w allows), "single" (any single-peaked kind) or "double"."""
    t = np.linspace(0, 1, w)
    taper = .5 - .5 * np.cos(np.pi * np.clip(np.minimum(t, 1 - t) / .25, 0, 1))
    if kind is None:
        kind = "double" if w >= DOUBLE_MIN_W and rng.uniform() < DOUBLE_P else "single"
    for _ in range(1000):
        k = "double" if kind == "double" else SINGLE[int(rng.integers(len(SINGLE)))]
        if k == "hump":
            core = np.exp(-((t - rng.uniform(.35, .65)) / rng.uniform(.12, .22)) ** 2)
        elif k == "skewed":
            a, b = rng.uniform(1.5, 4), rng.uniform(1.5, 4)
            core = t ** a * (1 - t) ** b
        elif k == "ramp_up":
            core = t ** rng.uniform(1, 3)
        elif k == "ramp_down":
            core = (1 - t) ** rng.uniform(1, 3)
        elif k == "plateau":
            a, b = rng.uniform(.15, .35), rng.uniform(.65, .85)
            core = 1 / (1 + np.exp(-(t - a) * 30)) / (1 + np.exp((t - b) * 30))
        else:
            c1, c2 = rng.uniform(.25, .38), rng.uniform(.62, .75)
            level = rng.uniform(.45, .65)
            shelf = 1 / (1 + np.exp(-(t - c1) * 25)) / (1 + np.exp((t - c2) * 25))
            bumps = sum(rng.uniform(.7, 1.) * np.exp(-((t - c) / rng.uniform(.06, .1)) ** 2) for c in (c1, c2))
            core = level * shelf + (1 - level) * bumps
        shape = core * taper
        shape = shape / shape.max()
        if burst_ok(shape, k):
            return shape, k
    raise RuntimeError(f"no {kind} burst of {w} steps")


def smooth_wave(n, rng):
    """Irregular smooth wave in [0, 1] without a period (Gaussian-smoothed noise)."""
    scale = rng.uniform(2.5, 4.0)
    k = np.exp(-.5 * (np.arange(-12, 13) / scale) ** 2)
    w = np.convolve(rng.normal(size=n + 80), k / k.sum(), mode="same")[40:40 + n]
    return (w - w.min()) / (w.max() - w.min()), float(scale)


def curvature_ok(y, joints, limit):
    """|second difference| at and next to every joint stays within the wave's own 90th percentile."""
    return all(abs(y[t + 1] - 2 * y[t] + y[t - 1]) <= limit for j in joints for t in (j - 1, j) if 0 < t < len(y) - 1)


# ------------------------------------------------------------------------------------------------ IMP scenes
def imp_layout(rng, P, W, D):
    """v15.1 layout: lead >= 1.5 cycles, tail >= 1 cycle, >= 2 usual steps on either side of the interval."""
    lead, tail = int(np.ceil(1.5 * P)) + int(rng.integers(2, 7)), P + int(rng.integers(2, 7))
    L = int(rng.integers(max(lead + W + tail, IMP_L[0], 4 * W), IMP_L[1] + 1))
    gs = int(rng.integers(lead, L - tail - W + 1))
    s = gs + int(rng.integers(2, W - D - 1))
    return L, gs, s


def imp_probe(rng, bf, P):
    probe = bases.sample_imp_base(bf, P, IMP_L[1], rng)
    S, c_off = bases.observed_stats(probe, probe["R"])
    sigma = float(rng.uniform(.02, .04) * S)
    return probe, S, c_off, sigma


def imp_render(probe, L, c_off, rate=None):
    base = dict(probe, L=L, slope=probe["slope"] * (IMP_L[1] - 1) / (L - 1))
    usual, centre = bases.render_imp(base)
    mean = bases.render_imp(base, rate)[0] if rate is not None else usual.copy()
    return base, usual, centre + c_off, mean


def imp_periodic(family, j, rng, bf, W, attempt=0):
    """One v15.1-style scene (shift / flat / bounding / amplitude / frequency) with a single condition."""
    sign = (1, -1)[j % 2]
    if family == "frequency":
        slow = j % 2 == 1
        if bf == "harmonic":
            P = 20
        elif slow:
            P = 12 if W == 32 else int(rng.choice([16, 20]))
        else:
            P = int(rng.choice([12, 16, 20]))
        if slow:
            ratio, D = 2.0, 2 * P
        else:
            valid = [r for r in (.5, 1 / 3, .25) if abs(r * P - round(r * P)) < 1e-9 and round(r * P) >= 4]
            ratio, D = valid[(j // 2) % len(valid)], P
        if D > W - 4:
            return None
        probe, S, c_off, sigma = imp_probe(rng, bf, P)
        L, gs, s = imp_layout(rng, P, W, D)
        period = int(round(P * ratio))
        rate = np.ones(L)
        rate[s:s + D] = P / period
        base, usual, centre, mean = imp_render(probe, L, c_off, rate)
        ev = {"s": s, "D": D, "period": period, "ratio": ratio, "direction": "slower" if slow else "faster"}
        head = f"From step {s} through step {s + D - 1}, the pattern repeated "
        tail = "; its shape, amplitude and level were unchanged, and it was not reset at either end."
        contexts = {1: head + f"every {period} steps instead of at its usual interval" + tail,
                    2: head + f"at {word(PERIOD_RATIO, ratio)}" + tail + " " + IMP_DEF["interval"]}
        return dict(P=P, S=S, sigma=sigma, L=L, gs=gs, s=s, D=D, base=base, usual=usual, centre=centre, mean=mean,
                    event=ev, contexts=contexts, censor=None)
    if family == "amplitude":
        anchor = "centre" if bf in SMOOTH else "lower_envelope"
        hi = min(24, W - 4)
        P = 18 if bf == "harmonic" else int(rng.integers(12, hi + 1))
        probe = bases.sample_imp_base(bf, P, IMP_L[1], rng)
        if anchor == "centre" and not align_centre_sample(probe, rng):
            return None
        S0, _ = bases.observed_stats(probe, probe["R"])
        sigma = float(rng.uniform(.02, .04) * S0)
        grid = 10.0 ** -util.decimals_for(sigma)
        unit = 2 * grid                                    # factor x usual range lands on the display grid
        probe["R"] *= (unit * max(1, round(S0 / unit))) / S0
        S, c_off = bases.observed_stats(probe, probe["R"])
        D = P
        lead, tail = int(np.ceil(1.5 * P)) + int(rng.integers(2, 7)), P + int(rng.integers(2, 7))
        L = int(rng.integers(max(lead + W + tail, IMP_L[0], 4 * W), IMP_L[1] + 1))
        gs = int(rng.integers(lead, L - tail - W + 1))
        base, usual, centre, _ = imp_render(probe, L, c_off)
        origin = centre if anchor == "centre" else centre - S / 2
        starts = [t for t in range(gs + 2, gs + W - D - 1) if abs(usual[t] - origin[t]) <= 1e-7 * S]
        if not starts:
            return None
        s = starts[int(rng.integers(len(starts)))]
        factor = (.5, 1.5)[j % 2]
        # Anchor the baseline to two decimals (v15.1 prepare_imp): level at s and slope per step are exact.
        slope, level = round(float(base["slope"]), 2), round(float(origin[s]), 2)
        offset = c_off - (S / 2 if anchor != "centre" else 0)
        base["slope"] = slope
        base["b0"] = level - offset - slope * (s - (L - 1) / 2)
        usual, centre = bases.render_imp(base)
        centre = centre + c_off
        origin = centre if anchor == "centre" else centre - S / 2
        if not np.allclose(usual[[s, s + D]], origin[[s, s + D]], rtol=0, atol=1e-7 * S):
            return None
        mean = usual.copy()
        mean[s:s + D] = origin[s:s + D] + factor * (usual[s:s + D] - origin[s:s + D])
        ev = {"s": s, "D": D, "factor": factor, "anchor": anchor, "level": level, "slope": slope,
              "direction": "reduced" if factor < 1 else "increased"}
        head = f"From step {s} through step {s + D - 1}, "
        if slope == 0:
            rule = f"scale the underlying pattern's deviations from the fixed level {level:.2f} units by a factor of {number(factor)}. "
        else:
            rule = (f"scale the underlying pattern's deviations from a baseline by a factor of {number(factor)}. "
                    f"The baseline is {level:.2f} units at step {s} and {'increases' if slope > 0 else 'decreases'} by "
                    f"{abs(slope):.2f} units per step. ")
        amount = word(AMP_RANGE, factor)
        if anchor == "centre":
            rel = (head + "the pattern's deviations from its centre line were scaled by one common factor, so that over that interval "
                   f"the distance between its upper and lower envelopes became {amount}; its centre line, shape and timing were unchanged.")
        else:
            rel = (head + "the pattern's heights above its lower envelope were scaled by one common factor, so that over that interval "
                   f"the distance between its upper and lower envelopes became {amount}; its lower envelope, shape and timing were unchanged.")
        contexts = {1: head + rule + "Keep each value on the same side of this baseline. The pattern's timing and background noise were unchanged.",
                    2: rel + " " + IMP_AMP_DEF}
        return dict(P=P, S=S, sigma=sigma, L=L, gs=gs, s=s, D=D, base=base, usual=usual, centre=centre, mean=mean,
                    event=ev, contexts=contexts, censor=None)
    # shift / flat / bounding
    P = int(rng.integers(12, 29))
    if bf == "harmonic":
        P = max(P, 18)
    probe, S, c_off, sigma = imp_probe(rng, bf, P)
    d = util.decimals_for(sigma)
    grid = 10.0 ** -d
    D = {"shift": lambda: int(rng.integers(10, 17)), "flat": lambda: int(rng.integers(8, 17)),
         "bounding": lambda: int(rng.integers(max(8, int(.6 * P)), min(P, 20) + 1))}[family]()
    if D > W - 4:
        return None
    L, gs, s = imp_layout(rng, P, W, D)
    base, usual, centre, mean = imp_render(probe, L, c_off)
    head = f"From step {s} through step {s + D - 1}, "
    censor = None
    if family == "shift":
        frac = (.25, .5, .75)[(j // 2) % 3]
        amount = q(frac * S, grid)
        mean[s:s + D] += sign * amount
        ev = {"s": s, "D": D, "fraction": frac, "amount": amount, "sign": sign, "direction": "up" if sign > 0 else "down"}
        shifted = "upward" if sign > 0 else "downward"
        contexts = {1: head + f"the whole pattern was shifted {shifted} by {amount:.{d}f} units; its shape and timing were unchanged.",
                    2: head + f"the whole pattern was shifted {shifted} by {word(RANGE_FRAC, frac)}; its shape and timing were unchanged. "
                    + IMP_DEF["range"]}
    elif family == "flat":
        frac = (.25, .5, 1.)[(j // 2) % 3]
        level = round(float(centre[s] + sign * frac * S / 2), 2)
        mean[s:s + D] = level
        ev = {"s": s, "D": D, "fraction": frac, "level": level, "sign": sign, "direction": "up" if sign > 0 else "down"}
        contexts = {1: head + f"the underlying signal was held at the constant value {level:.2f} instead of following its usual pattern. Noise continued as usual.",
                    2: head + "the signal was held at a constant level instead of following its usual pattern; noise continued as usual. "
                    f"The level was {word(LEVEL_FRAC, sign * frac)} as of step {s}. " + IMP_DEF["centre"]}
    else:
        side = ("upper", "lower")[j % 2]
        frac = (.25, .5, -.25, -.5)[((j // 2) + attempt // 400) % 4]
        upper = side == "upper"
        bound = round(float(centre[s] + (1 if upper else -1) * frac * S / 2), 2)
        censor = {"s": s, "e": s + D, "side": side, "level": bound}
        ev = {"s": s, "D": D, "fraction": frac, "bound": bound, "side": side, "direction": side}
        cap, beyond = ("cap", "above") if upper else ("floor", "below")
        tail = f" and stayed at that fixed value throughout the interval. Readings {beyond} it were recorded as the {cap}; other readings were unaffected."
        contexts = {1: head + f"a constant {cap} was in effect. The {cap} was at {bound:.2f} units" + tail,
                    2: head + f"a constant {cap} was in effect. The {cap} was {word(LEVEL_FRAC, frac if upper else -frac)} as of step {s}"
                    + tail + " " + IMP_DEF["centre"]}
    return dict(P=P, S=S, sigma=sigma, L=L, gs=gs, s=s, D=D, base=base, usual=usual, centre=centre, mean=mean,
                event=ev, contexts=contexts, censor=censor)


def imp_self(j, rng, W):
    """Quiet level with irregular bursts; the interval inside the missing block repeats the first or second burst.
    The referenced burst has two peaks in every other pair of scenes; the other of first / second has the same width."""
    which = ("first", "second")[j % 2]
    ref_kind = "double" if (j // 2) % 2 == 1 else "single"
    R, quiet = float(10 ** rng.uniform(1, 2)), float(rng.uniform(-50, 150))
    lo, hi = {24: (8, 15), 32: (10, 19), 48: (12, 25)}[W]
    if ref_kind == "double":
        lo = max(lo, DOUBLE_MIN_W)
    w_ref = int(rng.integers(lo, hi))
    k_ref = 0 if which == "first" else 1
    bursts, t = [], int(rng.integers(4, 17))
    for k in range(2 + int(rng.integers(0, 3))):
        w = w_ref if k < 2 else int(rng.integers(8, 19))
        unit, kind = burst_shape(w, rng, ref_kind if k == k_ref else None)
        bursts.append({"start": t, "shape": unit * rng.uniform(.45, 1.0) * R, "kind": kind})
        t += w + int(rng.integers(8, 29))
    gs = t + int(rng.integers(0, 12))
    ge = gs + W
    r0 = gs + int(rng.integers(2, W - w_ref - 1))
    t = ge + int(rng.integers(8, 21))
    for k in range(int(rng.integers(0, 3))):
        w = int(rng.integers(8, 19))
        unit, kind = burst_shape(w, rng)
        bursts.append({"start": t, "shape": unit * rng.uniform(.45, 1.0) * R, "kind": kind})
        t += w + int(rng.integers(8, 29))
    while t + 4 < IMP_L[0]:
        w = int(rng.integers(8, 19))
        unit, kind = burst_shape(w, rng)
        bursts.append({"start": t, "shape": unit * rng.uniform(.45, 1.0) * R, "kind": kind})
        t += w + int(rng.integers(8, 29))
    L = t + int(rng.integers(4, 17))
    if L > IMP_L[1]:
        return None
    starts = [b["start"] for b in bursts]
    gaps_between = np.diff(starts)
    if gaps_between.max() - gaps_between.min() < 6:                      # no near-regular spacing
        return None
    usual = np.full(L, quiet)
    for b in bursts:
        usual[b["start"]:b["start"] + len(b["shape"])] += b["shape"]
    src = bursts[k_ref]
    a0 = src["start"]
    mean = usual.copy()
    mean[r0:r0 + w_ref] = quiet + src["shape"]
    S = float(np.ptp(usual))
    sigma = float(rng.uniform(.02, .04) * S)
    # Copying any other burst instead (the same number of steps from its start) must be clearly worse: build.py
    # scores these alternatives against the truth and rejects the scene otherwise.
    alternatives = []
    for b in bursts:
        b0 = b["start"]
        if b is src or not (b0 + w_ref <= gs or b0 >= ge) or b0 + w_ref > L:
            continue
        alt = mean.copy()
        alt[r0:r0 + w_ref] = usual[b0:b0 + w_ref]
        alternatives.append(alt)
    r1 = r0 + w_ref - 1
    ev = {"s": r0, "D": w_ref, "source": [a0, a0 + w_ref - 1], "which": which, "source_kind": src["kind"],
          "bursts": [[b["start"], len(b["shape"]), b["kind"]] for b in bursts], "quiet": quiet}
    tail = "; noise continued as usual."
    contexts = {1: f"From step {r0} through step {r1}, the series repeated the pattern it showed from step {a0} through step {a0 + w_ref - 1}" + tail,
                2: f"From step {r0} through step {r1}, the series repeated the pattern of its {which} burst" + tail}
    return dict(P=None, S=S, sigma=sigma, L=L, gs=gs, s=r0, D=w_ref, base={"quiet": quiet, "R": R}, usual=usual,
                centre=None, mean=mean, event=ev, contexts=contexts, censor=None, alternatives=alternatives)


def imp_delay(j, rng, W):
    """Two series on an irregular wave. Y follows X with delay d0; inside the interval the delay is d0 / 2 or 2 d0."""
    longer = j % 2 == 1
    d0 = int(rng.choice([6, 8, 10, 12, 14, 16]))
    d = 2 * d0 if longer else d0 // 2
    L = int(rng.integers(IMP_L[0], IMP_L[1] + 1))
    off = 2 * max(d0, d) + 16
    R, level = float(10 ** rng.uniform(1, 2)), float(rng.uniform(-50, 150))
    unit, scale = smooth_wave(L + off, rng)
    xw = level + R * unit

    def X(t):
        return xw[np.asarray(t) + off]
    t_all = np.arange(L)
    usual = X(t_all - d0)
    lead = max(40, 3 * d0 + 10)
    gs = int(rng.integers(lead, L - 16 - W + 1))
    ge = gs + W
    limit = float(np.percentile(np.abs(np.diff(xw, 2)), 90))
    observed = np.ones(L, bool)
    observed[gs:ge] = False
    S = float(np.ptp(usual[observed]))
    lo = max(8, int(np.ceil(.25 * W)))
    cands = []
    for r0 in range(gs + 2, ge - 2 - lo + 1):
        for length in range(lo, min(W - 4, ge - 2 - r0) + 1):
            r1 = r0 + length - 1
            delay = np.full(L, d0)
            delay[r0:r1 + 1] = d
            y = X(t_all - delay)
            if not curvature_ok(y, (r0, r1 + 1), limit):
                continue
            if np.mean(np.abs(y[r0:r1 + 1] - usual[r0:r1 + 1])) < DELAY_EFFECT * S:
                continue
            cands.append((r0, r1))
    if not cands:
        return None
    r0, r1 = cands[int(rng.integers(len(cands)))]
    joints = (r0, r1 + 1)
    delay = np.full(L, d0)
    delay[r0:r1 + 1] = d
    mean = X(t_all - delay)
    sigma = float(rng.uniform(.02, .04) * S)
    x_obs = X(t_all) + rng.normal(0, sigma, L)
    ev = {"s": r0, "D": r1 - r0 + 1, "d0": d0, "d": d, "direction": "longer" if longer else "shorter",
          "source": [r0 - d, r1 - d], "wave_scale": scale,
          "joint_curvature": max(abs(mean[t + 1] - 2 * mean[t] + mean[t - 1]) for j_ in joints for t in (j_ - 1, j_)),
          "curvature_limit": limit}
    tail = "; noise continued as usual."
    contexts = {1: DELAY_HEADER + f"\nFrom step {r0} through step {r1}, the delay was {d} steps: series Y repeated what series X did "
                f"from step {r0 - d} through step {r1 - d}" + tail,
                2: DELAY_HEADER + f"\nFrom step {r0} through step {r1}, the delay was {'twice' if longer else 'half'} its usual value" + tail}
    return dict(P=None, S=S, sigma=sigma, L=L, gs=gs, s=r0, D=r1 - r0 + 1, base={"level": level, "R": R, "wave": xw.tolist(), "offset": off},
                usual=usual, centre=None, mean=mean, event=ev, contexts=contexts, censor=None, x_observed=x_obs)


# ------------------------------------------------------------------------------------------------ FC scenes
def fc_periodic(family, j, rng, bf, T, H, attempt=0):
    """roi_trend_v3 mechanics with a single condition; the change runs from step s to the end of the forecast."""
    L = T + H
    sign = 1 if j % 2 == 0 else -1
    p = int(rng.integers(6, H // 2 + 1))
    s = T + p
    if family == "frequency":
        slow = j % 2 == 1
        P = 20 if bf == "harmonic" else int(rng.choice([12, 16] if slow else [12, 16, 20]))
        if slow:
            ratio = 2.0
        else:
            valid = [r for r in (.5, 1 / 3, .25) if abs(P * r - round(P * r)) < 1e-8 and P * r >= 4]
            ratio = valid[(j // 2) % len(valid)]
    elif family == "amplitude":
        P = 18 if bf == "harmonic" else int(rng.integers(12, min(24, H - 4) + 1))
    elif family == "trend_rate":
        P = int(rng.choice([12, 16, 20]))
    else:
        P = int(rng.integers(18 if bf == "harmonic" else 12, 29))
    base = bases.sample_imp_base(bf, P, L, rng)
    if family == "amplitude" and bf in SMOOTH and not align_centre_sample(base, rng):
        return None
    S, off = bases.observed_stats(base, base["R"])
    if family != "trend_rate":
        grid = .08 if family in ("flat", "bounding") else .04 if family == "shift" else .02
        base["R"] *= max(grid, round(S / grid) * grid) / S                 # quarter fractions exact to two decimals
        S, off = bases.observed_stats(base, base["R"])
    usual, raw_centre = bases.render_imp(base)
    centre = raw_centre + off
    censor = None
    if family == "trend_rate":
        slope_sign = (1, -1)[j % 2]
        q0 = round(slope_sign * round(float(rng.uniform(.20, .40) * S / P) / .02) * .02, 2)
        if q0 == 0 or not .20 <= abs(q0) * P / S <= .40:
            return None
        base["slope"] = q0
        usual, raw_centre = bases.render_imp(base)
        centre = raw_centre + off
    if family == "amplitude":
        anchor = "centre" if bf in SMOOTH else "lower_envelope"
        origin = centre if anchor == "centre" else centre - S / 2
        starts = [t for t in range(T + 6, T + H // 2 + 1) if abs(usual[t] - origin[t]) <= 1e-7 * S]
        if not starts:
            return None
        s = int(starts[int(rng.integers(len(starts)))])
        slope, level = round(base["slope"], 2), round(float(origin[s]), 2)
        base["slope"] = slope
        base["b0"] = level - (off if anchor == "centre" else off - S / 2) - slope * (s - (L - 1) / 2)
        usual, raw_centre = bases.render_imp(base)
        centre = raw_centre + off
        origin = centre if anchor == "centre" else centre - S / 2
    elif family in ("flat", "bounding"):
        base["b0"] += round(float(centre[s]), 2) - float(centre[s])        # centre at step s exact to two decimals
        usual, raw_centre = bases.render_imp(base)
        centre = raw_centre + off
    sigma = float(rng.uniform(.02, .04) * S)
    mean = usual.copy()
    head = f"From step {s} onward, "
    if family == "frequency":
        period = int(round(P * ratio))
        rate = np.ones(L)
        rate[s:] = P / period
        mean = bases.render_imp(base, rate)[0]
        ev = {"s": s, "period": period, "ratio": ratio, "direction": "slower" if ratio > 1 else "faster"}
        tail = "; its shape, amplitude and level will be unchanged. Its phase will continue without restarting the pattern."
        contexts = {1: head + f"the pattern will repeat every {period} steps" + tail,
                    2: head + f"the pattern will repeat at {word(FC_RATIO, ratio)}" + tail + " " + FC_INTERVAL}
    elif family == "shift":
        frac = (.25, .5, .75)[(j // 2) % 3]
        amount = round(sign * frac * S, 2)
        mean[s:] += amount
        ev = {"s": s, "fraction": frac, "amount": amount, "sign": sign, "direction": "up" if sign > 0 else "down"}
        shifted = "upward" if sign > 0 else "downward"
        contexts = {1: head + f"the whole pattern will be shifted {shifted} by {abs(amount):.2f} units; its shape and timing will be unchanged.",
                    2: head + f"the whole pattern will be shifted {shifted} by {word(RANGE_FRAC, frac)}; its shape and timing will be unchanged. " + FC_RANGE}
    elif family == "flat":
        frac = (.25, .5, 1.)[(j // 2) % 3]
        level = round(float(centre[s] + sign * frac * S / 2), 2)
        mean[s:] = level
        ev = {"s": s, "fraction": frac, "level": level, "sign": sign, "direction": "up" if sign > 0 else "down"}
        contexts = {1: head + f"the underlying signal will be held at the constant value {level:.2f} instead of following its usual pattern. Noise will continue as usual.",
                    2: head + "the underlying signal will be held at a constant level instead of following its usual pattern; noise will continue as usual. "
                    f"The level will be {fc_relative_position(sign, frac, s)} and will stay fixed until the end of the forecast. " + FC_ENVELOPES}
    elif family == "bounding":
        frac = (-.25, .25, .5)[((j // 2) + attempt // 400) % 3]
        bound = round(float(centre[s] + sign * frac * S / 2), 2)
        censor = {"s": s, "e": L, "side": "upper" if sign > 0 else "lower", "level": bound}
        ev = {"s": s, "fraction": frac, "bound": bound, "sign": sign, "direction": censor["side"]}
        cap, beyond = ("cap", "above") if sign > 0 else ("floor", "below")
        tail = f" and will stay at that fixed value until the end of the forecast. Readings {beyond} it will be recorded as the {cap}; other readings will be unaffected."
        contexts = {1: head + f"a constant {cap} will be in effect. The {cap} will be at {bound:.2f} units" + tail,
                    2: head + f"a constant {cap} will be in effect. The {cap} will be {fc_relative_position(sign, frac, s)}" + tail + " " + FC_ENVELOPES}
    elif family == "amplitude":
        factor = (.5, 1.5)[j % 2]
        mean[s:] = origin[s:] + factor * (usual[s:] - origin[s:])
        ev = {"s": s, "factor": factor, "anchor": anchor, "level": level, "slope": slope,
              "direction": "reduced" if factor < 1 else "increased"}
        if slope == 0:
            where, drift = f"the fixed level {level:.2f} units", ""
        else:
            where = "a baseline"
            drift = (f" The baseline will be {level:.2f} units at step {s} and will {'increase' if slope > 0 else 'decrease'} "
                     f"by {abs(slope):.2f} units per step.")
        name = "centre line" if anchor == "centre" else "lower envelope"
        action = "halved" if factor == .5 else "increased to one and a half times their usual size"
        contexts = {1: head + f"the underlying pattern's deviations from {where} will be multiplied by {factor:g}." + drift
                    + " Each value will remain on the same side of this baseline. The pattern's timing and background noise will remain unchanged.",
                    2: head + f"the underlying pattern's deviations from its {name} will be {action}. Its {name} and timing will remain unchanged, "
                    "and background noise will continue as usual. " + FC_ENVELOPES}
    else:  # trend_rate
        factor = (.5, 2., -1.)[(j // 2) % 3]
        new = round(q0 * factor, 2)
        mean += (new - q0) * np.clip(np.arange(L) - s + 1, 0, None)
        ev = {"s": s, "usual_slope": q0, "slope": new, "factor": factor,
              "direction": "opposite" if factor < 0 else ("steeper" if factor > 1 else "flatter")}
        if factor == -1:
            action = f"move in the opposite direction at the same rate magnitude as its linear trend over observed steps 0 through {T - 1}"
        else:
            action = f"continue in the same direction at {'twice' if factor == 2 else 'half'} the rate of its linear trend over observed steps 0 through {T - 1}"
        tail = (f". This rate applies starting with the transition from step {s - 1} to step {s}. "
                "The repeating fluctuations keep their usual shape, size and timing. "
                "The trend is the linear movement of the underlying centre line, separate from periodic fluctuations and background noise.")
        contexts = {1: head + f"the underlying centre line will {'increase' if new > 0 else 'decrease'} by {abs(new):.2f} units per step" + tail,
                    2: head + "the underlying centre line will " + action + tail}
        if abs(q0) * P < 4 * sigma:
            return None
    return dict(P=P, S=S, sigma=sigma, L=L, s=s, base=base, usual=usual, centre=centre, mean=mean, event=ev,
                contexts=contexts, censor=censor)


def fc_self(j, rng, T, H):
    """History with irregular bursts; from step s the series replays its course from the start of an earlier burst.
    The referenced burst has two peaks in every other pair of scenes."""
    which = ("first", "second")[j % 2]
    ref_kind = "double" if (j // 2) % 2 == 1 else "single"
    k_ref = 0 if which == "first" else 1
    L = T + H
    p = int(rng.integers(6, H // 2 + 1))
    s = T + p
    n = L - s
    R, quiet = float(10 ** rng.uniform(1, 2)), float(rng.uniform(-50, 150))
    bursts, t = [], int(rng.integers(4, 17))
    while True:
        w = int(rng.integers(DOUBLE_MIN_W if (len(bursts) == k_ref and ref_kind == "double") else 8, 19))
        if t + w + 6 > T:
            break
        unit, kind = burst_shape(w, rng, ref_kind if len(bursts) == k_ref else None)
        bursts.append({"start": t, "shape": unit * rng.uniform(.45, 1.0) * R, "kind": kind})
        t += w + int(rng.integers(8, 29))
    if len(bursts) < 4:
        return None
    gaps_between = np.diff([b["start"] for b in bursts])
    if gaps_between.max() - gaps_between.min() < 6:
        return None
    usual = np.full(L, quiet)
    for b in bursts:
        usual[b["start"]:b["start"] + len(b["shape"])] += b["shape"]
    src = bursts[k_ref]
    a = src["start"]
    if a + n > T:
        return None
    S = float(np.ptp(usual[:T]))
    mean = usual.copy()
    mean[s:] = usual[a:a + n]
    sigma = float(rng.uniform(.02, .04) * S)
    alternatives = []
    for b in bursts:
        if b is src or b["start"] + n > T:
            continue
        alt = mean.copy()
        alt[s:] = usual[b["start"]:b["start"] + n]
        alternatives.append(alt)
    ev = {"s": s, "source_start": a, "which": which, "source_kind": src["kind"],
          "bursts": [[b["start"], len(b["shape"]), b["kind"]] for b in bursts], "quiet": quiet}
    tail = "; noise will continue as usual."
    contexts = {1: f"From step {s} onward, the series will repeat the pattern it showed from step {a} onward" + tail,
                2: f"From step {s} onward, the series will repeat the pattern it showed from the start of its {which} burst onward" + tail}
    return dict(P=None, S=S, sigma=sigma, L=L, s=s, base={"quiet": quiet, "R": R}, usual=usual, centre=None, mean=mean,
                event=ev, contexts=contexts, censor=None, alternatives=alternatives)


def fc_delay(j, rng, T, H):
    """Y follows X with delay d0 over the history; from step s the delay is d0 / 2 or 2 d0 to the end.
    X is observed from step H through the last forecast step, so the answer is in X for either direction."""
    longer = j % 2 == 1
    L = T + H
    d0 = int(rng.choice([6, 8, 10, 12, 14, 16]))
    d = 2 * d0 if longer else d0 // 2
    off = 2 * max(d0, d) + 16
    R, level = float(10 ** rng.uniform(1, 2)), float(rng.uniform(-50, 150))
    unit, scale = smooth_wave(L + off, rng)
    xw = level + R * unit

    def X(t):
        return xw[np.asarray(t) + off]
    t_all = np.arange(L)
    usual = X(t_all - d0)
    S = float(np.ptp(usual[:T]))
    limit = float(np.percentile(np.abs(np.diff(xw, 2)), 90))
    cands = []
    for p in range(6, H // 2 + 1):
        s = T + p
        y = X(t_all - np.where(t_all >= s, d, d0))
        if curvature_ok(y, (s,), limit) and np.mean(np.abs(y[s:] - usual[s:])) >= DELAY_EFFECT * S:
            cands.append(s)
    if not cands:
        return None
    s = cands[int(rng.integers(len(cands)))]
    mean = X(t_all - np.where(t_all >= s, d, d0))
    sigma = float(rng.uniform(.02, .04) * S)
    tx = np.arange(H, L)
    x_obs = X(tx) + rng.normal(0, sigma, len(tx))
    ev = {"s": s, "d0": d0, "d": d, "direction": "longer" if longer else "shorter", "source_start": s - d,
          "x_first": int(tx[0]), "wave_scale": scale,
          "joint_curvature": max(abs(mean[t + 1] - 2 * mean[t] + mean[t - 1]) for t in (s - 1, s)),
          "curvature_limit": limit}
    tail = "; noise will continue as usual."
    contexts = {1: DELAY_HEADER + f"\nFrom step {s} onward, the delay will be {d} steps: series Y will repeat what series X did from step {s - d} onward" + tail,
                2: DELAY_HEADER + f"\nFrom step {s} onward, the delay will be {'twice' if longer else 'half'} its usual value" + tail}
    return dict(P=None, S=S, sigma=sigma, L=L, s=s, base={"level": level, "R": R, "wave": xw.tolist(), "offset": off},
                usual=usual, centre=None, mean=mean, event=ev, contexts=contexts, censor=None, x_observed=x_obs, x_index=tx)


# ------------------------------------------------------------------------------------------------ prompts
def series_text(index, values, d):
    cells = [f"{i}:NaN" if v is None else f"{i}:{v:.{d}f}" for i, v in zip(index, values)]
    return "\n".join(", ".join(cells[k:k + 16]) for k in range(0, len(cells), 16))


def answer_line(n, d, of_y):
    return f"Respond ONLY with the {n} values{' of series Y' if of_y else ''}, one per line in (index, value) format with {places(d)}. Do not explain."


def imp_prompt(row):
    inp, family = row["input"], row["requirement"]
    gs, ge = inp["gap"]
    n, d = ge - gs, inp["decimals"]
    y_text = series_text(range(len(inp["values"])), inp["values"], d)
    if family == "delay":
        ref = inp["reference"]
        x_text = series_text(range(ref["start"], ref["start"] + len(ref["values"])), ref["values"], d)
        series = ("Here are two time series in index:value format. Series X is complete; series Y has missing steps marked NaN:\n"
                  f'<series name="X">\n{x_text}\n</series>\n<series name="Y">\n{y_text}\n</series>')
        rules = (f"Steps {gs}-{ge - 1} of series Y ({n} values) are missing.\n"
                 "Infer the usual delay and the noise strength from the observations.\n"
                 "Outside the interval described in the context, the delay has its usual value.\n")
    else:
        series = f"Here is the time series in index:value format, with the missing steps marked NaN:\n<series>\n{y_text}\n</series>"
        if family == "self_repetition":
            rules = (f"Steps {gs}-{ge - 1} ({n} values) are missing.\n"
                     "Infer the quiet level, the bursts and the noise strength from the observations.\n"
                     "The series stays at a quiet level apart from occasional short bursts; each burst leaves the quiet level and returns to it, with one or two peaks.\n"
                     "Outside the interval described in the context, the missing steps stay at the quiet level.\n")
        else:
            rules = (f"Steps {gs}-{ge - 1} ({n} values) are missing.\n"
                     "Infer the usual signal and the noise strength from the observations.\n"
                     "The usual signal is the repeating pattern without the change described in the context.\n"
                     "Outside the interval described in the context, continue the usual signal and noise without restarting the pattern.\n")
    user = ("I have a time series imputation task for you.\n\n"
            "Here is some context about what happens during the missing interval.\n"
            f"<context>\n{inp['context']}\n</context>\n\n{series}\n\n{rules}"
            "Noise is independent, zero-mean Gaussian with unchanged strength.\n"
            "Intervals include both endpoints.\n"
            "Return one plausible realization including noise, rather than a noise-free average.\n\n"
            + answer_line(n, d, family == "delay"))
    return [{"role": "system", "content": SYSTEM["imp"]}, {"role": "user", "content": user}]


def fc_prompt(row):
    inp, family = row["input"], row["requirement"]
    T, H, d, s = inp["forecast_start"], inp["horizon"], inp["decimals"], row["private"]["event"]["s"]
    y_text = series_text(range(T), inp["values"], d)
    if family == "delay":
        ref = inp["reference"]
        last = ref["start"] + len(ref["values"]) - 1
        x_text = series_text(range(ref["start"], last + 1), ref["values"], d)
        series = (f"Here are the observations of two series in index:value format. Series X is observed from step {ref['start']} "
                  f"through step {last}; series Y is observed through step {T - 1}:\n"
                  f'<series name="X">\n{x_text}\n</series>\n<series name="Y">\n{y_text}\n</series>')
        rules = (f"Forecast series Y for steps {T}-{T + H - 1} ({H} values) using only the observations and context.\n"
                 "Infer the usual delay and the noise strength from the observations.\n"
                 f"Before step {s}, the delay has its usual value.\n")
    else:
        series = f"Here are the past observations in index:value format:\n<series>\n{y_text}\n</series>"
        if family == "self_repetition":
            rules = (f"Forecast steps {T}-{T + H - 1} ({H} values) using only the past observations and context.\n"
                     "Infer the quiet level, the bursts and the noise strength from the past observations.\n"
                     "The series stays at a quiet level apart from occasional short bursts; each burst leaves the quiet level and returns to it, with one or two peaks.\n"
                     f"Before step {s}, the series stays at its quiet level.\n")
        else:
            rules = (f"Forecast steps {T}-{T + H - 1} ({H} values) using only the past observations and context.\n"
                     "Infer the usual signal and noise strength from the past observations.\n"
                     "The usual signal is the repeating pattern without the change described in the context, including any gradual drift.\n"
                     f"Before step {s}, continue the usual signal and noise without restarting the pattern.\n")
    user = ("I have a time series forecasting task for you.\n\n"
            "Here is some context about what will happen during the forecast interval.\n"
            f"<context>\n{inp['context']}\n</context>\n\n{series}\n\n{rules}"
            "Noise is independent, zero-mean Gaussian with unchanged strength.\n"
            "Intervals include both endpoints.\n"
            "Return one plausible realization including noise, rather than a noise-free average.\n\n"
            + answer_line(H, d, family == "delay"))
    return [{"role": "system", "content": SYSTEM["fc"]}, {"role": "user", "content": user}]


def request(task, row):
    return {"id": row["id"], "task": task, "dataset_version": VERSION[task], "format": "global",
            "max_tokens": 2048, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": imp_prompt(row) if task == "imp" else fc_prompt(row)}


# ------------------------------------------------------------------------------------------------ scenes -> rows
def realize(mean, noise, censor):
    y = mean + noise
    if censor:
        s, e = censor["s"], censor["e"]
        y[s:e] = (np.minimum if censor["side"] == "upper" else np.maximum)(y[s:e], censor["level"])
    return y


def effect_ok(sc, noise, lo, hi):
    """The change must move the interval away from the usual signal (after censoring for bounding)."""
    sigma = sc["sigma"]
    if sc["censor"]:
        clipped = realize(sc["mean"], noise, sc["censor"])[lo:hi]
        free = (sc["usual"] + noise)[lo:hi]
        n_clip = int(np.sum(np.abs(clipped - free) > 1e-12))
        sc["quality"] = {"clipped": n_clip, "unclipped": (hi - lo) - n_clip,
                         "effect_sigma": float(np.mean(np.abs(clipped - free)) / sigma)}
        return n_clip >= 4 and (hi - lo) - n_clip >= 3 and sc["quality"]["effect_sigma"] >= MIN_EFFECT_SIGMA
    sc["quality"] = {"effect_sigma": float(np.mean(np.abs(sc["mean"][lo:hi] - sc["usual"][lo:hi])) / sigma)}
    return sc["quality"]["effect_sigma"] >= MIN_EFFECT_SIGMA


def draw_scores(task, sc, noise, rng, means):
    """RCRPS of 25 draws around each mean (the first is the oracle and keeps the censoring), with the package's
    scoring: fair CRPS, the interval and the remaining target steps weighted 1/2 each, 1 / observed range."""
    y = realize(sc["mean"], noise, sc["censor"])
    if task == "imp":
        lo, hi = sc["gs"], sc["gs"] + sc["slot"]["gap"]
        observed = np.r_[y[:lo], y[hi:]]
    else:
        lo, hi = sc["slot"]["history_length"], sc["L"]
        observed = y[:lo]
    rho = 1.0 / float(np.ptp(observed))
    roi = (np.arange(lo, hi) >= sc["s"]) & (np.arange(lo, hi) < (sc["s"] + sc["D"] if task == "imp" else hi))
    truth = y[lo:hi]
    n = 25
    w = 2 * np.arange(1, n + 1) - n - 1
    out = []
    for i, mean in enumerate(means):
        x = mean[None, lo:hi] + rng.normal(0, sc["sigma"], (n, hi - lo))
        censor = sc["censor"] if i == 0 else None
        if censor:
            a, b = censor["s"] - lo, censor["e"] - lo
            x[:, a:b] = (np.minimum if censor["side"] == "upper" else np.maximum)(x[:, a:b], censor["level"])
        crps = np.abs(x - truth).mean(axis=0) - (np.sort(x, axis=0) * w[:, None]).sum(axis=0) / (n * (n - 1))
        out.append(float(rho * (crps[roi].mean() + crps[~roi].mean()) / 2))
    return out


def clearly_worse(score, oracle):
    """Margins over validate.py (0.005, 2x) against sampling noise."""
    return score - oracle >= .006 and score >= 2.2 * oracle


def schedule(task, family, j):
    """Balanced layout slots from the scene index: gap (IMP) or history / horizon (FC), and the base family."""
    fi = list(QUOTAS[task]).index(family)
    if family == "trend_rate":
        bf = [b for b in BASES if b != "trend_sine"][(j + fi) % 6]
    elif family in ("self_repetition", "delay"):
        bf = "burst" if family == "self_repetition" else "wave"
    else:
        bf = BASES[(j + fi) % len(BASES)]
    if task == "imp":
        return {"base_family": bf, "gap": imp_frequency_gap(j, bf) if family == "frequency" else GAPS[j % 3]}
    # direction / first-second / shorter-longer follow j % 2, so history and horizon use other index digits
    return {"base_family": bf, "history_length": HISTORY[(j // 2) % 2], "horizon": HORIZONS[j % 3]}


def imp_frequency_gap(j, bf):
    """Faster scenes (even j) take most 24-step blocks; slower ones need 32 (P = 12) or 48; slow harmonic needs 48."""
    if j % 2 == 0:
        k = j // 2
        return 48 if k in (5, 11) else 32 if k in (3, 8, 13) else 24
    k = j // 2
    if bf == "harmonic":
        return 48
    return 32 if k % 2 == 0 else 48


def build_scene(task, family, j):
    sid = f"{task}_{family}_{j:03d}"
    slot = schedule(task, family, j)
    for attempt in range(MAX_ATTEMPTS):
        rng = rng_for(task, sid, attempt)
        if task == "imp":
            W = slot["gap"]
            if family == "self_repetition":
                sc = imp_self(j, rng, W)
            elif family == "delay":
                sc = imp_delay(j, rng, W)
            else:
                sc = imp_periodic(family, j, rng, slot["base_family"], W, attempt)
        else:
            T, H = slot["history_length"], slot["horizon"]
            if family == "self_repetition":
                sc = fc_self(j, rng, T, H)
            elif family == "delay":
                sc = fc_delay(j, rng, T, H)
            else:
                sc = fc_periodic(family, j, rng, slot["base_family"], T, H, attempt)
        if sc is None:
            continue
        noise = rng.normal(0, sc["sigma"], sc["L"])
        lo, hi = (sc["s"], sc["s"] + sc["D"]) if task == "imp" else (sc["s"], sc["L"])
        if not effect_ok(sc, noise, lo, hi):
            continue
        sc.update(noise=noise, attempt=attempt, slot=slot)
        # Ignoring the context, or (self-repetition) copying any other burst, must be clearly worse than the answer.
        scores = draw_scores(task, sc, noise, rng_for(task, sid, f"gain{attempt}"),
                             [sc["mean"], sc["usual"], *sc.get("alternatives", [])])
        oracle, ignore, alts = scores[0], scores[1], scores[2:]
        if not clearly_worse(ignore, oracle) or not all(clearly_worse(a, oracle) for a in alts):
            continue
        sc["quality"].update(context_gain=ignore - oracle, context_ratio=ignore / oracle,
                             wrong_copy_ratio=min(alts) / oracle if alts else None)
        return sid, sc
    raise RuntimeError(f"no layout for {sid} ({slot})")


def rows_for(task, family, j):
    sid, sc = build_scene(task, family, j)
    slot, ev, sigma = sc["slot"], sc["event"], sc["sigma"]
    d = util.decimals_for(sigma)
    y = realize(sc["mean"], sc["noise"], sc["censor"])
    usual_truth = sc["usual"] + sc["noise"]
    tags = {"task": task, "axis": axis_of(family), "requirement": family, "base_family": slot["base_family"],
            "usual_period": sc["P"], "direction": ev.get("direction") or ev.get("which"),
            "effect_sigma": round(sc["quality"]["effect_sigma"], 3), "generation_attempt": sc["attempt"]}
    if task == "imp":
        gs, ge = sc["gs"], sc["gs"] + slot["gap"]
        observed = np.ones(sc["L"], bool)
        observed[gs:ge] = False
        if not np.allclose(y[observed], usual_truth[observed]):
            raise RuntimeError(f"{sid}: the change leaks outside the missing block")
        values = [round(float(v), d) if o else None for v, o in zip(y, observed)]
        roi = np.zeros(ge - gs, bool)
        roi[sc["s"] - gs:sc["s"] - gs + sc["D"]] = True
        target = slice(gs, ge)
        inp = {"values": values, "gap": [gs, ge], "decimals": d}
        tags.update(gap_length=ge - gs, roi_start=sc["s"], roi_length=sc["D"], roi_share=round(float(roi.mean()), 3),
                    series_length=sc["L"])
    else:
        T, H = slot["history_length"], slot["horizon"]
        values = [round(float(v), d) for v in y[:T]]
        roi = np.arange(T, T + H) >= sc["s"]
        target = slice(T, T + H)
        inp = {"values": values, "forecast_start": T, "horizon": H, "decimals": d}
        tags.update(history_length=T, horizon=H, roi_start=sc["s"], roi_length=int(roi.sum()),
                    lead_steps=sc["s"] - T, roi_share=round(float(roi.mean()), 3))
    if family == "delay":
        index = sc.get("x_index", np.arange(sc["L"]))
        inp["reference"] = {"name": "X", "start": int(index[0]), "values": [round(float(v), d) for v in sc["x_observed"]]}
        tags.update(usual_delay=ev["d0"], new_delay=ev["d"])
    if family == "self_repetition":
        tags.update(source_kind=ev["source_kind"], wrong_copy_ratio=round(sc["quality"]["wrong_copy_ratio"], 2))
    rho = 1.0 / float(np.ptp([v for v in values if v is not None]))
    private = {"truth": y[target].tolist(), "gt_mean": sc["mean"][target].tolist(), "usual_mean": sc["usual"][target].tolist(),
               "roi_mask": roi.tolist(), "sigma": sigma, "normalization": rho, "event": ev,
               "censor": sc["censor"], "quality": sc["quality"], "usual_range": sc["S"],
               "base": {k: v for k, v in sc["base"].items() if k != "wave"}, "seed_attempt": sc["attempt"]}
    rows = []
    for level in (1, 2):
        row = {"id": f"{sid}_L{level}", "version": VERSION[task], "task": task, "scene_id": sid, "pair_id": sid,
               "level": level, "condition": None, "axis": axis_of(family), "requirement": family,
               "input": dict(inp, context=sc["contexts"][level]),
               "tags": dict(tags, level=level, information_mode="explicit" if level == 1 else "relative"),
               "private": private}
        rows.append(row)
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def build(out, tasks):
    summary = {"tasks": {}, "model_inference_performed": False}
    for task in tasks:
        rows = []
        for family, n in QUOTAS[task].items():
            for j in range(n):
                rows.extend(rows_for(task, family, j))
            print(f"{task}: {family} {n} scenes", flush=True)
        dest = out / task
        write_jsonl(dest / "records.jsonl", rows)
        write_jsonl(dest / "inputs.jsonl", [{"id": r["id"], **r["input"]} for r in rows])
        write_jsonl(dest / "prompts.jsonl", [request(task, r) for r in rows])
        write_jsonl(dest / "tags.jsonl", [{"id": r["id"], "scene_id": r["scene_id"], **r["tags"]} for r in rows])
        write_jsonl(dest / "scenes.jsonl", [{"scene_id": r["scene_id"], "task": task, "axis": r["axis"], "requirement": r["requirement"],
                                             "L1": r["id"], "L2": r["id"][:-1] + "2"} for r in rows if r["level"] == 1])
        scenes = [r for r in rows if r["level"] == 1]
        key = "gap_length" if task == "imp" else "horizon"
        info = {"version": VERSION[task], "scenes": len(scenes), "queries": len(rows),
                "scenes_by_requirement": dict(Counter(r["requirement"] for r in scenes)),
                "queries_by_level_axis": dict(Counter(f"L{r['level']}/{r['axis']}" for r in rows)),
                "directions": {f: dict(Counter(str(r["tags"]["direction"]) for r in scenes if r["requirement"] == f)) for f in QUOTAS[task]},
                f"{key}_by_requirement": {f: dict(Counter(r["tags"][key] for r in scenes if r["requirement"] == f)) for f in QUOTAS[task]},
                "base_families": {f: dict(Counter(r["tags"]["base_family"] for r in scenes if r["requirement"] == f)) for f in QUOTAS[task]}}
        if task == "fc":
            info["history_by_requirement"] = {f: dict(Counter(r["tags"]["history_length"] for r in scenes if r["requirement"] == f)) for f in QUOTAS[task]}
        (dest / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
        summary["tasks"][task] = info
    manifest = {"versions": VERSION, "quotas": QUOTAS,
                "data_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*.jsonl"))},
                "code_sha256": {p.name: sha(p) for p in (HERE / "build.py", HERE.parent / "_generator/bases.py", HERE.parent / "_generator/util.py")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "evalsets")
    ap.add_argument("--tasks", default="imp,fc")
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.tasks.split(",")), indent=1))
