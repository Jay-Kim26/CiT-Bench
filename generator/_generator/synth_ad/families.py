"""주기 패밀리 렌더러 (SPEC §5).

각 패밀리는 정규화된 주기 좌표 u ∈ [0,1) 위의 파형 함수로 정의된다.
period jitter 는 주기별 길이 P_k = P·(1+u_k), u_k ~ U[-j, +j] 로 구현하고
각 주기를 u ∈ [0,1) 로 리샘플해 이어붙인다 (jitter=0 금지, §5.2).
dual_seasonality / AM_beat 은 절대 시각 t 에 의존하는 envelope 을 추가로 곱한다.
"""
from __future__ import annotations

import numpy as np

from . import config


# ---------------------------------------------------------------- 파형 함수 u∈[0,1)
def _sine(u, p):
    return np.sin(2 * np.pi * u)


def _harmonic_sum(u, p):
    out = np.zeros_like(u)
    for k, (a, ph) in enumerate(zip(p["amps"], p["phases"]), start=1):
        out += a * np.sin(2 * np.pi * k * u + ph)
    return out


def _square(u, p):
    return np.where(u < p["duty"], 1.0, -1.0)


def _sawtooth(u, p):
    return 2.0 * u - 1.0


def _triangle(u, p):
    return 1.0 - 4.0 * np.abs(u - 0.5)


def _rectified_sine(u, p):
    return np.abs(np.sin(np.pi * u))


def _saturated_sine(u, p):
    return np.clip(p["gain"] * np.sin(2 * np.pi * u), -1.0, 1.0)


def _exp_rise_decay(u, p):
    r, tau = p["rise"], p["tau"]
    rising = u / r
    decay = np.exp(-(u - r) / tau)
    return np.where(u < r, rising, decay)


def _asym_gauss(u, u0, sl, sr, h):
    d = u - u0
    d = d - np.round(d)                      # 주기 경계 wrap
    s = np.where(d < 0, sl, sr)
    return h * np.exp(-0.5 * (d / s) ** 2)


def _pulse_train(u, p):
    return _asym_gauss(u, p["u0"], p["sig_l"], p["sig_r"], 1.0)


def _asym_bimodal(u, p):
    return (_asym_gauss(u, p["u1"], p["s1"], p["s1"], 1.0)
            + _asym_gauss(u, p["u2"], p["s2"], p["s2"], p["h2"]))


def _random_template(u, p):
    out = np.zeros_like(u)
    for k, (a, ph) in enumerate(zip(p["amps"], p["phases"]), start=1):
        out += a * np.sin(2 * np.pi * k * u + ph)
    return out                                # 주기 기저의 합 → 양끝점 일치 보장


WAVE_FN = {
    "pure_sine": _sine,
    "harmonic_sum": _harmonic_sum,
    "square": _square,
    "sawtooth": _sawtooth,
    "triangle": _triangle,
    "rectified_sine": _rectified_sine,
    "saturated_sine": _saturated_sine,
    "exp_rise_decay": _exp_rise_decay,
    "pulse_train": _pulse_train,
    "asym_bimodal": _asym_bimodal,
    "dual_seasonality": _sine,
    "AM_beat": _sine,
    "random_template_tiling": _random_template,
}


# ---------------------------------------------------------------- 형태 파라미터
def sample_shape_params(family: str, rng: np.random.Generator) -> dict:
    if family == "harmonic_sum":
        a1 = 1.0
        rest = rng.uniform(0.15, 0.85, size=3) * a1     # a1 >= max(a2..a4)
        return {"amps": np.concatenate([[a1], rest]),
                "phases": rng.uniform(0, 2 * np.pi, size=4)}
    if family == "square":
        return {"duty": 0.5}                            # 정준값 고정 (§10)
    if family == "saturated_sine":
        return {"gain": float(rng.uniform(1.2, 1.8))}
    if family == "exp_rise_decay":
        return {"rise": float(rng.uniform(0.05, 0.12)),
                "tau": float(rng.uniform(0.15, 0.30))}
    if family == "pulse_train":
        sig_l = float(rng.uniform(1 / 28, 1 / 21))      # sig_r = 1.7*sig_l <= P/12
        return {"u0": float(rng.uniform(0.25, 0.45)),
                "sig_l": sig_l, "sig_r": 1.7 * sig_l}
    if family == "asym_bimodal":
        return {"u1": float(rng.uniform(0.18, 0.26)), "s1": float(rng.uniform(0.04, 0.06)),
                "u2": float(rng.uniform(0.55, 0.65)), "s2": float(rng.uniform(0.08, 0.11)),
                "h2": float(rng.uniform(0.45, 0.70))}
    if family == "random_template_tiling":
        k = 8
        amps = rng.uniform(0.2, 1.0, size=k) / (np.arange(1, k + 1) ** 0.7)
        return {"amps": amps, "phases": rng.uniform(0, 2 * np.pi, size=k)}
    if family in ("dual_seasonality", "AM_beat"):
        return {"depth": float(rng.uniform(0.35, 0.60)), "ratio": 7}
    return {}


# ---------------------------------------------------------------- 렌더
def _jittered_u(L: int, P: int, phase0: float, jitter: float,
                rng: np.random.Generator) -> np.ndarray:
    """주기별 길이 jitter 를 적용한 u ∈ [0,1) 좌표열 (길이 L)."""
    chunks, total = [], 0
    need = L + 2 * P
    while total < need:
        n = int(round(P * (1.0 + rng.uniform(-jitter, jitter))))
        n = max(n, 8)
        chunks.append(np.arange(n) / n)
        total += n
    u = np.concatenate(chunks)
    off = int(round(phase0 * P)) % P
    return u[off:off + L]


def render_base(family: str, L: int, P: int, rng: np.random.Generator):
    """단위분산·영평균으로 표준화된 무잡음 베이스와 생성 파라미터를 반환."""
    shape = sample_shape_params(family, rng)
    phase0 = float(rng.uniform(0.0, 1.0))
    jitter = float(rng.uniform(config.JITTER_MIN, config.JITTER_MAX))
    u = _jittered_u(L, P, phase0, jitter, rng)
    y = WAVE_FN[family](u, shape).astype(np.float64)

    if family in ("dual_seasonality", "AM_beat"):
        t = np.arange(L)
        P2 = shape["ratio"] * P
        env = 1.0 + shape["depth"] * np.sin(2 * np.pi * t / P2 + 2 * np.pi * phase0)
        y = y * env
        shape = dict(shape, P2=P2)

    y = y - y.mean()
    sd = y.std()
    y = y / (sd if sd > 1e-9 else 1.0)
    params = {"family": family, "P": int(P), "phase0": phase0, "jitter": jitter,
              "shape": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in shape.items()}}
    return y, params


def add_noise_trend(base_clean: np.ndarray, noise_sigma: float, slope: float,
                    rng: np.random.Generator):
    """§5.2 잡음·추세 부가. base_clean 은 단위분산이므로 계수는 곧 ·A 배율."""
    L = len(base_clean)
    t = np.arange(L)
    trend = slope * (t - t.mean()) / L
    noise = rng.normal(0.0, noise_sigma, size=L)
    return base_clean + trend + noise, trend, noise
