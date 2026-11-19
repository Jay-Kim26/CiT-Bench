"""CiTB Synthetic Context-in-AD track — 공통 상수/경로.

SPEC: Spec_ad_synthetic_track (2026-08-23 확정본).
모든 무작위성은 blake2b 시드 파생만 사용 (Python 내장 hash() 금지).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[1]          # Synthetic_Context_in_AD/
PKG = ROOT / "synth_ad"
DATASET = ROOT / "dataset"
PHASE1 = DATASET / "phase1"
REPORTS = ROOT / "reports"
GALLERY = ROOT / "gallery"

REGISTRY_YAML = PKG / "registry.yaml"
COMPAT_YAML = PKG / "compat_matrix.yaml"
CLAUSE_YAML = PKG / "clause_cards.yaml"

# ---------------------------------------------------------------- 시드 (§7)
SEED_ROOT = "42"
TRACK = "synth_ad"
GENERATOR_VERSION = "synth_ad-0.1.0"

# ---------------------------------------------------------------- 인스턴스 (§1, §5.2)
# 길이는 인스턴스마다 뽑는다 (v3). config.L 은 의도적으로 제거했다 —
# 남아 있으면 어딘가에서 조용히 512 를 쓰게 된다.
L_MIN, L_MAX = 160, 256     # 시계열 길이 범위 (v3, 상한 256 엄수)
CYC_MIN, CYC_MAX = 5, 8     # 가시 주기 수 → P ∈ [L/CYC_MAX, L/CYC_MIN]
STRIDE = 4                  # VQ stride (UCR AD 트랙과 동일)
W_MIN = 2 * STRIDE          # 국소 유형 하한 (§6.2)
# 유형별 폭 상한 (·P). 기본 1.0 = SPEC §6.2 의 W <= P.
# spike 는 Hann 테이퍼라 넓어지면 날카로운 특징이 사라져 amplitude_change 와 구별되지 않는다
# → 1/3 로 제한해 "국소 첨두" 성격을 유지한다 (2026-08-26).
TYPE_W_MAX_RATIO = {"spike": 1.0 / 3.0}
# 주기는 L 에 비례해 뽑는다 (아래 CYC_*). 절대 하한만 남긴다.
P_FLOOR = 20                # 전 축 공통 하한
P_MIN_D_AXIS = 20           # D축 하한. flat·level_shift 는 P=20 에서 지속시간 margin 2.5배가
                            # 100% 성립함을 실측 확인(2026-09-16). 구 값 56 은 spike×D 를 위한
                            # 것이었으나 spike×D 는 compat_matrix 에서 이미 배제돼 있다.
EDGE_MARGIN = 4             # 이탈 양 경계가 관측 가능하도록 시계열 끝에서 띄움
NOISE_LO, NOISE_HI = 0.02, 0.15         # ·A (A=1, base 는 단위분산 정규화)
NOISE_MIN_VARCOLLAPSE = 0.06            # 행렬 conditional 셀 (§6.3)
TREND_LO, TREND_HI = -0.3, 0.3          # ·A/L
JITTER_MIN, JITTER_MAX = 0.01, 0.05     # ·P, 0 금지 (§5.2)

# ---------------------------------------------------------------- 검증 (§1.1)
STD_SHIFT_MAX = 0.20        # 주입 후 std 변화율 상한
MP_PARITY_MIN = 0.75        # matrix profile 스코어 비 (min/max) 하한 — D·R축
MP_PARITY_RELAXED = 0.65    # 재샘플 소진 시 기록용 완화 임계
MP_EXCL_DIV = 2             # MP exclusion zone = m // 2
DUR_RATIO_MIN = 2.5         # D축 지속시간 margin (긴 쪽 / 짧은 쪽)
MAG_RATIO_MIN = 2.8         # M축 크기 margin
STRENGTH_LO_D, STRENGTH_HI_D = 0.20, 1.00   # D축 parity 탐색 범위
STRENGTH_LO_R, STRENGTH_HI_R = 0.80, 1.25   # R축 parity 탐색 범위
STRENGTH_FLOOR = {"variance_collapse": 0.60}    # 붕괴가 가시적으로 남도록 하한 별도
VC_RESIDUAL_MAX_RATIO = 0.50    # variance_collapse 가시성: 무잡음 베이스 대비 잔차 std 비
MAX_SCENE_ATTEMPTS = 60
MP_SEARCH_ITERS = 14

# ---------------------------------------------------------------- 구성 (§1, §8)
NULL_RATIO_TARGET = 0.25    # 구 스케줄 전용. 균형 스케줄은 PAIRS/NULLS_PER_CELL 을 쓴다
# v3 균형 스케줄 (REDESIGN_PLAN §2.2 A안) — (유형 × 축) 6셀에 균등 배분
#   6셀 × (10쌍 × 2 + 5 null) = 150 인스턴스, 축 주변분포 D=M=R=20 완전 균등
BALANCED_SCHEDULE = True
PAIRS_PER_CELL = 10
NULLS_PER_CELL = 5
PHASE1_FAMILIES = ("rectified_sine", "exp_rise_decay", "pulse_train", "asym_bimodal")
PHASE1_TYPES = ("spike", "flat", "level_shift")

# ---------------------------------------------------------------- 어휘 (§4.1, §6.1)
ANOMALY_VOCAB = (
    "amplitude_change", "period_change", "phase_jump", "level_shift",
    "flat", "dropout", "spike", "shape_replace", "time_reversal",
    "variance_burst", "variance_collapse", "quantization", "clipping",
    "missing_beat", "extra_beat", "trend_local", "stutter", "none",
)
# 17종 유형 어휘를 낱말 단위로 푼 전 카드 공통 금지어
TYPE_FORBIDDEN = (
    "amplitude", "period", "periodic", "phase", "jump", "level", "shift",
    "flat", "dropout", "spike", "shape", "replace", "reversal", "reverse",
    "variance", "collapse", "burst", "quantization", "quantize", "clipping",
    "clip", "missing", "extra", "beat", "trend", "stutter", "anomaly",
    "anomalous", "anomalies",
)
# 파형 패밀리명 (§4.1 — 파형 모양 기술 금지)
FAMILY_FORBIDDEN = (
    "sine", "sinusoid", "sinusoidal", "square", "sawtooth", "saw-tooth",
    "triangle", "triangular", "harmonic", "rectified", "saturated",
    "exponential", "pulse", "bimodal", "seasonality", "seasonal", "beat",
    "template", "tiling", "wave", "waveform",
)
