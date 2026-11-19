"""Constants of the Final_2027ICLR build: subsets, event families, capability groups, cells, bases, counts.

Implements TimeSeries/ad_imp_capability_design.md (2026-09-22). Internal intervals are half-open
[s, e); prompts state inclusive endpoints once.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = ROOT.parent
OUT = ROOT / "evalsets"
VERSION = "final-2027iclr-v0"
SEED = 20260922

FAMILIES = ("trend", "frequency", "point", "range_excursion", "bounding", "flat")
# primary capability group -> event families it applies to (design doc, section 3)
GROUPS = {
    "TR-RATE": ("trend",), "FR-PERIOD": ("frequency",), "PT-MAGNITUDE": ("point",), "RG-MARGIN": ("range_excursion",),
    "BD-BOUND": ("bounding",), "FL-LEVEL": ("flat",),
    "TM-DURATION": ("trend", "frequency", "range_excursion", "bounding", "flat"),
    "TM-POSITION": FAMILIES,
}
TARGET_PARAM = {"SH-AMOUNT": "amount", "TR-RATE": "rate", "FR-PERIOD": "period", "PT-MAGNITUDE": "magnitude", "RG-MARGIN": "margin",
                "BD-BOUND": "bound", "FL-LEVEL": "level", "TM-DURATION": "duration", "TM-POSITION": "delay"}
# how the context locates a reference event: the locator must not give away the target parameter
LOCATOR = {"TM-DURATION": "start", "TM-POSITION": "marker"}          # every other group: "span"
# cells = (group, family, side); bounding runs both sides in its own group, one side elsewhere
CELLS = []
for g, fams in GROUPS.items():
    for f in fams:
        if f == "bounding" and g == "BD-BOUND":
            CELLS += [(g, f, "upper"), (g, f, "lower")]
        else:
            CELLS.append((g, f, None))
SCENES_PER_CELL = 10
# IMP (build_imp_v1.py): one cell per family, the event interval always stated in steps
IMP_FAMILIES = ("shift", "frequency", "flat", "bounding")
IMP_SCENES = 12
BASES = ("sine", "harmonic", "trend_sine", "rectified_sine", "exp_rise_decay", "pulse_train", "asym_bimodal")
IMP_GAPS = (24, 32, 48)
IMP_L_MIN, IMP_L_MAX = 128, 256
AD_L_MIN, AD_L_MAX = 160, 256
K_IMP, N_AD = 25, 5
# (P', D) per P for frequency events: the net phase change is a whole number of cycles (the pattern after
# the segment is in step with the pattern before it) and the new interval stays within 0.5x-2x of the usual one.
FREQ_COMBOS = {12: [(6, 12), (6, 24), (8, 24), (24, 24), (18, 36)],
               16: [(8, 16), (8, 32), (32, 32)],
               18: [(9, 18), (9, 36), (36, 36)],
               20: [(10, 20), (10, 40), (40, 40)],
               24: [(12, 24), (8, 24), (8, 12)]}
