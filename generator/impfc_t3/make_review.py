#!/usr/bin/env python3
"""Review sheets for the t3 sets: one figure per task x type with four scenes each, and review/REVIEW.md."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import build as B  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "review"
INK, BLUE, GRAY, ORANGE = "#404040", "#2a78d6", "#8f8f8f", "#eb6834"
TARGET_BG, ROI_BG = "#eeeeee", "#d9e7f7"
NAMES = {"shift": "shift", "amplitude": "amplitude", "flat": "flat", "bounding": "bounding", "trend_rate": "trend rate",
         "frequency": "frequency", "self_repetition": "self-repetition", "delay": "delay"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def panel(ax, task, r):
    inp, pv = r["input"], r["private"]
    roi = np.asarray(pv["roi_mask"], bool)
    if task == "imp":
        first = inp["gap"][0]
        steps = np.arange(first, inp["gap"][1])
    else:
        first = inp["forecast_start"]
        steps = np.arange(first, first + inp["horizon"])
    values = np.array([np.nan if v is None else v for v in inp["values"]])
    ax.axvspan(steps[0] - .5, steps[-1] + .5, color=TARGET_BG, lw=0, zorder=0)
    idx = np.flatnonzero(roi)
    ax.axvspan(steps[idx[0]] - .5, steps[idx[-1]] + .5, color=ROI_BG, lw=0, zorder=0)
    if r["requirement"] == "delay":
        ref = inp["reference"]
        ax.plot(np.arange(ref["start"], ref["start"] + len(ref["values"])), ref["values"], color=ORANGE, lw=.9, alpha=.8, zorder=2)
    ax.plot(np.arange(len(values)), values, color=INK, lw=.8, marker="o", ms=1.2, zorder=3)
    truth = np.asarray(pv["truth"])
    before = values[first - 1] if first - 1 < len(values) and not np.isnan(values[first - 1]) else np.nan
    t_line = np.r_[first - 1, steps]
    y_line = np.r_[before, truth]
    after = first + len(steps)
    if task == "imp" and after < len(values):
        t_line, y_line = np.r_[t_line, after], np.r_[y_line, values[after]]
    ax.plot(t_line, y_line, color=BLUE, lw=1.5, zorder=4)
    ax.plot(steps, pv["usual_mean"], color=GRAY, lw=1.1, ls="--", zorder=4)
    ev = pv["event"]
    extra = {"self_repetition": lambda: f"{ev['which']} burst",
             "delay": lambda: f"delay {ev['d0']} -> {ev['d']}",
             "frequency": lambda: f"{ev['direction']}, period {ev['period']}"}.get(r["requirement"], lambda: str(r["tags"]["direction"]))()
    ax.set_title(f"{r['scene_id']}  ({extra})", loc="left", fontsize=8.5, color=INK, pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors="#555555")
    ax.grid(axis="y", color="#ececec", lw=.5)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "fig").mkdir(exist_ok=True)
    lines = ["# t3 IMP·FC 검수용 시트", "",
             "과제·유형마다 앞 네 장면을 그렸습니다. 검정은 관측값, 파랑은 정답, 회색 점선은 문맥을 무시한 답입니다.",
             "배경의 회색은 결측(IMP) 또는 예측(FC) 구간, 하늘색은 문맥이 말하는 구간입니다. 지연 유형의 주황은 시리즈 X입니다.", ""]
    for task in ("imp", "fc"):
        rows = read_jsonl(HERE / "evalsets" / task / "records.jsonl")
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene_id"], {})[r["level"]] = r
        lines += [f"## {task.upper()}", ""]
        for family in B.QUOTAS[task]:
            scenes = [v for k, v in by_scene.items() if v[1]["requirement"] == family][:4]
            fig, axes = plt.subplots(len(scenes), 1, figsize=(11, 2.1 * len(scenes)), gridspec_kw={"hspace": .6})
            for ax, sc in zip(np.atleast_1d(axes), scenes):
                panel(ax, task, sc[1])
            handles = [Line2D([], [], color=INK, lw=1.2, label="observed"), Line2D([], [], color=BLUE, lw=1.6, label="answer"),
                       Line2D([], [], color=GRAY, lw=1.2, ls="--", label="context ignored"),
                       Patch(facecolor=ROI_BG, label="interval in the context")]
            if family == "delay":
                handles.insert(1, Line2D([], [], color=ORANGE, lw=1.2, label="series X"))
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(.5, .995))
            name = f"{task}_{family}.png"
            fig.savefig(OUT / "fig" / name, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            first = scenes[0]
            lines += [f"### {task.upper()} · {NAMES[family]} ({B.QUOTAS[task][family]}장면)", "", f"![{task} {family}](fig/{name})", "",
                      f"`{first[1]['scene_id']}` 문맥", "", "| Level | 문맥 |", "|---|---|"]
            for lv in (1, 2):
                lines.append(f"| L{lv} | {first[lv]['input']['context'].replace(chr(10), ' ')} |")
            lines.append("")
    (OUT / "REVIEW.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "REVIEW.md")


if __name__ == "__main__":
    main()
