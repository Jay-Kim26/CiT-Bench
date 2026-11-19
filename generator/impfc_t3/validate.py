#!/usr/bin/env python3
"""Checks for the t3 IMP / FC sets (CPU only, no model):

  1. files, counts, quotas, cells, L1/L2 structure (same observations and ground truth, different context only);
  2. layout: missing block / forecast, interval position and the step numbers written in the context;
  3. prompts: Final global answer line, no "RoI" / "event" / "<answer>", series text equals the stored values;
  4. context necessity (replaces the A/B pairs): a 25-sample answer that follows the context (oracle) against one that
     ignores it (the usual signal; for delay scenes the usual delay applied to X), scored with the package's RCRPS;
  5. type-specific checks: single-peaked bursts and a distinct first / second burst, smooth joints for delay scenes.
Writes validation.json next to this file.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build as B

HERE = Path(__file__).resolve().parent
DATA = HERE / "evalsets"
PAIR = re.compile(r"\(?\s*(\d+)\s*[,:]\s*(-?\d+(?:\.\d+)?)\s*\)?")
MIN_GAIN = 0.005         # RCRPS(ignore context) - RCRPS(oracle), in normalized units
MIN_RATIO = 2.0          # and ignoring the context must at least double the oracle's RCRPS


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fair_crps(samples, truth):
    n = len(samples)
    w = 2 * np.arange(1, n + 1) - n - 1
    spread = (np.sort(samples, axis=0) * w[:, None]).sum(axis=0) / (n * (n - 1))
    return np.abs(samples - truth).mean(axis=0) - spread


def rcrps(crps, roi, rho):
    if roi.all() or not roi.any():
        return float(rho * crps.mean())
    return float(rho * (crps[roi].mean() + crps[~roi].mean()) / 2)


def draws(mean, sigma, censor, first, rng, n=25):
    x = np.asarray(mean)[None, :] + rng.normal(0, sigma, (n, len(mean)))
    if censor:
        lo, hi = censor["s"] - first, censor["e"] - first
        x[:, lo:hi] = (np.minimum if censor["side"] == "upper" else np.maximum)(x[:, lo:hi], censor["level"])
    return x


def series_blocks(user):
    return re.findall(r"<series(?: name=\"(\w)\")?>\n(.*?)\n</series>", user, re.S)


def parse_block(text):
    out = {}
    for cell in re.split(r",\s*|\n", text.strip()):
        i, v = cell.split(":")
        out[int(i)] = None if v == "NaN" else float(v)
    return out


def check_task(task):
    problems = []
    rows = read_jsonl(DATA / task / "records.jsonl")
    prompts = {p["id"]: p for p in read_jsonl(DATA / task / "prompts.jsonl")}
    ids = [r["id"] for r in rows]
    if len(ids) != 400 or len(set(ids)) != 400 or list(prompts) != ids:
        problems.append("ids / counts / prompt alignment")
    quotas = Counter(r["requirement"] for r in rows if r["level"] == 1)
    if dict(quotas) != B.QUOTAS[task]:
        problems.append(f"quotas {dict(quotas)}")
    cells = Counter(f"L{r['level']}/{r['axis']}" for r in rows)
    if set(cells.values()) != {100}:
        problems.append(f"cells {dict(cells)}")
    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[r["scene_id"]][r["level"]] = r
    rng = np.random.default_rng(20260925)
    gains = defaultdict(list)
    extra = defaultdict(list)
    for sid, lv in by_scene.items():
        if set(lv) != {1, 2}:
            problems.append(f"{sid}: levels {sorted(lv)}")
            continue
        a, b = lv[1], lv[2]
        same = {k: v for k, v in a["input"].items() if k != "context"} == {k: v for k, v in b["input"].items() if k != "context"}
        if not same or a["private"]["truth"] != b["private"]["truth"] or a["input"]["context"] == b["input"]["context"]:
            problems.append(f"{sid}: L1/L2 must share observations and truth and differ in context")
        family, inp, pv = a["requirement"], a["input"], a["private"]
        truth, roi = np.asarray(pv["truth"]), np.asarray(pv["roi_mask"], bool)
        s = pv["event"]["s"]
        # ---- layout and context steps
        if task == "imp":
            gs, ge = inp["gap"]
            missing = [i for i, v in enumerate(inp["values"]) if v is None]
            if missing != list(range(gs, ge)) or len(truth) != ge - gs or len(roi) != ge - gs:
                problems.append(f"{sid}: missing block")
            idx = np.flatnonzero(roi)
            if not (len(idx) and idx[-1] - idx[0] + 1 == len(idx) and idx[0] >= 2 and (ge - gs) - 1 - idx[-1] >= 2):
                problems.append(f"{sid}: interval must be contiguous with >= 2 other missing steps on each side")
            first, last = gs + idx[0], gs + idx[-1]
            for r in (a, b):
                if f"From step {first} through step {last}," not in r["input"]["context"]:
                    problems.append(f"{r['id']}: context interval")
            offset = gs
        else:
            T, H = inp["forecast_start"], inp["horizon"]
            if len(inp["values"]) != T or len(truth) != H or list(roi) != [t >= s for t in range(T, T + H)] or s - T < 6:
                problems.append(f"{sid}: forecast layout")
            for r in (a, b):
                if f"From step {s} onward," not in r["input"]["context"]:
                    problems.append(f"{r['id']}: context start step")
            offset = T
        if not np.isfinite(truth).all():
            problems.append(f"{sid}: truth not finite")
        # ---- prompts
        for r in (a, b):
            user = prompts[r["id"]]["messages"][1]["content"]
            n, d = len(truth), inp["decimals"]
            line = B.answer_line(n, d, family == "delay")
            if not user.endswith(line) or "RoI" in user or "event" in user or "<answer>" in user:
                problems.append(f"{r['id']}: prompt wording")
            blocks = series_blocks(user)
            want = 2 if family == "delay" else 1
            if len(blocks) != want:
                problems.append(f"{r['id']}: {len(blocks)} series blocks")
                continue
            y = parse_block(blocks[-1][1])
            if [y[i] for i in sorted(y)] != inp["values"]:
                problems.append(f"{r['id']}: series text differs from values")
            if family == "delay":
                x = parse_block(blocks[0][1])
                ref = inp["reference"]
                if sorted(x) != list(range(ref["start"], ref["start"] + len(ref["values"]))) or [x[i] for i in sorted(x)] != ref["values"]:
                    problems.append(f"{r['id']}: reference series text")
            if r["input"]["context"] not in user:
                problems.append(f"{r['id']}: context missing from prompt")
        # ---- context necessity: follow the context vs ignore it
        rho = 1.0 / float(np.ptp([v for v in inp["values"] if v is not None]))
        oracle = rcrps(fair_crps(draws(pv["gt_mean"], pv["sigma"], pv["censor"], offset, rng), truth), roi, rho)
        ignore = rcrps(fair_crps(draws(pv["usual_mean"], pv["sigma"], None, offset, rng), truth), roi, rho)
        gains[family].append((ignore - oracle, oracle, ignore))
        if ignore - oracle < MIN_GAIN or ignore < MIN_RATIO * oracle:
            problems.append(f"{sid}: ignoring the context costs only {ignore - oracle:.4f} ({ignore / oracle:.2f}x the oracle)")
        # ---- type-specific
        ev = pv["event"]
        if family == "self_repetition":
            bursts = ev["bursts"]
            k = 0 if ev["which"] == "first" else 1
            src = ev["source"][0] if task == "imp" else ev["source_start"]
            if bursts[k][0] != src or bursts[k][2] != ev["source_kind"]:
                problems.append(f"{sid}: L1 source is not the {ev['which']} burst")
            if any(bursts[i + 1][0] - (bursts[i][0] + bursts[i][1]) < 8 for i in range(len(bursts) - 1)):
                problems.append(f"{sid}: bursts closer than 8 quiet steps")
            if task == "imp":                                     # the repeated burst: one peak, or two with a high dip
                seg = np.asarray(pv["gt_mean"])[roi] - ev["quiet"]
                if not B.burst_ok(seg / seg.max(), "double" if ev["source_kind"] == "double" else "single"):
                    problems.append(f"{sid}: repeated burst shape")
            # copying another burst from the observations must score clearly worse than copying the right one
            y = np.array([np.nan if v is None else v for v in inp["values"]])
            if task == "imp":
                gs, ge = inp["gap"]
                r0, n = ev["s"], ev["D"]

                def copy(start):
                    m = np.full(ge - gs, ev["quiet"])
                    m[r0 - gs:r0 - gs + n] = y[start:start + n]
                    return m
                limit = len(y)
            else:
                T, H = inp["forecast_start"], inp["horizon"]
                n = T + H - s

                def copy(start):
                    m = np.full(H, ev["quiet"])
                    m[s - T:] = y[start:start + n]
                    return m
                limit = T
            right = rcrps(fair_crps(draws(copy(src), pv["sigma"], None, 0, rng), truth), roi, rho)
            wrong = [rcrps(fair_crps(draws(copy(b0), pv["sigma"], None, 0, rng), truth), roi, rho)
                     for b0, _, _ in bursts if b0 != src and b0 + n <= limit and not np.isnan(y[b0:b0 + n]).any()]
            if wrong:
                extra["wrong_copy_ratio"].append(min(wrong) / right)
                if min(wrong) < 1.5 * right:
                    problems.append(f"{sid}: copying another burst scores {min(wrong) / right:.2f}x the right copy")
            extra["double"].append(ev["source_kind"] == "double")
        if family == "delay":
            extra["joint_curvature_ratio"].append(ev["joint_curvature"] / ev["curvature_limit"])
            if ev["joint_curvature"] > ev["curvature_limit"] + 1e-9:
                problems.append(f"{sid}: joint curvature")
            if task == "fc" and inp["reference"]["start"] + len(inp["reference"]["values"]) != inp["forecast_start"] + inp["horizon"]:
                problems.append(f"{sid}: X must be observed through the last forecast step")
    lengths = [len(p["messages"][1]["content"]) for p in prompts.values()]
    report = {
        "queries": len(rows), "scenes": len(by_scene), "quotas": dict(quotas), "cells": dict(cells),
        "context_gain_rcrps": {f: {"min": round(min(g[0] for g in v), 4), "median": round(float(np.median([g[0] for g in v])), 4),
                                   "oracle_median": round(float(np.median([g[1] for g in v])), 4),
                                   "ignore_median": round(float(np.median([g[2] for g in v])), 4),
                                   "ratio_min": round(min(g[2] / g[1] for g in v), 2)} for f, v in gains.items()},
        "self_repetition_wrong_copy_ratio": {"min": round(min(extra["wrong_copy_ratio"]), 2),
                                             "median": round(float(np.median(extra["wrong_copy_ratio"])), 2)} if extra["wrong_copy_ratio"] else None,
        "self_repetition_double_peaked_source": int(sum(extra["double"])),
        "delay_joint_curvature_ratio_max": round(max(extra["joint_curvature_ratio"]), 3) if extra["joint_curvature_ratio"] else None,
        "prompt_chars": {"max": max(lengths), "median": int(np.median(lengths))},
        "problems": problems,
    }
    return report


def main():
    out = {task: check_task(task) for task in ("imp", "fc")}
    out["status"] = "PASS" if not any(out[t]["problems"] for t in ("imp", "fc")) else "FAIL"
    (HERE / "validation.json").write_text(json.dumps(out, indent=2) + "\n")
    for task in ("imp", "fc"):
        r = out[task]
        print(f"== {task}: {r['scenes']} scenes, {r['queries']} queries, problems {len(r['problems'])}")
        for p in r["problems"][:15]:
            print("   ", p)
        for f, g in r["context_gain_rcrps"].items():
            print(f"   {f:16s} gain min {g['min']:.4f} median {g['median']:.4f}  (oracle {g['oracle_median']:.4f}, ignore {g['ignore_median']:.4f}, ratio min {g['ratio_min']:.1f}x)")
        print("   wrong-copy / right-copy", r["self_repetition_wrong_copy_ratio"], "| two-peaked referenced bursts",
              r["self_repetition_double_peaked_source"], "| joint curvature ratio max", r["delay_joint_curvature_ratio_max"],
              "| prompt chars", r["prompt_chars"])
    print(out["status"])


if __name__ == "__main__":
    main()
