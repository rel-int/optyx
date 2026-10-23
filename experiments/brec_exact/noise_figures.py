"""Figures of the noise paper from the stored results: brec_ideal_classical.pdf
(pairs separated against the ideal count, classical sources and linear
networks), sources_classical.pdf (counts against brightness per class, and
the phase-locked laser against the random-phase one) and
loss_sweep_classical.pdf.

    python noise_figures.py <figures-dir>
"""
import json
import os
import sys
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brecphot as bp  # noqa: E402
from report_noise import best_counts  # noqa: E402
from report_full import SEEDS, FLOOR  # noqa: E402

OUT = os.path.join(HERE, "results")
FIG = sys.argv[1] if len(sys.argv) > 1 else OUT
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
CLASSES = ["Basic", "Regular", "Extension", "CFI", "Strongly_regular", "4-vertex_condition", "Distance_regular"]
SHORT = {"Basic": "Basic", "Regular": "Regular", "Extension": "Extension", "CFI": "CFI", "Strongly_regular": "Strongly regular",
         "4-vertex_condition": "4-vertex condition", "Distance_regular": "Distance regular"}
THRESHOLD = 1e-12

noise = json.load(open(f"{OUT}/raw_noise.json"))
circ, nmax = best_counts(noise)
real_raw = json.load(open(f"{OUT}/raw_real.json"))
real, real_sep = {}, {}
for r in real_raw.values():
    if not isinstance(r, dict):
        continue
    real_sep[r["pair"]] = real_sep.get(r["pair"], False) or any(v > THRESHOLD for v in r["gaps"]["uniform"].values())
    if r["nmax"] == nmax[r["pair"]]:
        for w, by in r["ideal"]["uniform"].items():
            d = real.setdefault(r["pair"], {})
            d[float(w)] = min(d.get(float(w), np.inf), by[str(r["nmax"])])
ideal = json.load(open(f"{OUT}/raw_ideal_full.json"))
pairs = sorted(real)


def separated(pair, source):
    rs = [r for r in noise.values() if isinstance(r, dict) and r["pair"] == pair]
    return any(v > THRESHOLD for r in rs for v in r["gaps"][source].values())


def count(pair, source, w):
    if source == "uniform":
        return real[pair].get(w, np.inf) if real_sep.get(pair) else np.inf
    if source in ("step-lin", "init-lin"):
        recs = [ideal.get(f"gnn-compare-v4/{pair}/s{s}/{source}-ideal.json") for s in SEEDS]
        recs = [r for r in recs if isinstance(r, dict)]
        vals = [r["ideal_samples"][str(r["nmax"])] for r in recs if any(v["max_rel_diff"] > FLOOR for v in r["exact"].values())]
        return min(vals) if vals else np.inf
    return circ[pair][source].get(w, np.inf) if separated(pair, source) else np.inf


# ---- (1) pairs separated against the ideal count
series = [("coherent", 100.0, "laser, $w=100$", "C2", "-"), ("coherent", 1.0, "laser, $w=1$", "C2", "--"),
          ("uniform", 100.0, "phase-locked laser, $w=100$", "C4", "-"), ("uniform", 1.0, "phase-locked laser, $w=1$", "C4", "--"),
          ("photons p=0", 1.0, "distinguishable photons, $w=1$", "C1", "-"),
          ("thermal", 100.0, "thermal, $w=100$", "C3", "-"), ("step-lin", None, "Gaussian network, noise every step", "C5", ":"),
          ("init-lin", None, "Gaussian network, noise at init", "C5", "-.")]
grid = np.logspace(2, 20, 400)
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for source, w, label, colour, ls in series:
    vals = np.array([count(p, source, w) for p in pairs])
    ys = [(vals <= b).sum() for b in grid]
    ax.plot(grid, ys, ls, color=colour, lw=1.2, label=label)
    print(f"{label:36s} 1e6 {int((vals <= 1e6).sum()):3d}  1e9 {int((vals <= 1e9).sum()):3d}  hour {int((vals <= 3.6e12).sum()):3d}  day {int((vals <= 8.64e13).sum()):3d}  year {int((vals <= 3.16e16).sum()):3d}  total {int(np.isfinite(vals).sum()):3d}")
for y, label in ((270, "3-WL"), (298, "best trained network of BREC")):
    ax.axhline(y, color="k", lw=0.5, ls=(0, (1, 2)))
    ax.text(1.5e2, y + 3, label, fontsize=6)
ax.set_xscale("log")
ax.set_xlim(1e2, 1e20)
ax.set_ylim(0, 400)
ax.set_xlabel("ideal samples per network $T^\\star$")
ax.set_ylabel("BREC pairs separated")
top = ax.secondary_xaxis("top", functions=(lambda t: t * 1e-9, lambda s: s * 1e9))
top.set_xlabel("device time at one tick per nanosecond (s)", fontsize=7)
top.tick_params(labelsize=6)
ax.legend(frameon=False, loc="lower right", fontsize=5.8)
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(f"{FIG}/brec_ideal_classical.pdf")

# ---- (2) counts against brightness per class, and the phase-locked laser against the random-phase one
groups = [("order 1 (Basic, Regular, Extension)", lambda c: c in ("Basic", "Regular", "Extension")), ("CFI", lambda c: c == "CFI"),
          ("Strongly regular", lambda c: c == "Strongly_regular"), ("4-vertex condition", lambda c: c == "4-vertex_condition"),
          ("Distance regular", lambda c: c == "Distance_regular")]
wq = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
wc = wq + [2.0, 5.0, 10.0, 20.0, 100.0]
styles = {"coherent": ("laser", "C2", "-.", "^", wc), "uniform": ("phase-locked laser", "C4", "-", "v", wc),
          "photons p=0": ("distinguishable photons", "C1", "--", "s", wq), "thermal": ("thermal", "C3", ":", "o", wc)}
fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.4))
for ax, (title, pred) in zip(axes, groups):
    ps = [p for p in pairs if pred(p.rsplit("_", 1)[0])]
    for s, (label, colour, ls, marker, ws) in styles.items():
        ys = []
        for w in ws:
            v = [count(p, s, w) for p in ps]
            v = [u for u in v if np.isfinite(u)]
            ys.append(median(v) if v else np.nan)
        if not np.all(np.isnan(ys)):
            ax.plot(ws, ys, ls, color=colour, marker=marker, ms=3, lw=1.1, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{title} ({len(ps)})", fontsize=7.5)
    ax.set_xlabel("mean photons per pulse $w$")
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("ideal samples per network, median")
axes[0].legend(frameon=False, loc="upper right")
ax = axes[5]
for title, pred in groups:
    ps = [p for p in pairs if pred(p.rsplit("_", 1)[0])]
    ratio = []
    for w in wc:
        r = [count(p, "uniform", w) / count(p, "coherent", w) for p in ps if np.isfinite(count(p, "uniform", w)) and np.isfinite(count(p, "coherent", w))]
        ratio.append(median(r) if r else np.nan)
    ax.plot(wc, ratio, marker="o", ms=3, lw=1.1, label=title.split(" (")[0])
ax.axhline(1, color="k", lw=0.5, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("phase-locked / random-phase laser", fontsize=7.5)
ax.set_xlabel("mean photons per pulse $w$")
ax.legend(frameon=False, fontsize=5.5)
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(f"{FIG}/sources_classical.pdf")

# ---- (3) loss sweep, classical sources
pilot = json.load(open(f"{OUT}/results.json"))["sweep"]
gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
for ax, cat in zip(axes, ["Strongly_regular", "CFI", "4-vertex_condition"]):
    for source, (label, colour, ls) in {"coherent": ("laser", "C2", "-."), "photons p=0": ("distinguishable photons", "C1", "--"),
                                       "thermal": ("thermal", "C3", ":")}.items():
        ys = [median(pilot[f"{cat}|{g}|{source}"]) if pilot.get(f"{cat}|{g}|{source}") else np.nan for g in gammas]
        ax.plot(gammas, ys, ls, color=colour, marker="o", ms=3, lw=1.1, label=label)
    ax.set_yscale("log")
    ax.set_title(SHORT[cat] + f" ({len(bp.SUBSET[cat])} pairs)", fontsize=8)
    ax.set_xlabel(r"loop transmissivity $\gamma$")
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("ideal samples per network, median")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/loss_sweep_classical.pdf")

# ---- numbers for the text: reach by class per source, device-time table
print("\nreach (exact rule) by class:")
for s in ("coherent", "uniform", "photons p=0", "thermal"):
    row = {SHORT[c]: sum((real_sep.get(p, False) if s == "uniform" else separated(p, s)) for p in pairs if p.rsplit("_", 1)[0] == c) for c in CLASSES}
    print(f"  {s:12s}", row, "total", sum(row.values()))
for s in ("step-lin", "init-lin"):
    row = {SHORT[c]: sum(np.isfinite(count(p, s, None)) for p in pairs if p.rsplit("_", 1)[0] == c) for c in CLASSES}
    print(f"  {s:12s}", row, "total", sum(row.values()))
print("written", FIG)
