"""Figures of the quantum paper's expressivity section: the distance-regular
class against the best classical randomised light and distinguishable
photons (dr_pairs.pdf, dr_brightness.pdf) and the block decomposition of
the gaps (blocks.pdf).

    python quantum_figures.py <figures-dir>
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
from report_noise import best_counts  # noqa: E402

OUT = os.path.join(HERE, "results")
FIG = sys.argv[1] if len(sys.argv) > 1 else OUT
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
CLASSES = ["Basic", "Regular", "Extension", "CFI", "Strongly_regular", "4-vertex_condition", "Distance_regular"]
SHORT = {"Basic": "Basic", "Regular": "Regular", "Extension": "Extension", "CFI": "CFI", "Strongly_regular": "Str. regular",
         "4-vertex_condition": "4-vertex", "Distance_regular": "Dist. regular"}
THRESHOLD = 1e-12

noise = json.load(open(f"{OUT}/raw_noise.json"))
circ, nmax = best_counts(noise)
real_raw = json.load(open(f"{OUT}/raw_real.json"))
real = {}
for r in real_raw.values():
    if isinstance(r, dict) and r["nmax"] == nmax[r["pair"]]:
        for w, by in r["ideal"]["uniform"].items():
            d = real.setdefault(r["pair"], {})
            d[float(w)] = min(d.get(float(w), np.inf), by[str(r["nmax"])])


def separated(pair, source):
    rs = [r for r in noise.values() if isinstance(r, dict) and r["pair"] == pair]
    return any(v > THRESHOLD for r in rs for v in r["gaps"][source].values())


def floor(pair, source):
    table = real[pair] if source == "uniform" else circ[pair][source]
    return min(table.values()) if table else np.inf


pairs = sorted(real)
dr = sorted([p for p in pairs if p.startswith("Distance_regular") and separated(p, "photons p=1")],
            key=lambda p: circ[p]["photons p=1"][1.0])

# ---- (1) per-pair counts on the distance-regular class
fig, ax = plt.subplots(figsize=(3.4, 2.6))
x = np.arange(len(dr))
series = [("photons p=1", lambda p: circ[p]["photons p=1"][1.0], "single photons, $w=1$", "C0", "o", True),
          ("coherent", lambda p: floor(p, "coherent"), "laser, best $w$", "C2", "^", True),
          ("uniform", lambda p: floor(p, "uniform"), "phase-locked laser, best $w$", "C2", "v", False),
          ("photons p=0", lambda p: floor(p, "photons p=0"), "distinguishable photons, best $w$", "C1", "s", True),
          ("fock 10", lambda p: circ[p]["fock 10"].get(10.0, np.inf), "Fock $|10\\rangle$, $w=10$", "C0", "D", False)]
for key, f, label, colour, marker, filled in series:
    ys = [f(p) for p in dr]
    ax.plot(x, ys, marker, color=colour, ms=4, markerfacecolor=colour if filled else "white", label=label, lw=0)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([p.split("_")[-1] for p in dr], fontsize=6)
ax.set_xlabel("distance-regular pair, ordered by the photons' count")
ax.set_ylabel("ideal samples per network")
ax.grid(alpha=0.3, lw=0.5, axis="y")
ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/dr_pairs.pdf")
print("dr pairs:", len(dr))
for p in dr:
    q = circ[p]["photons p=1"][1.0]
    print(f"  {p:20s} photons@1 {q:.1e}  laser/ {floor(p, 'coherent') / q:5.2f}  locked/ {floor(p, 'uniform') / q:5.2f}  p=0/ {floor(p, 'photons p=0') / q:5.2f}  fock10/ {circ[p]['fock 10'].get(10.0, np.inf) / q:5.2f}")

# ---- (2) brightness curves on the class, and the ratio of the best classical floor to photons at w = 1 per class
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))
grid_q = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
grid_c = grid_q + [2.0, 5.0, 10.0, 20.0, 100.0]
styles = {"photons p=1": ("single photons", "C0", "-", "o", grid_q), "photons p=0": ("distinguishable photons", "C1", "--", "s", grid_q),
          "coherent": ("laser", "C2", "-.", "^", grid_c), "uniform": ("phase-locked laser", "C2", ":", "v", grid_c)}
for s, (label, colour, ls, marker, grid) in styles.items():
    ys = []
    for w in grid:
        v = [(real[p] if s == "uniform" else circ[p][s]).get(w, np.nan) for p in dr]
        v = [u for u in v if np.isfinite(u)]
        ys.append(median(v) if v else np.nan)
    ax.plot(grid, ys, ls, color=colour, marker=marker, ms=3, lw=1.1, label=label)
fock = [median([circ[p]["photons p=1"][1.0] for p in dr])] + [median([circ[p][f"fock {K}"][float(K)] for p in dr if f"fock {K}" in circ[p]]) for K in (2, 5, 10)]
ax.plot([1.0, 2.0, 5.0, 10.0], fock, "-", color="C0", lw=1.1, marker="o", ms=3.5, markerfacecolor="white", label="Fock $|w\\rangle$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("mean photons per pulse $w$")
ax.set_ylabel("ideal samples per network, median")
ax.set_title(f"(a) distance regular ({len(dr)} pairs)", fontsize=8, loc="left")
ax.legend(frameon=False)
ax.grid(alpha=0.3, lw=0.5)
labels, best_cl, p0, lock, las = [], [], [], [], []
for c in CLASSES:
    ps = [p for p in pairs if p.rsplit("_", 1)[0] == c and separated(p, "photons p=1") and np.isfinite(circ[p]["photons p=1"].get(1.0, np.inf))]
    q = {p: circ[p]["photons p=1"][1.0] for p in ps}
    labels.append({"Basic": "Basic", "Regular": "Regular", "Extension": "Ext.", "CFI": "CFI", "Strongly_regular": "SR", "4-vertex_condition": "4-vertex", "Distance_regular": "DR"}[c] + f"\n({len(ps)})")
    las.append(median([floor(p, "coherent") / q[p] for p in ps]))
    lock.append(median([floor(p, "uniform") / q[p] for p in ps]))
    p0.append(median([floor(p, "photons p=0") / q[p] for p in ps]))
xs = np.arange(len(CLASSES))
bx.bar(xs - 0.27, las, 0.27, color="C2", label="laser, best $w$")
bx.bar(xs, lock, 0.27, color="C2", alpha=0.45, label="phase-locked laser, best $w$")
bx.bar(xs + 0.27, p0, 0.27, color="C1", label="distinguishable photons, best $w$")
bx.axhline(1, color="k", lw=0.7, ls="--")
bx.set_yscale("log")
bx.set_xticks(xs)
bx.set_xticklabels(labels, fontsize=6.5)
bx.set_ylabel("samples / single photons at $w=1$")
bx.set_title("(b) best brightness of each source, median per class", fontsize=8, loc="left")
bx.legend(frameon=False, loc="upper left")
bx.grid(alpha=0.3, lw=0.5, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG}/dr_brightness.pdf")
print("class ratios laser/lock/p0:", dict(zip([SHORT[c] for c in CLASSES], zip(np.round(las, 2), np.round(lock, 2), np.round(p0, 2)))))

# ---- (3) block decomposition of the gaps
mech = json.load(open(f"{OUT}/raw_mechanism.json"))
order_of = {"Distance_regular_1": 3, "Distance_regular_18": 4, "Distance_regular_19": 4, "Distance_regular_10": 3, "Distance_regular_0": 3,
            "Strongly_regular_0": 3, "Strongly_regular_1": 3, "4-vertex_condition_0": 3, "CFI_17": 2, "CFI_12": 2, "Basic_0": 2, "Extension_11": 2}
show = ["Basic_0", "Extension_11", "CFI_12", "CFI_17", "Strongly_regular_0", "Strongly_regular_1", "4-vertex_condition_0",
        "Distance_regular_0", "Distance_regular_1", "Distance_regular_10", "Distance_regular_18", "Distance_regular_19"]
fig, ax = plt.subplots(figsize=(7.0, 2.5))
blocks = [("pair", "pair block, $f(2)=-1$", "0.6"), ("triple", "three-photon block, $f(3)=1$", "0.3"), ("quad", "four-photon block, $f(4)=1$", "0.0")]
sources = [("photons p=1", "single photons", "C0", "o"), ("coherent", "laser", "C2", "^"), ("photons p=0", "distinguishable", "C1", "s")]
xs = np.arange(len(show))
for i, (key, label, colour) in enumerate(blocks):
    ys = []
    for p in show:
        N = order_of[p]
        v = [r["gaps"][str(N)][key]["gap"] for r in mech if r["pair"] == p]
        ys.append(median(v))
    ax.bar(xs + (i - 1) * 0.27, ys, 0.27, color=colour, label=label)
for key, label, colour, marker in sources:
    ys = [median([r["gaps"][str(order_of[p])][key]["gap"] for r in mech if r["pair"] == p]) for p in show]
    ax.plot(xs, ys, marker, color=colour, ms=4.5, label=label, lw=0, markeredgecolor="white", markeredgewidth=0.5)
ax.set_yscale("log")
ax.set_ylim(1e-13, 1)
ax.set_xticks(xs)
ax.set_xticklabels([p.replace("_condition", "").replace("Strongly_regular", "SR").replace("Distance_regular", "DR").replace("4-vertex", "4V").replace("Extension", "Ext") + f"\n$N={order_of[p]}$" for p in show], fontsize=6)
ax.set_ylabel("gap of the sorted profile at order $N$")
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.grid(alpha=0.3, lw=0.5, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG}/blocks.pdf")
for p in show:
    N = order_of[p]
    g = {k: median([r["gaps"][str(N)][k]["gap"] for r in mech if r["pair"] == p]) for k in ("pair", "triple", "quad", "photons p=1", "coherent", "photons p=0")}
    print(f"  {p:22s} N={N} triple/pair {g['triple'] / g['pair']:5.2f}  photons/laser {g['photons p=1'] / g['coherent']:5.2f} -> samples {(g['photons p=1'] / g['coherent']) ** 2:5.1f}  photons/p0 {g['photons p=1'] / g['photons p=0']:5.2f}")
print("written", FIG)
