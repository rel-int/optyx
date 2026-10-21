"""Figures of the paper's expressivity section from the stored results:
brec_ideal.pdf (pairs separated against the ideal sample count on full
BREC), gnn_sampling.pdf (the protocol on real samples, subset) and
loss_sweep.pdf (ideal samples against the loop transmissivity)."""
import json
import os
import sys
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from report_full import SEEDS, FLOOR  # noqa: E402
from report_gnn import separated, BUDGETS, MODELS, LABEL  # noqa: E402
from launch import pairs_of  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = sys.argv[1] if len(sys.argv) > 1 else OUT
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
CLASSES = ["Basic", "Regular", "Strongly_regular", "Extension", "CFI", "4-vertex_condition", "Distance_regular"]
SHORT = {"Basic": "Basic", "Regular": "Regular", "Strongly_regular": "Str. regular", "Extension": "Extension",
         "CFI": "CFI", "4-vertex_condition": "4-vertex", "Distance_regular": "Dist. regular"}

full = json.load(open(f"{OUT}/raw_full.json"))
ideal = json.load(open(f"{OUT}/raw_ideal_full.json"))
data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
pairs = [(c, f"{c}_{i}") for c in data for i in range(len(data[c]))]


def cell_counts():
    """(model, pair) -> ideal samples, best cell, inf when the exact rule does not separate."""
    counts = {}
    for cat, pair in pairs:
        runs = [full.get(f"brec-full-coherent/{pair}/s{s}/g0.9.json") for s in SEEDS]
        runs = [r for r in runs if isinstance(r, dict)]
        for w in ("0.05", "1.0", "100.0"):
            vals = [r["coherent_samples_by_brightness"][w][str(r["nmax"])] for r in runs
                    if any(r["orders"][str(N)]["coherent"]["max_rel_diff"] > FLOOR for N in range(1, r["nmax"] + 1))]
            counts[(f"coherent-{w}", pair)] = min(vals) if vals else np.inf
        for source in ("photons p=1", "photons p=0", "thermal"):
            vals = [r["orders"][str(N)][source]["ideal_samples_poisson"] for r in runs for N in range(1, r["nmax"] + 1)
                    if r["orders"][str(N)][source]["max_rel_diff"] > FLOOR]
            counts[(source, pair)] = min(vals) if vals else np.inf
        for model in ("step-lin", "init-lin"):
            recs = [ideal.get(f"gnn-compare-v4/{pair}/s{s}/{model}-ideal.json") for s in SEEDS]
            recs = [r for r in recs if isinstance(r, dict)]
            vals = [r["ideal_samples"][str(r["nmax"])] for r in recs
                    if any(v["max_rel_diff"] > FLOOR for v in r["exact"].values())]
            counts[(model, pair)] = min(vals) if vals else np.inf
    return counts


counts = cell_counts()
grid = np.logspace(2, 22, 401)
STYLE = {"coherent-100.0": ("coherent light, $w=100$", "C2", "-"),
         "coherent-1.0": ("coherent light, $w=1$", "C2", "--"),
         "coherent-0.05": ("coherent light, $w=0.05$", "C2", ":"),
         "photons p=1": ("photons $p=1$, $w=0.05$", "C0", "-"),
         "photons p=0": ("photons $p=0$, $w=0.05$", "C1", "-"),
         "thermal": ("thermal light, $w=0.05$", "C3", "-"),
         "step-lin": ("GNN, noise every step", "C5", "-"),
         "init-lin": ("GNN, noise at init", "C4", "-")}

fig, ax = plt.subplots(figsize=(3.4, 3.1))
for model, (label, color, ls) in STYLE.items():
    vals = np.array([counts[(model, p)] for _, p in pairs])
    ax.plot(grid, [(vals <= g).sum() for g in grid], ls, color=color, label=label, lw=1.2)
for y, text in ((270, "3-WL"), (298, "best trained GNN"), (375, "exact, photons")):
    ax.axhline(y, color="k", lw=0.5, ls=(0, (1, 2)))
    ax.text(1.5e2, y + 3, text, fontsize=6)
ax.set_xscale("log")
ax.set_xlim(1e2, 1e23)
ax.set_ylim(0, 400)
ax.set_xlabel("ideal samples per network, best of four cells")
ax.set_ylabel("BREC pairs separated")
ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=2.2)
top = ax.secondary_xaxis("top", functions=(lambda T: T / 1e9, lambda t: t * 1e9))
top.set_xticks([1e-6, 1e-3, 1, 60, 3600, 86400, 3.16e7, 3.16e10])
top.set_xticklabels(["µs", "ms", "s", "min", "hour", "day", "year", "1000 yr"], fontsize=6)
top.set_xlabel("device time at one sample per nanosecond", fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/brec_ideal.pdf")

# numbers for the text
print("pairs within budgets (ideal, best cell):")
for model in STYLE:
    vals = np.array([counts[(model, p)] for _, p in pairs])
    print(f"  {model:16s}", {f"{b:.0e}": int((vals <= b).sum()) for b in (1e6, 1e9, 3.6e12, 8.64e13, 3.16e16)},
          "median per class:", {SHORT[c]: f"{median([counts[(model, p)] for cc, p in pairs if cc == c and np.isfinite(counts[(model, p)])]):.1e}"
                                for c in CLASSES if any(cc == c and np.isfinite(counts[(model, p)]) for cc, p in pairs)})

# ---- the protocol on real samples, subset
gnn = json.load(open(f"{OUT}/raw_gnn.json"))
runs = {}
for key, value in gnn.items():
    if not isinstance(value, dict):
        continue
    _, pair, seed, name = key.split("/")
    model, _, kind = name[:-5].rpartition("-")
    if kind == "retest":
        model, _, tkind = model.rpartition("-")
        runs.setdefault((pair, int(seed[1:]), model), {"budgets": {}})["budgets"].setdefault(tkind[1:], {})["retest"] = value["test"]
    elif kind.startswith("T"):
        runs.setdefault((pair, int(seed[1:]), model), {"budgets": {}})["budgets"][kind[1:]] = value
subset = sorted({p for (p, s, m) in runs})
print("subset pairs with results:", len(subset))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
COL = {"coherent-w100": ("C2", "-"), "step-lin": ("C5", "-"), "step-tanh": ("C5", "--"),
       "init-lin": ("C4", "-"), "init-tanh": ("C4", "--")}
for model in MODELS:
    ys = [len({p for (p, s, m), r in runs.items() if m == model and separated(r, T)}) for T in BUDGETS]
    color, ls = COL[model]
    ax.plot(BUDGETS, ys, ls, color=color, marker="o", ms=3, lw=1.2, label=LABEL[model])
exact_sub = {m: sum(1 for p in subset if np.isfinite(counts[(m, p)])) for m in ("coherent-100.0", "step-lin", "init-lin", "photons p=1")}
print("exact rule on the subset:", exact_sub)
ax.axhline(exact_sub["coherent-100.0"], color="C2", lw=0.6, ls=(0, (1, 2)))
ax.axhline(exact_sub["step-lin"], color="C5", lw=0.6, ls=(0, (1, 2)))
ax.text(3e6, exact_sub["coherent-100.0"] + 0.5, "exact rule, photonics", fontsize=6, color="C2")
ax.text(3e6, exact_sub["step-lin"] - 2.2, "exact rule, linear networks", fontsize=6, color="C5")
ax.set_xscale("log")
ax.set_xlabel("samples per network $T$")
ax.set_ylabel(f"pairs separated, of {len(subset)}")
ax.set_ylim(0, len(subset) + 2)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/gnn_sampling.pdf")

# ---- loss sweep
pilot = json.load(open(f"{OUT}/results.json"))["sweep"]
gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
for ax, cat in zip(axes, ["Strongly_regular", "CFI", "4-vertex_condition"]):
    for source, (label, color, ls) in {"photons p=1": ("photons $p=1$", "C0", "-"), "photons p=0": ("photons $p=0$", "C1", "--"),
                                       "coherent": ("coherent", "C2", "-."), "thermal": ("thermal", "C3", ":")}.items():
        ys = [median(pilot[f"{cat}|{g}|{source}"]) if pilot.get(f"{cat}|{g}|{source}") else np.nan for g in gammas]
        ax.plot(gammas, ys, ls, color=color, marker="o", ms=3, lw=1.1, label=label)
    ax.set_yscale("log")
    ax.set_title(SHORT[cat].replace("Str.", "Strongly") + f" ({len(bp.SUBSET[cat])} pairs)", fontsize=8)
    ax.set_xlabel(r"loop transmissivity $\gamma$")
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("ideal samples per network, median")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/loss_sweep.pdf")
print("written", FIG)
