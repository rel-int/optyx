"""The cost law of antibunching: measured against predicted ratios of ideal
counts, laser over antibunched source, and the exceptions explained by what
thermal light sees.

    python antibunching_figure.py <figures-dir>
"""
import json
import os
import sys
from statistics import median

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CLASSES = ["Basic", "Regular", "Extension", "CFI", "Strongly_regular", "4-vertex_condition", "Distance_regular"]
SHORT = {"Basic": "Basic", "Regular": "Regular", "Extension": "Extension", "CFI": "CFI", "Strongly_regular": "Strongly regular",
         "4-vertex_condition": "4-vertex cond.", "Distance_regular": "Distance regular"}
W = 0.05


def f2(K):
    """Pair weight of the Fock state |K>, -1 for single photons."""
    return -(1 + 1 / K) / 2


def predicted(K):
    """Samples with the laser over samples with |K>, pair-dominated case."""
    return 4 * f2(K) ** 2


def best_full(raw):
    """pair -> source -> w -> ideal count at the top order, best cell (full BREC run)."""
    nmax, counts = {}, {}
    for key, r in raw.items():
        if isinstance(r, dict):
            nmax[r["pair"]] = max(nmax.get(r["pair"], 0), r["nmax"])
    for r in raw.values():
        if isinstance(r, dict) and r["nmax"] == nmax[r["pair"]]:
            for source, by_w in r["ideal"].items():
                for w, by_order in by_w.items():
                    d = counts.setdefault(r["pair"], {}).setdefault(source, {})
                    d[float(w)] = min(d.get(float(w), np.inf), by_order[str(r["nmax"])])
    return counts


def best_kmode(raw):
    """(pair, K, placement, source, w) -> ideal count at the top order, best cell."""
    counts = {}
    for r in raw.values():
        if not isinstance(r, dict):
            continue
        for placement, by_source in r["ideal"].items():
            for source, by_w in by_source.items():
                for w, by_order in by_w.items():
                    key = (r["pair"], r["local_modes"], placement, source, float(w))
                    counts[key] = min(counts.get(key, np.inf), by_order[str(r["nmax"])])
    return counts


def ratios(values):
    v = np.array([x for x in values if np.isfinite(x) and x > 0])
    return v


def main(figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    full = best_full(json.load(open(f"{OUT}/raw_noise.json")))
    kmode = best_kmode(json.load(open(f"{OUT}/raw_kmode.json")))
    kpairs = sorted({k[0] for k in kmode})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 4.3))

    # (a) measured ratio against K, the two predictions
    def show(x, v, label, colour, marker, filled=True):
        lo, mid, hi = np.percentile(v, [10, 50, 90])
        ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], fmt=marker, color=colour, ms=4, lw=0.8, capsize=1.5,
                    markerfacecolor=colour if filled else "white", label=label)
        print(f"x={x:.3f} median={mid:.2f} 10-90%={lo:.2f}-{hi:.2f} n={len(v)} {label}")

    for K in (1, 2, 4, 8):
        v = ratios([kmode.get((p, K, "separate", "coherent", W), np.nan) / kmode.get((p, K, "separate", "photons p=1", W), np.nan)
                    for p in kpairs])
        show(K * 0.97, v, "$K$ photons in $K$ modes" if K == 1 else None, "C0", "o")
    for K in (2, 4, 8):
        for placement, marker, dx in (("symmetric", "s", 1.0), ("single", "D", 1.06)):
            v = ratios([kmode.get((p, K, placement, "coherent", K * W), np.nan) / kmode.get((p, K, placement, f"fock {K}", K * W), np.nan)
                        for p in kpairs])
            show(K * dx, v, {"symmetric": "$|K\\rangle$ in the symmetric mode", "single": "$|K\\rangle$ in one mode"}[placement] if K == 2 else None,
                 "C1", marker, filled=False)
    v = ratios([full[p]["coherent"].get(W, np.nan) / full[p]["photons p=1"].get(W, np.nan) for p in full])
    show(1 * 1.03, v, "all 400 pairs, $w=0.05$", "k", "x")
    for K in (2, 5, 10):
        v = ratios([full[p]["coherent"].get(float(K), np.nan) / full[p][f"fock {K}"].get(float(K), np.nan) for p in full if f"fock {K}" in full[p]])
        show(K * 1.03, v, "all 400 pairs, $|K\\rangle$ at $w=K$" if K == 2 else None, "k", "+")
    grid = np.logspace(0, np.log10(11), 50)
    ax.axhline(4, color="C0", lw=0.7, ls="--", label="$4$")
    ax.plot(grid, (1 + 1 / grid) ** 2, "--", color="C1", lw=0.7, label="$(1+1/K)^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 12)
    ax.set_xlabel("photons per vertex and tick $K$ (detected flux $0.05K$)")
    ax.set_ylabel("samples, laser / antibunched source")
    ax.set_ylim(0.8, 400)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_title("(a)", loc="left", fontsize=8)

    # (b) the exceptions: what thermal light sees
    for i, c in enumerate(CLASSES):
        xs, ys = [], []
        for p in full:
            if p.rsplit("_", 1)[0] != c:
                continue
            ph, co, th = (full[p][s].get(W, np.inf) for s in ("photons p=1", "coherent", "thermal"))
            if np.isfinite(ph) and np.isfinite(co):
                xs.append(max(ph / th, 1e-16) if np.isfinite(th) else 1e-16)
                ys.append(co / ph)
        bx.plot(xs, ys, "o", ms=2.5, color=f"C{i}", alpha=0.8, label=f"{SHORT[c]} ({len(xs)})", mew=0)
    grid = np.logspace(-16, 0, 200)
    bx.plot(grid, 4 / (1 + 3 * grid), "k--", lw=0.7, label="$4/(1+3x)$")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(3e-17, 2)
    bx.set_xlabel("$x$ = samples with photons / samples with thermal light")
    bx.set_ylim(0.8, 120)
    bx.set_ylabel("samples, laser / photons")
    bx.legend(frameon=False, ncol=2, loc="upper right")
    bx.grid(alpha=0.3, lw=0.5)
    bx.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{figdir}/antibunching.pdf")
    # numbers for the text
    xs = np.array([full[p]["photons p=1"].get(W, np.inf) / full[p]["thermal"].get(W, np.inf) for p in full
                   if np.isfinite(full[p]["photons p=1"].get(W, np.inf)) and np.isfinite(full[p]["coherent"].get(W, np.inf))])
    ys = np.array([full[p]["coherent"].get(W, np.inf) / full[p]["photons p=1"].get(W, np.inf) for p in full
                   if np.isfinite(full[p]["photons p=1"].get(W, np.inf)) and np.isfinite(full[p]["coherent"].get(W, np.inf))])
    xs = np.where(np.isfinite(xs), xs, 0)
    pred = 4 / (1 + 3 * xs)
    dev = ys / pred
    print(f"(b): {len(ys)} pairs, measured/predicted median {median(dev):.3f}, within 20% on {np.mean(np.abs(np.log(dev)) < np.log(1.2)):.0%}, "
          f"within 2x on {np.mean(np.abs(np.log(dev)) < np.log(2)):.0%}")
    for c in CLASSES:
        d = [ys[i] / pred[i] for i, p in enumerate([p for p in full if np.isfinite(full[p]["photons p=1"].get(W, np.inf)) and np.isfinite(full[p]["coherent"].get(W, np.inf))]) if p.rsplit("_", 1)[0] == c]
        if d:
            print(f"   {c:20s} median measured/predicted {median(d):.2f}, 10-90% {np.percentile(d, 10):.2f}-{np.percentile(d, 90):.2f}, n={len(d)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/user/wiki/paper/figures")
