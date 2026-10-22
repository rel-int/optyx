"""Fetch and tabulate the exact-noise run, and draw the brightness figure.

    python report_noise.py fetch
    python report_noise.py tables [figures-dir]
"""
import base64
import gzip
import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402
from noise_app import BRIGHTNESSES, CLASSICAL_BRIGHTNESSES, FOCK, EXPERIMENT  # noqa: E402
from report_full import CLASSES, SHORT, SEEDS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER1 = ("Basic", "Regular", "Extension")


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    data = bp.load_brec(BREC_RAW)
    pairs = [f"{c}_{i}" for c in data for i in range(len(data[c]))]
    chunks = [pairs[i:i + 10] for i in range(0, len(pairs), 10)]
    out = {}
    names = ["g0.9-N4.json", "g0.9-N3.json", "g0.9-N2.json"]
    for blob in read_many.starmap([(EXPERIMENT, ch, names) for ch in chunks]):
        out.update(json.loads(gzip.decompress(base64.b64decode(blob))))
    json.dump(out, open(f"{OUT}/raw_noise.json", "w"))
    print(len(out), "files")


def best_counts(raw):
    """(pair) -> source -> w -> ideal count at the top order, best cell."""
    counts, nmax = {}, {}
    for key, r in raw.items():
        if isinstance(r, dict):
            pair = key.split("/")[1]
            nmax[pair] = max(nmax.get(pair, 0), r["nmax"])
    for key, r in raw.items():
        if not isinstance(r, dict):
            continue
        pair = key.split("/")[1]
        if r["nmax"] != nmax[pair]:
            continue
        for source, by_w in r["ideal"].items():
            for w, by_order in by_w.items():
                c = by_order[str(r["nmax"])]
                d = counts.setdefault(pair, {}).setdefault(source, {})
                d[float(w)] = min(d.get(float(w), np.inf), c)
    return counts, nmax


def tables(figdir=None):
    raw = json.load(open(f"{OUT}/raw_noise.json"))
    counts, nmax = best_counts(raw)
    pairs = sorted(counts)
    lines = [f"## Exact noise: pairs {len(pairs)}, orders " + str(sorted(set(nmax.values()))) + "\n"]
    lines.append("## Median ideal samples per class against brightness (best cell, all orders)\n")
    grid = sorted(set(BRIGHTNESSES) | set(CLASSICAL_BRIGHTNESSES))
    lines.append("| Class | Source | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    lines.append("|---|---|" + "---|" * len(grid))
    for c in CLASSES:
        ps = [p for p in pairs if p.rsplit("_", 1)[0] == c]
        for source in list(bp.SOURCES) + [f"fock {K}" for K in FOCK]:
            cells = []
            for w in grid:
                v = [counts[p][source][w] for p in ps if source in counts[p] and w in counts[p][source]
                     and np.isfinite(counts[p][source][w])]
                cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
            lines.append(f"| {SHORT[c]} | {source} | " + " | ".join(cells) + " |")
    lines.append("\n## Pairs within a device-time budget at 1 GHz, all orders\n")
    lines.append("| Source | w | an hour | a day | a year |")
    lines.append("|---|---|---|---|---|")
    for source in bp.SOURCES:
        for w in grid:
            vals = [counts[p][source].get(w, np.inf) for p in pairs if source in counts[p]]
            if not any(np.isfinite(vals)):
                continue
            lines.append(f"| {source} | {w:g} | " + " | ".join(str(sum(v <= b for v in vals)) for b in (3.6e12, 8.64e13, 3.16e16)) + " |")
    lines.append("\n## Coherent light relative to indistinguishable photons at equal brightness, median per class\n")
    lines.append("| Class | " + " | ".join(f"w = {w:g}" for w in BRIGHTNESSES) + " |")
    lines.append("|---|" + "---|" * len(BRIGHTNESSES))
    for c in CLASSES:
        ps = [p for p in pairs if p.rsplit("_", 1)[0] == c]
        cells = []
        for w in BRIGHTNESSES:
            r = [counts[p]["coherent"][w] / counts[p]["photons p=1"][w] for p in ps
                 if np.isfinite(counts[p]["photons p=1"].get(w, np.inf)) and np.isfinite(counts[p]["coherent"].get(w, np.inf))]
            cells.append(f"{median(r):.2f} ({len(r)})" if r else "-")
        lines.append(f"| {SHORT[c]} | " + " | ".join(cells) + " |")
    # sanity: the exact counts at w = 0.05 against the Poisson ones of the full run, and against its coherent Monte Carlo
    full_path = f"{OUT}/raw_full.json"
    if os.path.exists(full_path):
        full = json.load(open(full_path))
        lines.append("\n## Exact noise against the full run at w = 0.05: ratio of counts, median (10th-90th percentile) over pairs, best cells\n")
        lines.append("| Comparison | ratio |")
        lines.append("|---|---|")
        order1 = {}
        for key, r in raw.items():
            if isinstance(r, dict) and r["nmax"] == nmax[key.split("/")[1]]:
                for source in ("photons p=1", "photons p=0", "thermal"):
                    d = order1.setdefault(key.split("/")[1], {})
                    d[source] = min(d.get(source, np.inf), r["ideal"][source]["0.05"]["1"])
        for source, label in (("photons p=1", "photons p=1, exact / Poisson, order 1"), ("photons p=0", "photons p=0, exact / Poisson, order 1"),
                              ("thermal", "thermal, exact / Poisson, order 1"), ("coherent", "coherent, exact / Monte Carlo, all orders")):
            ratios = []
            for p in pairs:
                runs = [full.get(f"brec-full-coherent/{p}/s{s}/g0.9.json") for s in SEEDS]
                runs = [r for r in runs if isinstance(r, dict) and r["nmax"] == nmax[p]]
                if not runs:
                    continue
                if source == "coherent":
                    if not np.isfinite(counts[p][source].get(0.05, np.inf)):
                        continue
                    ref = min(r["coherent_samples_by_brightness"]["0.05"][str(r["nmax"])] for r in runs)
                    ratios.append(counts[p][source][0.05] / ref)
                else:
                    ref = min(r["orders"]["1"][source]["ideal_samples_poisson"] for r in runs)
                    if np.isfinite(ref) and np.isfinite(order1.get(p, {}).get(source, np.inf)) and ref < 1e25:
                        ratios.append(order1[p][source] / ref)
            if ratios:
                lines.append(f"| {label} | {median(ratios):.3f} ({np.percentile(ratios, 10):.3f}-{np.percentile(ratios, 90):.3f}), {len(ratios)} pairs |")
    seconds = [r["seconds"] for r in raw.values() if isinstance(r, dict)]
    lines.append(f"\nGPU seconds per pair and cell: median {median(seconds):.0f}, max {max(seconds):.0f}, total hours {sum(seconds) / 3600:.1f}")
    open(f"{OUT}/tables_noise.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if figdir:
        figure(counts, pairs, figdir)


def figure(counts, pairs, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    groups = [("order 1 (Basic, Regular, Extension)", lambda c: c in ORDER1), ("CFI", lambda c: c == "CFI"),
              ("Strongly regular", lambda c: c == "Strongly_regular"), ("4-vertex condition", lambda c: c == "4-vertex_condition"),
              ("Distance regular", lambda c: c == "Distance_regular")]
    fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.4))
    styles = {"photons p=1": ("photons $p=1$", "C0", "-", "o"), "photons p=0": ("photons $p=0$", "C1", "--", "s"),
              "coherent": ("coherent", "C2", "-.", "^"), "thermal": ("thermal", "C3", ":", "v")}

    def med(ps, source, w):
        v = [counts[p][source][w] for p in ps if source in counts[p] and w in counts[p][source] and np.isfinite(counts[p][source][w])]
        return median(v) if v else np.nan, len(v)

    for ax, (title, pred) in zip(axes, groups):
        ps = [p for p in pairs if pred(p.rsplit("_", 1)[0])]
        for source, (label, color, ls, marker) in styles.items():
            ws = sorted(BRIGHTNESSES + (CLASSICAL_BRIGHTNESSES if source in ("coherent", "thermal") else ()))
            ys = [med(ps, source, w)[0] for w in ws]
            if not np.all(np.isnan(ys)):
                ax.plot(ws, ys, ls, color=color, marker=marker, ms=3, lw=1.1, label=label)
        fock_w = [float(K) for K in FOCK]
        ys = [med(ps, "photons p=1", 1.0)[0]] + [med(ps, f"fock {K}", float(K))[0] for K in FOCK]
        if not np.all(np.isnan(ys)):
            ax.plot([1.0] + fock_w, ys, "-", color="C0", lw=1.1,
                    marker="o", ms=3.5, markerfacecolor="white", label="Fock $|w\\rangle$")
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
        ws = list(BRIGHTNESSES)
        ratio = []
        for w in ws:
            r = [counts[p]["coherent"][w] / counts[p]["photons p=1"][w] for p in ps
                 if np.isfinite(counts[p]["photons p=1"].get(w, np.inf)) and np.isfinite(counts[p]["coherent"].get(w, np.inf))]
            ratio.append(median(r) if r else np.nan)
        if np.all(np.isnan(ratio)):
            continue
        ax.plot(ws, ratio, marker="o", ms=3, lw=1.1, label=title.split(" (")[0])
        fr = []
        for K in FOCK:
            r = [counts[p]["coherent"].get(float(K), np.inf) / counts[p][f"fock {K}"][float(K)] for p in ps
                 if np.isfinite(counts[p][f"fock {K}"].get(float(K), np.inf)) and np.isfinite(counts[p]["coherent"].get(float(K), np.inf))]
            fr.append(median(r) if r else np.nan)
        ax.plot([float(K) for K in FOCK], fr, "o", ms=3.5, markerfacecolor="white", color=ax.lines[-1].get_color())
    wgrid = np.logspace(np.log10(0.02), 1, 50)
    ax.plot(wgrid, (1 + 1 / wgrid) ** 2, "k--", lw=0.8, label="$(1+1/w)^2$")
    ax.axhline(4, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("coherent / photons", fontsize=7.5)
    ax.set_xlabel("mean photons per pulse $w$")
    ax.legend(frameon=False, fontsize=5.5)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{figdir}/sources_brightness.pdf")
    print("figure written")


if __name__ == "__main__":
    {"fetch": fetch, "tables": lambda: tables(sys.argv[2] if len(sys.argv) > 2 else None)}[sys.argv[1]]()
