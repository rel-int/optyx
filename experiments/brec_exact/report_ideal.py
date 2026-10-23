"""The exact rows of the linear networks on full BREC next to the photonic
ones, and the ideal sample counts of both, from `results/raw_full.json`
(the GPU run) and `results/raw_ideal_full.json` (the ideal jobs)."""
import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from report_full import CLASSES, SHORT, SEEDS, FLOOR  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NETWORKS = {"step-lin": "GNN, noise every step, linear", "init-lin": "GNN, noise at init, linear"}
SOURCES = {"photons p=1": "indistinguishable photons", "photons p=0": "distinguishable photons",
           "coherent": "coherent light", "thermal": "thermal light"}


def exact_order(records, key):
    """First order with a relative gap above the floor in any seed, else None."""
    for N in range(1, 5):
        if any(str(N) in key(r) and key(r)[str(N)]["max_rel_diff"] > FLOOR for r in records):
            return N
    return None


def main():
    full = json.load(open(f"{OUT}/raw_full.json"))
    ideal = json.load(open(f"{OUT}/raw_ideal_full.json"))
    data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
    pairs = [(c, f"{c}_{i}") for c in data for i in range(len(data[c]))]
    rows = {}   # (model, pair) -> separation order or None
    counts = {}  # (model, pair) -> ideal sample count, best cell, all orders
    nmax = {}
    for cat, pair in pairs:
        runs = [full.get(f"brec-full-coherent/{pair}/s{s}/g0.9.json") for s in SEEDS]
        runs = [r for r in runs if isinstance(r, dict)]
        for source in SOURCES:
            rows[(source, pair)] = exact_order(runs, lambda r, s=source: {N: v[s] for N, v in r["orders"].items()})
        if runs:
            counts[("coherent", pair)] = min(r["coherent_samples_by_brightness"]["100.0"][str(r["nmax"])] for r in runs)
            nmax[("photonic", pair)] = runs[0]["nmax"]
        for model in NETWORKS:
            recs = [ideal.get(f"gnn-compare-v4/{pair}/s{s}/{model}-ideal.json") for s in SEEDS]
            recs = [r for r in recs if isinstance(r, dict)]
            rows[(model, pair)] = exact_order(recs, lambda r: r["exact"]) if recs else "missing"
            if recs:
                counts[(model, pair)] = min(r["ideal_samples"][str(r["nmax"])] for r in recs)
                nmax[(model, pair)] = recs[0]["nmax"]
    missing = [p for (m, p), v in rows.items() if v == "missing"]
    lines = [f"## Exact rule on full BREC (relative gap > {FLOOR:g}, any of four cells, orders up to 4 below 65 vertices, 3 above)\n"]
    lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
    lines.append("|---|" + "---|" * (len(CLASSES) + 1))
    for model, label in list(SOURCES.items()) + list(NETWORKS.items()):
        cells = [sum(1 for c, p in pairs if c == cl and rows[(model, p)] not in (None, "missing")) for cl in CLASSES]
        lines.append(f"| {label} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    lines.append("| pairs | " + " | ".join(str(sum(1 for c, _ in pairs if c == cl)) for cl in CLASSES) + f" | {len(pairs)} |")
    lines.append("\n## Pairs the photonic sources and the linear networks separate, exact rule\n")
    lines.append("| Comparison | both | first only | second only | neither |")
    lines.append("|---|---|---|---|---|")
    for a, b in (("coherent", "step-lin"), ("coherent", "init-lin"), ("photons p=1", "step-lin"),
                 ("photons p=0", "step-lin"), ("thermal", "step-lin"), ("step-lin", "init-lin")):
        A = {p for _, p in pairs if rows[(a, p)] not in (None, "missing")}
        Bs = {p for _, p in pairs if rows[(b, p)] not in (None, "missing")}
        lines.append(f"| {SOURCES.get(a, NETWORKS.get(a))} vs {SOURCES.get(b, NETWORKS.get(b))} | "
                     f"{len(A & Bs)} | {len(A - Bs)} | {len(Bs - A)} | {len(pairs) - len(A | Bs)} |")
    only = {p for _, p in pairs if rows[("coherent", p)] not in (None, "missing")
            and all(rows[(m, p)] in (None, "missing") for m in NETWORKS)}
    net_only = {p for _, p in pairs if rows[("coherent", p)] is None
                and any(rows[(m, p)] not in (None, "missing") for m in NETWORKS)}
    lines.append(f"\nCoherent light only: {sorted(only)}\n\nA network only: {sorted(net_only)}\n")
    lines.append("## Separation order, pairs per order: coherent light / step-lin / init-lin\n")
    lines.append("| Class | " + " | ".join(f"order {N}" for N in range(1, 5)) + " | none |")
    lines.append("|---|" + "---|" * 5)
    for cl in CLASSES:
        ps = [p for c, p in pairs if c == cl]
        cells = [" / ".join(str(sum(rows[(m, p)] == N for p in ps)) for m in ("coherent", "step-lin", "init-lin")) for N in range(1, 5)]
        cells.append(" / ".join(str(sum(rows[(m, p)] is None for p in ps)) for m in ("coherent", "step-lin", "init-lin")))
        lines.append(f"| {SHORT[cl]} | " + " | ".join(cells) + " |")
    lines.append("\n## Ideal sample counts, exact gap over the estimated noise, all orders, best cell: median per class (pairs with a finite count)\n")
    lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
    lines.append("|---|" + "---|" * len(CLASSES))
    for model, label in (("coherent", "coherent light, w = 100"), *NETWORKS.items()):
        cells = []
        for cl in CLASSES:
            v = [counts[(model, p)] for c, p in pairs if c == cl and (model, p) in counts and np.isfinite(counts[(model, p)])]
            cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("\n**Ratio of a network's ideal count to coherent light's at w = 100, median over pairs where both are finite (pairs)**\n")
    lines.append("| Network | " + " | ".join(SHORT[c] for c in CLASSES) + " | all |")
    lines.append("|---|" + "---|" * (len(CLASSES) + 1))
    for model, label in NETWORKS.items():
        cells, allr = [], []
        for cl in CLASSES:
            r = [counts[(model, p)] / counts[("coherent", p)] for c, p in pairs if c == cl
                 and (model, p) in counts and ("coherent", p) in counts
                 and np.isfinite(counts[(model, p)]) and np.isfinite(counts[("coherent", p)])]
            allr += r
            cells.append(f"{median(r):.3g} ({len(r)})" if r else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {median(allr):.3g} ({len(allr)}) |")
    lines.append("\n**Pairs within a budget, best cell, all orders**\n")
    budgets = [1e6, 1e8, 1e10, 1e12]
    lines.append("| Model | " + " | ".join(f"T <= {b:.0e}" for b in budgets) + " |")
    lines.append("|---|" + "---|" * len(budgets))
    for model, label in (("coherent", "coherent light, w = 100"), *NETWORKS.items()):
        cells = [sum(1 for _, p in pairs if (model, p) in counts and counts[(model, p)] <= b) for b in budgets]
        lines.append(f"| {label} | " + " | ".join(map(str, cells)) + " |")
    hot_path = f"{OUT}/raw_gnn_hotelling.json"
    if os.path.exists(hot_path):
        hot = json.load(open(hot_path))
        lines.append("\n## BREC's test on the exact profiles of the linear networks (seed 100), verbatim / scaled to the rounding noise\n")
        lines.append("| Network | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
        lines.append("|---|" + "---|" * (len(CLASSES) + 1))
        for model, label in NETWORKS.items():
            verb = {p for (c, p) in pairs if isinstance(hot.get(f"gnn-compare-v4/{p}/s100/{model}-hotelling-exact.json"), dict)
                    and any(v["separated_verbatim"] for v in hot[f"gnn-compare-v4/{p}/s100/{model}-hotelling-exact.json"]["orders"].values())}
            scal = {p for (c, p) in pairs if isinstance(hot.get(f"gnn-compare-v4/{p}/s100/{model}-hotelling-exact.json"), dict)
                    and any(v["separated_scaled"] for v in hot[f"gnn-compare-v4/{p}/s100/{model}-hotelling-exact.json"]["orders"].values())}
            have = sum(1 for (c, p) in pairs if isinstance(hot.get(f"gnn-compare-v4/{p}/s100/{model}-hotelling-exact.json"), dict))
            cells = [f"{sum(1 for c, p in pairs if c == cl and p in verb)} / {sum(1 for c, p in pairs if c == cl and p in scal)}" for cl in CLASSES]
            lines.append(f"| {label} ({have} pairs) | " + " | ".join(cells) + f" | {len(verb)} / {len(scal)} |")
    if missing:
        lines.append(f"\nMissing ideal files: {len(missing)} pair-models: {sorted(set(missing))[:20]}")
    open(f"{OUT}/tables_ideal.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
