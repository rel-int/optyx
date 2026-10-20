"""Aggregate the results of the two experiments into tables and a figure.

    python report.py fetch     # pull every small result file from the volume
    python report.py tables    # markdown tables + results.json for the page
"""

import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from launch import GAMMAS, SEEDS, pairs_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results"
CLASSES = ["Basic", "Regular", "Strongly_regular", "Extension", "CFI",
           "4-vertex_condition", "Distance_regular"]
SHORT = {"Basic": "Basic", "Regular": "Regular", "Strongly_regular": "Str. regular",
         "Extension": "Extension", "CFI": "CFI", "4-vertex_condition": "4-vertex",
         "Distance_regular": "Dist. regular"}
FLOOR = 1e-12


def fetch():
    import modal
    read_json = modal.Function.from_name("brec-photonic", "read_json")
    paths = []
    for pair in pairs_of(None):
        for seed in SEEDS:
            paths.append(f"column-pilot/{pair}/s{seed}/test.json")
            for gamma in GAMMAS:
                paths.append(f"loss-sweep/{pair}/s{seed}/g{gamma}.json")
    for seed in SEEDS:
        paths.append(f"loss-sweep/Distance_regular_3/s{seed}/g0.9.json")
    results = {}
    for start in range(0, len(paths), 150):
        results.update(read_json.remote(paths[start:start + 150]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/raw.json", "w"))
    missing = [p for p, v in results.items() if isinstance(v, str)]
    print(f"{len(results) - len(missing)} files, {len(missing)} missing")
    return results


def category_of(pair):
    return pair.rsplit("_", 1)[0]


def column_tables(raw, pairs):
    """Pairs separated per category under BREC's test, both regularisations,
    and under the exact test of the draft; plus a per-pair table."""
    rows, per_pair = {}, []
    for pair in pairs:
        n = bp.adjacency(pairs[pair][0]).shape[0]
        tests = [raw.get(f"column-pilot/{pair}/s{seed}/test.json") for seed in SEEDS]
        tests = [t for t in tests if isinstance(t, dict)]
        if not tests:
            continue
        orders = sorted(int(N) for N in tests[0]["orders"])
        record = {"pair": pair, "n": n, "seeds": len(tests)}
        for source in bp.SOURCES:
            for variant in ("verbatim", "scaled", "exact"):
                sep_order = None
                unreliable = 0
                for N in orders:
                    for t in tests:
                        cell = t["orders"][str(N)][source]
                        if variant == "exact":
                            ok = cell["exact"]["separated_1e-12"]
                        else:
                            ok = cell[variant]["separated"]
                            unreliable += not cell[variant]["reliable"]
                        if ok and sep_order is None:
                            sep_order = N
                record[(source, variant)] = sep_order
                record[(source, variant, "unreliable")] = unreliable
            record[(source, "max_rel_diff")] = max(
                t["orders"][str(N)][source]["exact"]["max_rel_diff"]
                for t in tests for N in orders)
            record[(source, "floor")] = max(
                t["orders"][str(N)][source]["exact"]["relabel_floor"]
                for t in tests for N in orders)
            record[(source, "T2_pair")] = max(
                t["orders"][str(orders[-1])][source]["verbatim"]["T2_pair"] for t in tests)
            record[(source, "T2_rel")] = max(
                t["orders"][str(orders[-1])][source]["verbatim"]["T2_reliability"] for t in tests)
        per_pair.append(record)
        rows.setdefault(category_of(pair), []).append(record)
    lines = ["| Source | Test | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |",
             "|---|---|" + "---|" * (len(CLASSES) + 1)]
    for source in bp.SOURCES:
        for variant, label in (("verbatim", "BREC, eps = 1e-7"),
                               ("scaled", "BREC, eps at the noise"),
                               ("exact", "exact, gap > 1e-12")):
            counts = [sum(r[(source, variant)] is not None for r in rows.get(c, []))
                      for c in CLASSES]
            lines.append(f"| {source} | {label} | " + " | ".join(map(str, counts))
                         + f" | {sum(counts)} |")
    totals = ["| pairs run | | " + " | ".join(str(len(rows.get(c, []))) for c in CLASSES)
              + f" | {sum(len(v) for v in rows.values())} |"]
    unreliable = {source: sum(r[(source, "scaled", "unreliable")] for r in per_pair)
                  for source in bp.SOURCES}
    detail = ["| Pair | n | seeds | " + " | ".join(
        f"{s}: order (BREC / exact), max rel. gap" for s in bp.SOURCES) + " | relabel floor |",
        "|---|---|---|" + "---|" * (len(bp.SOURCES) + 1)]
    for r in per_pair:
        cells = []
        for s in bp.SOURCES:
            cells.append(f"{r[(s, 'verbatim')]} / {r[(s, 'exact')]}, {r[(s, 'max_rel_diff')]:.1e}")
        detail.append(f"| {r['pair']} | {r['n']} | {r['seeds']} | " + " | ".join(cells)
                      + f" | {max(r[(s, 'floor')] for s in bp.SOURCES):.0e} |")
    return "\n".join(lines + totals), "\n".join(detail), unreliable, per_pair


def sweep_tables(raw, pairs):
    """Median ideal sample count (cumulative over the orders) against the
    loss, per category and source, and the separation order against loss."""
    table = {}   # (category, gamma, source) -> list of samples
    orders = {}  # (category, gamma, source) -> list of separation orders
    rel = {}     # (category, gamma, source) -> list of max relative gaps
    seconds = []
    for pair in pairs:
        cat = category_of(pair)
        for seed in SEEDS:
            for gamma in GAMMAS:
                r = raw.get(f"loss-sweep/{pair}/s{seed}/g{gamma}.json")
                if not isinstance(r, dict):
                    continue
                seconds.append((pair, gamma, r["seconds"], r["slots"]["G"]))
                nmax = max(int(N) for N in r["orders"])
                for source in bp.SOURCES:
                    top = r["orders"][str(nmax)][source]
                    table.setdefault((cat, gamma, source), []).append(
                        top["ideal_samples_cumulative"])
                    sep = next((int(N) for N in sorted(r["orders"], key=int)
                                if r["orders"][N][source]["max_rel_diff"] > FLOOR), None)
                    orders.setdefault((cat, gamma, source), []).append(sep)
                    rel.setdefault((cat, gamma, source), []).append(
                        max(r["orders"][N][source]["max_rel_diff"] for N in r["orders"]))
    lines = ["| Class | Source | " + " | ".join(f"gamma = {g}" for g in GAMMAS) + " |",
             "|---|---|" + "---|" * len(GAMMAS)]
    for cat in CLASSES:
        for source in bp.SOURCES:
            cells = []
            for gamma in GAMMAS:
                vals = table.get((cat, gamma, source), [])
                finite = [v for v in vals if np.isfinite(v)]
                cells.append(f"{median(finite):.2g}" if finite else "-")
            lines.append(f"| {SHORT[cat]} | {source} | " + " | ".join(cells) + " |")
    ratio = ["| Class | Source | " + " | ".join(f"gamma = {g}" for g in GAMMAS) + " |",
             "|---|---|" + "---|" * len(GAMMAS)]
    for cat in CLASSES:
        for source in bp.SOURCES[1:]:
            cells = []
            for gamma in GAMMAS:
                a = table.get((cat, gamma, "photons p=1"), [])
                b = table.get((cat, gamma, source), [])
                rs = [y / x for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
                cells.append(f"{median(rs):.2f}" if rs else "-")
            ratio.append(f"| {SHORT[cat]} | {source} | " + " | ".join(cells) + " |")
    sep_lines = ["| Class | Source | " + " | ".join(f"gamma = {g}" for g in GAMMAS) + " |",
                 "|---|---|" + "---|" * len(GAMMAS)]
    for cat in CLASSES:
        for source in bp.SOURCES:
            cells = []
            for gamma in GAMMAS:
                vals = orders.get((cat, gamma, source), [])
                if not vals:
                    cells.append("-")
                    continue
                counts = {N: vals.count(N) for N in sorted(set(vals), key=lambda x: (x is None, x))}
                cells.append(", ".join(f"{'none' if N is None else N}: {c}" for N, c in counts.items()))
            sep_lines.append(f"| {SHORT[cat]} | {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines), "\n".join(ratio), "\n".join(sep_lines), table, seconds


def figure(table, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    styles = {"photons p=1": ("C0", "-"), "photons p=0": ("C1", "--"),
              "coherent": ("C2", "-."), "thermal": ("C3", ":")}
    for ax, cat in zip(axes.ravel(), CLASSES):
        for source, (color, style) in styles.items():
            ys = []
            for gamma in GAMMAS:
                vals = [v for v in table.get((cat, gamma, source), []) if np.isfinite(v)]
                ys.append(median(vals) if vals else np.nan)
            ax.plot(GAMMAS, ys, style, color=color, marker="o", label=source)
        ax.set_yscale("log")
        ax.set_title(SHORT[cat])
        ax.set_xlabel("transmissivity gamma")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("ideal samples per network (median)")
    axes[1, 0].set_ylabel("ideal samples per network (median)")
    ax = axes[1, 3]
    sizes = sorted({bp.adjacency(g).shape[0] for g, _ in pairs.values()})
    for n in (10, 16, 30, 63):
        ax.plot(GAMMAS, [bp.W * n / (1 - g) for g in GAMMAS], marker="s", label=f"n = {n}")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_title("mean occupation mu = w n / (1 - gamma)")
    ax.set_xlabel("transmissivity gamma")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_sweep.svg")
    fig.savefig(f"{OUT}/loss_sweep.png", dpi=110)


def main():
    pairs = pairs_of(None)
    if sys.argv[1] == "fetch":
        fetch()
        return
    raw = json.load(open(f"{OUT}/raw.json"))
    summary, detail, unreliable, per_pair = column_tables(raw, pairs)
    samples, ratio, seps, table, seconds = sweep_tables(raw, pairs)
    figure(table, pairs)
    with open(f"{OUT}/tables.md", "w") as f:
        f.write("## Column: pairs separated\n\n" + summary + "\n\n")
        f.write(f"Unreliable (seed, order) cases under the noise-scaled test: {unreliable}\n\n")
        f.write("## Column: per pair\n\n" + detail + "\n\n")
        f.write("## Sweep: median ideal samples\n\n" + samples + "\n\n")
        f.write("## Sweep: samples relative to indistinguishable photons\n\n" + ratio + "\n\n")
        f.write("## Sweep: separation order (count of pair-seed cases)\n\n" + seps + "\n\n")
        if seconds:
            by_gamma = {}
            for pair, gamma, s, slots in seconds:
                by_gamma.setdefault(gamma, []).append(s)
            f.write("## Sweep timings (seconds per pair and seed, both graphs)\n\n")
            f.write("| gamma | median | max |\n|---|---|---|\n")
            for gamma in GAMMAS:
                if gamma in by_gamma:
                    f.write(f"| {gamma} | {median(by_gamma[gamma]):.0f} | {max(by_gamma[gamma]):.0f} |\n")
    order5 = {p: v for p, v in raw.items() if "Distance_regular_3" in p and isinstance(v, dict)}
    with open(f"{OUT}/tables.md", "a") as f:
        f.write("## Order 5 on Distance_regular_3 (38 vertices), gamma = 0.9\n\n")
        f.write("| seed | source | " + " | ".join(f"order {N}: max rel. gap" for N in range(1, 6)) + " | seconds |\n")
        f.write("|---|---|" + "---|" * 6 + "\n")
        for p, v in sorted(order5.items()):
            for source in bp.SOURCES:
                cells = [f"{v['orders'][str(N)][source]['max_rel_diff']:.1e}" if str(N) in v["orders"] else "-"
                         for N in range(1, 6)]
                f.write(f"| {v['seed']} | {source} | " + " | ".join(cells) + f" | {v['seconds']:.0f} |\n")
    json.dump({"column": [{str(k): v for k, v in r.items()} for r in per_pair],
               "order5": order5,
               "sweep": {f"{c}|{g}|{s}": v for (c, g, s), v in table.items()},
               "timings": seconds},
              open(f"{OUT}/results.json", "w"))
    print(open(f"{OUT}/tables.md").read())


if __name__ == "__main__":
    main()
