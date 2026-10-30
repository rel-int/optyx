"""Tables of the squeezed-light run against the stored exact-noise run.

    python report_squeezed.py RAW_SQUEEZED RAW_NOISE [RAW_REAL] [RAW_FULL]

RAW_* are decompressed result files (results/raw_*.json.gz); RAW_FULL, the
reach run of the other sources (order 4 up to 96 vertices, 3 above), adds
their separation orders as the columns marked "(reach run)".  A pair is
separated by a source when its largest relative gap between sorted rates
exceeds 1e-12 at some order, in at least one cell; squeezed light counts at
any brightness of the grid.  Ideal counts are cumulative over the orders,
at the largest order run, in the best cell, for the pairs a source separates.
"""
import json
import sys
from statistics import median

import numpy as np

THRESHOLD = 1e-12
CLASSES = ("Basic", "Regular", "Extension", "CFI", "Strongly_regular", "4-vertex_condition", "Distance_regular")


def order_of(gaps):
    """Smallest order with a gap above the threshold, or None."""
    sep = [int(N) for N, g in gaps.items() if g > THRESHOLD]
    return min(sep) if sep else None


def best(orders):
    orders = [o for o in orders if o is not None]
    return min(orders) if orders else None


def collect(sq_raw, noise_raw, real_raw, full_raw=None):
    pairs = {}
    for r in sq_raw.values():
        p = pairs.setdefault(r["pair"], {"n": r["n"], "nmax": r["nmax"], "order": {}, "count": {}})
        o = p["order"]
        o.setdefault("squeezed", []).append(best([order_of(g) for g in r["gaps"]["squeezed"].values()]))
        for j in range(r["nmax"] + 1):
            gaps = {N: rows[j] for N, rows in r["component_gaps"].items() if j < len(rows)}
            o.setdefault(f"squeezed m_{j}", []).append(order_of(gaps))
        o.setdefault("thermal (this run)", []).append(order_of(r["gaps"]["thermal"]["1.0"]))
        o.setdefault("gaussian real (this run)", []).append(order_of(r["gaps"]["gaussian real"]["1.0"]))
        for kind, by_w in r["ideal"].items():
            for w, by_N in by_w.items():
                d = p["count"].setdefault(kind, {})
                d[float(w)] = min(d.get(float(w), np.inf), by_N[str(r["nmax"])])
    for r in noise_raw.values():
        if not isinstance(r, dict) or r["pair"] not in pairs or r["nmax"] != pairs[r["pair"]]["nmax"]:
            continue
        p = pairs[r["pair"]]
        for s in ("photons p=1", "coherent", "thermal"):
            p["order"].setdefault(s, []).append(order_of(r["gaps"][s]))
            for w, by_N in r["ideal"][s].items():
                d = p["count"].setdefault(s, {})
                d[float(w)] = min(d.get(float(w), np.inf), by_N[str(r["nmax"])])
    for r in (real_raw or {}).values():
        if isinstance(r, dict) and r["pair"] in pairs and r["nmax"] == pairs[r["pair"]]["nmax"]:
            pairs[r["pair"]]["order"].setdefault("gaussian real (stored)", []).append(order_of(r["gaps"]["gaussian real"]))
    for r in (full_raw or {}).values():
        if isinstance(r, dict) and "orders" in r and r["pair"] in pairs:
            for s in ("photons p=1", "coherent", "thermal"):
                gaps = {N: by_source[s]["max_rel_diff"] for N, by_source in r["orders"].items()}
                pairs[r["pair"]]["order"].setdefault(f"{s} (reach run)", []).append(order_of(gaps))
    for p in pairs.values():
        p["order"] = {s: best(v) for s, v in p["order"].items()}
        for s, d in p["count"].items():
            if p["order"].get(s if s != "gaussian real" else "gaussian real (this run)") is None:
                p["count"][s] = {w: np.inf for w in d}
    return pairs


def class_of(pair):
    return pair.rsplit("_", 1)[0]


def tables(pairs):
    order = sorted(pairs, key=lambda p: (CLASSES.index(class_of(p)), pairs[p]["n"], p))
    cols = ["photons p=1", "coherent", "thermal", "thermal (this run)", "gaussian real (stored)",
            "gaussian real (this run)", "squeezed", "squeezed m_0", "squeezed m_1", "squeezed m_2"]
    out = ["## Separation order per pair (best of four cells, '-' = not separated up to the largest order)\n",
           "| pair | n | N max | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for p in order:
        o = pairs[p]["order"]
        out.append(f"| {p} | {pairs[p]['n']} | {pairs[p]['nmax']} | " + " | ".join(
            str(o.get(c)) if o.get(c) is not None else "-" for c in cols) + " |")
    out.append("\n| pairs separated | | | " + " | ".join(
        str(sum(pairs[p]["order"].get(c) is not None for p in order)) for c in cols) + " |")

    grid = (0.02, 0.05, 0.2, 1.0, 2.0, 5.0, 20.0, 100.0)
    sources = ("squeezed", "thermal", "gaussian real", "coherent", "photons p=1")
    out.append("\n## Ideal samples per network, best cell, all orders (inf = not separated)\n")
    out.append("| pair | source | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    out.append("|---|---|" + "---|" * len(grid))
    for p in order:
        for s in sources:
            d = pairs[p]["count"].get(s, {})
            out.append(f"| {p} | {s} | " + " | ".join(f"{d[w]:.2g}" if w in d else "" for w in grid) + " |")

    out.append("\n## Squeezed light relative to each source at equal brightness, median over the pairs both separate\n")
    out.append("Samples with squeezed light over samples with the other source; below 1 squeezed light wins.\n")
    out.append("| against | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    out.append("|---|" + "---|" * len(grid))
    for s in ("thermal", "gaussian real", "coherent", "photons p=1"):
        cells = []
        for w in grid:
            v = [pairs[p]["count"]["squeezed"][w] / pairs[p]["count"][s][w] for p in order
                 if w in pairs[p]["count"].get(s, {}) and np.isfinite(pairs[p]["count"][s][w])
                 and np.isfinite(pairs[p]["count"]["squeezed"][w])]
            cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
        out.append(f"| {s} | " + " | ".join(cells) + " |")

    out.append("\n## Each source at its best brightness, relative to single photons at w = 1, per pair\n")
    out.append("| pair | " + " | ".join(sources) + " |")
    out.append("|---|" + "---|" * len(sources))
    for p in order:
        c = pairs[p]["count"]
        ref = c["photons p=1"].get(1.0, np.inf)
        cells = []
        for s in sources:
            ws = [w for w in c.get(s, {}) if s != "photons p=1" or w <= 1.0]
            b = min((c[s][w] for w in ws), default=np.inf)
            cells.append(f"{b / ref:.2g}" if np.isfinite(b) and np.isfinite(ref) else "-")
        out.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def summary(pairs):
    """Per class: pairs separated, median ideal counts against brightness,
    squeezed light relative to each source, and pairs within device-time
    budgets at one tick per nanosecond."""
    cols = ["photons p=1", "coherent", "thermal", "thermal (this run)", "gaussian real (stored)",
            "gaussian real (this run)", "squeezed"]
    if any("thermal (reach run)" in p["order"] for p in pairs.values()):
        cols = ["photons p=1 (reach run)", "coherent (reach run)", "thermal (reach run)", "thermal (this run)",
                "gaussian real (this run)", "squeezed"]
    out = ["## Pairs separated per class (best of four cells, exact rule)\n",
           "| class | pairs | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for c in CLASSES + ("Total",):
        ps = [p for p in pairs if c == "Total" or class_of(p) == c]
        out.append(f"| {c} | {len(ps)} | " + " | ".join(
            str(sum(pairs[p]["order"].get(k) is not None for p in ps)) for k in cols) + " |")
    ref = "thermal (reach run)" if "thermal (reach run)" in cols else "thermal"
    agree = sum(pairs[p]["order"].get("squeezed") == pairs[p]["order"].get(ref) for p in pairs)
    out.append(f"\nSqueezed light separates a pair at the same order as the stored {ref} on {agree} of {len(pairs)} pairs.")

    grid = (0.02, 0.05, 0.2, 1.0, 5.0, 100.0)
    sources = ("squeezed", "thermal", "gaussian real", "coherent", "photons p=1")
    out.append("\n## Median ideal samples per class over the pairs each source separates\n")
    out.append("| class | source | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    out.append("|---|---|" + "---|" * len(grid))
    for c in CLASSES:
        ps = [p for p in pairs if class_of(p) == c]
        for src in sources:
            cells = []
            for w in grid:
                v = [pairs[p]["count"][src][w] for p in ps if w in pairs[p]["count"].get(src, {})
                     and np.isfinite(pairs[p]["count"][src][w])]
                cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
            out.append(f"| {c} | {src} | " + " | ".join(cells) + " |")

    out.append("\n## Squeezed light over each source at equal brightness, median per class over the pairs both separate\n")
    for other in ("thermal", "coherent", "photons p=1"):
        out.append(f"\n**against {other}**\n")
        out.append("| class | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
        out.append("|---|" + "---|" * len(grid))
        for c in CLASSES + ("Total",):
            ps = [p for p in pairs if c == "Total" or class_of(p) == c]
            cells = []
            for w in grid:
                v = [pairs[p]["count"]["squeezed"][w] / pairs[p]["count"][other][w] for p in ps
                     if w in pairs[p]["count"].get(other, {}) and np.isfinite(pairs[p]["count"][other][w])
                     and np.isfinite(pairs[p]["count"]["squeezed"].get(w, np.inf))]
                cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
            out.append(f"| {c} | " + " | ".join(cells) + " |")

    out.append("\n## Pairs within a device-time budget at one tick per nanosecond\n")
    out.append("| source | w | 1e6 samples | 1e9 | an hour | a day |")
    out.append("|---|---|---|---|---|---|")
    for src in sources:
        for w in grid:
            vals = [pairs[p]["count"].get(src, {}).get(w) for p in pairs]
            if all(v is None for v in vals):
                continue
            vals = [np.inf if v is None else v for v in vals]
            out.append(f"| {src} | {w:g} | " + " | ".join(str(sum(v <= b for v in vals)) for b in (1e6, 1e9, 3.6e12, 8.64e13)) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    sq_raw = json.load(open(sys.argv[1]))
    noise_raw = json.load(open(sys.argv[2]))
    real_raw = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    full_raw = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    pairs = collect(sq_raw, noise_raw, real_raw, full_raw)
    print(summary(pairs) if len(pairs) > 50 else tables(pairs))
