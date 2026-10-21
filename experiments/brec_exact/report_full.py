"""Aggregate the full-BREC coherent-light run.

    python report_full.py fetch
    python report_full.py tables
"""

import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-user/2258d4a3-8088-52d8-9954-5113d3147de7/scratchpad/work")
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW, BRIGHTNESSES  # noqa: E402
from report import CLASSES, SHORT, FLOOR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results"
SEEDS = (100, 101, 102, 103)
TICK = 1e9  # ticks per second of a device
BUDGETS = (("an hour", 3600 * TICK), ("a day", 86400 * TICK), ("a year", 365 * 86400 * TICK))
PAPER = {"Basic": 60, "Regular": 139, "Extension": 100, "CFI": 71}  # the paper's photons row


def all_pairs():
    data = bp.load_brec(BREC_RAW)
    return {f"{c}_{i}": pair for c in data for i, pair in enumerate(data[c])}


def fetch():
    import base64
    import gzip
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    pairs = list(all_pairs())
    results = {}
    for start in range(0, len(pairs), 60):
        blob = read_many.remote("brec-full-coherent", pairs[start:start + 60],
                                ["g0.9.json", "hotelling.json"])
        results.update(json.loads(gzip.decompress(base64.b64decode(blob))))
        print(f"{len(results)} files after {start + 60} pairs", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/raw_full.json", "w"))
    missing = [p for p, v in results.items() if isinstance(v, str)]
    print(f"{len(results) - len(missing)} files, {len(missing)} missing")
    if missing:
        print(missing[:20])


def tables():
    raw = json.load(open(f"{OUT}/raw_full.json"))
    pairs = all_pairs()
    per_class = {}
    order4 = {}
    seconds = []
    for pair in pairs:
        cat = pair.rsplit("_", 1)[0]
        runs = [raw.get(f"brec-full-coherent/{pair}/s{seed}/g0.9.json") for seed in SEEDS]
        runs = [r for r in runs if isinstance(r, dict)]
        if not runs:
            continue
        rec = {"pair": pair, "n": runs[0]["n"], "nmax": runs[0]["nmax"], "seeds": len(runs)}
        seconds.append((runs[0]["n"], runs[0]["nmax"], median(r["seconds"] for r in runs)))
        order4.setdefault(cat, []).append(rec["nmax"] == 4)
        tests = [raw.get(f"brec-full-coherent/{pair}/s{seed}/hotelling.json") for seed in SEEDS]
        tests = [t for t in tests if isinstance(t, dict)]
        for source in bp.SOURCES:
            sep = None
            for N in range(1, rec["nmax"] + 1):
                if any(r["orders"][str(N)][source]["max_rel_diff"] > FLOOR for r in runs):
                    sep = N
                    break
            rec[source] = sep
            for variant in ("verbatim", "scaled"):
                sep = None
                for N in range(1, rec["nmax"] + 1):
                    if any(t["orders"][str(N)][source][f"separated_{variant}"] for t in tests):
                        sep = N
                        break
                rec[(source, variant)] = sep
            rec[(source, "projected_gap")] = max(
                (t["orders"][str(t["nmax"])][source]["projected_gap"] for t in tests), default=None)
        # coherent light: samples per network at every brightness, best cell,
        # cumulated over all orders
        for w in BRIGHTNESSES:
            vals = [r["coherent_samples_by_brightness"][str(w)][str(r["nmax"])] for r in runs]
            rec[f"T@{w}"] = min(vals)
        rec["T_poisson"] = min(r["orders"][str(r["nmax"])]["coherent"]["ideal_samples_poisson"]
                               for r in runs)
        per_class.setdefault(cat, []).append(rec)
    lines = []
    lines.append("## Pairs separated (exact test, relative gap > 1e-12, any of four cells)\n")
    lines.append("| Source | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
    lines.append("|---|" + "---|" * (len(CLASSES) + 1))
    for source in bp.SOURCES:
        counts = [sum(r[source] is not None for r in per_class.get(c, [])) for c in CLASSES]
        lines.append(f"| {source} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    lines.append("| pairs run | " + " | ".join(str(len(per_class.get(c, []))) for c in CLASSES)
                 + f" | {sum(len(v) for v in per_class.values())} |")
    lines.append("| at order 4 | " + " | ".join(str(sum(order4.get(c, []))) for c in CLASSES)
                 + f" | {sum(sum(v) for v in order4.values())} |")
    lines.append("\n## Pairs separated under BREC's test computed from the profiles\n")
    lines.append("| Source | Test | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
    lines.append("|---|---|" + "---|" * (len(CLASSES) + 1))
    for source in bp.SOURCES:
        for variant, label in (("verbatim", "eps = 1e-7"), ("scaled", "eps at the pilot's rounding noise")):
            counts = [sum(r.get((source, variant)) is not None for r in per_class.get(c, [])) for c in CLASSES]
            lines.append(f"| {source} | {label} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    lines.append("\n## Separation order (pairs per order), indistinguishable photons / coherent\n")
    lines.append("| Class | " + " | ".join(f"order {N}" for N in range(1, 5)) + " | none |")
    lines.append("|---|" + "---|" * 5)
    for c in CLASSES:
        recs = per_class.get(c, [])
        cells = [f"{sum(r['photons p=1'] == N for r in recs)} / {sum(r['coherent'] == N for r in recs)}"
                 for N in range(1, 5)]
        cells.append(f"{sum(r['photons p=1'] is None for r in recs)} / {sum(r['coherent'] is None for r in recs)}")
        lines.append(f"| {SHORT[c]} | " + " | ".join(cells) + " |")
    lines.append("\n## Coherent light: median samples per network against brightness (best of four cells)\n")
    lines.append("| Class | top order only, Poisson noise, w = 0.05 | " + " | ".join(f"all orders, w = {w}" for w in BRIGHTNESSES) + " |")
    lines.append("|---|---|" + "---|" * len(BRIGHTNESSES))
    for c in CLASSES:
        recs = [r for r in per_class.get(c, []) if r["coherent"] is not None]
        if not recs:
            continue
        cells = [f"{median(r['T_poisson'] for r in recs):.2g}"]
        for w in BRIGHTNESSES:
            cells.append(f"{median(r[f'T@{w}'] for r in recs):.2g}")
        lines.append(f"| {SHORT[c]} ({len(recs)}) | " + " | ".join(cells) + " |")
    lines.append("\n## Coherent light: pairs separated within a device-time budget at 1 GHz\n")
    lines.append("| Brightness | " + " | ".join(name for name, _ in BUDGETS) + " |")
    lines.append("|---|" + "---|" * len(BUDGETS))
    for w in BRIGHTNESSES:
        cells = []
        for name, ticks in BUDGETS:
            cells.append(str(sum(r[f"T@{w}"] <= ticks for recs in per_class.values() for r in recs)))
        lines.append(f"| w = {w} | " + " | ".join(cells) + " |")
    lines.append("\n## Coherent light: gain in samples from brightness, median per class\n")
    lines.append("| Class | " + " | ".join(f"w = {w} vs 0.05" for w in BRIGHTNESSES[1:]) + " |")
    lines.append("|---|" + "---|" * (len(BRIGHTNESSES) - 1))
    for c in CLASSES:
        recs = [r for r in per_class.get(c, []) if r["coherent"] is not None]
        if not recs:
            continue
        cells = [f"{median(r['T@0.05'] / r[f'T@{w}'] for r in recs):.3g}" for w in BRIGHTNESSES[1:]]
        lines.append(f"| {SHORT[c]} | " + " | ".join(cells) + " |")
    lines.append("\n## GPU time per pair and cell (seconds, median over cells)\n")
    lines.append("| size | order | pairs | median | max |\n|---|---|---|---|---|")
    for lo, hi in ((0, 20), (20, 40), (40, 64), (64, 110), (110, 200)):
        sel = [s for n, nmax, s in seconds if lo <= n < hi]
        if sel:
            nm = {nmax for n, nmax, s in seconds if lo <= n < hi}
            lines.append(f"| {lo}-{hi - 1} | {','.join(map(str, sorted(nm)))} | {len(sel)} | {median(sel):.0f} | {max(sel):.0f} |")
    text = "\n".join(lines)
    open(f"{OUT}/tables_full.md", "w").write(text)
    json.dump({"per_class": {c: [{str(k): v for k, v in r.items()} for r in recs] for c, recs in per_class.items()},
               "seconds": seconds}, open(f"{OUT}/results_full.json", "w"))
    print(text)


if __name__ == "__main__":
    {"fetch": fetch, "tables": tables}[sys.argv[1]]()
