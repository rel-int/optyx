"""Fetch and tabulate the phase-locked uniform source, against the circular
sources of the exact-noise run.

    python report_real.py fetch
    python report_real.py tables
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
from noise_app import REAL_EXPERIMENT, REAL_BRIGHTNESSES  # noqa: E402
from report_full import CLASSES, SHORT  # noqa: E402
from report_noise import best_counts  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
THRESHOLD = 1e-12


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    data = bp.load_brec(BREC_RAW)
    pairs = [f"{c}_{i}" for c in data for i in range(len(data[c]))]
    chunks = [pairs[i:i + 10] for i in range(0, len(pairs), 10)]
    out = {}
    for blob in read_many.starmap([(REAL_EXPERIMENT, ch, ["g0.9-N4.json", "g0.9-N3.json", "g0.9-N2.json"]) for ch in chunks]):
        out.update(json.loads(gzip.decompress(base64.b64decode(blob))))
    json.dump(out, open(f"{OUT}/raw_real.json", "w"))
    print(len(out), "files")


def tables():
    raw = json.load(open(f"{OUT}/raw_real.json"))
    runs = [r for r in raw.values() if isinstance(r, dict)]
    noise = json.load(open(f"{OUT}/raw_noise.json"))
    circular, nmax = best_counts(noise)
    pairs = sorted({r["pair"] for r in runs})
    lines = [f"## Phase-locked sources through the chip: {len(runs)} pair-cell files, {len(pairs)} pairs\n"]
    # reach: a pair is separated by a source if any cell has a gap above the threshold at any order
    sources = ["uniform", "rademacher", "gaussian real"]
    lines.append("## Pairs separated (exact rule, any cell, orders up to the run's), with the circular sources of the exact-noise run\n")
    lines.append("| Class | pairs | " + " | ".join(sources) + " | coherent | thermal | photons p=1 |")
    lines.append("|---|---|" + "---|" * (len(sources) + 3))

    def separated(pair, source, orders=None):
        rs = [r for r in runs if r["pair"] == pair]
        return any(v > THRESHOLD for r in rs for N, v in r["gaps"][source].items() if orders is None or int(N) <= orders)

    def separated_circular(pair, source):
        rs = [r for r in noise.values() if isinstance(r, dict) and r["pair"] == pair]
        return any(v > THRESHOLD for r in rs for v in r["gaps"][source].values())

    totals = {s: 0 for s in sources + ["coherent", "thermal", "photons p=1"]}
    for c in CLASSES + ["all"]:
        ps = pairs if c == "all" else [p for p in pairs if p.rsplit("_", 1)[0] == c]
        row = []
        for s in sources:
            k = sum(separated(p, s) for p in ps)
            row.append(str(k))
        for s in ("coherent", "thermal", "photons p=1"):
            row.append(str(sum(separated_circular(p, s) for p in ps)))
        lines.append(f"| {SHORT.get(c, c)} | {len(ps)} | " + " | ".join(row) + " |")
    lines.append("\n## Separation order of the uniform source against coherent light, pairs by (uniform order, coherent order)\n")

    def order(pair, source, circ=False):
        rs = [r for r in (noise.values() if circ else runs) if isinstance(r, dict) and r["pair"] == pair]
        found = [int(N) for r in rs for N, v in r["gaps"][source].items() if v > THRESHOLD]
        return min(found) if found else None
    from collections import Counter
    cnt = Counter((order(p, "uniform"), order(p, "coherent", True)) for p in pairs)
    lines.append("| uniform \\ coherent | " + " | ".join(str(o) for o in (1, 2, 3, 4, None)) + " |")
    lines.append("|---|" + "---|" * 5)
    for ou in (1, 2, 3, 4, None):
        lines.append(f"| {ou} | " + " | ".join(str(cnt.get((ou, oc), 0)) for oc in (1, 2, 3, 4, None)) + " |")
    # ideal counts
    real = {}
    for r in runs:
        if r["nmax"] != nmax[r["pair"]]:
            continue
        for w, by_order in r["ideal"]["uniform"].items():
            d = real.setdefault(r["pair"], {})
            d[float(w)] = min(d.get(float(w), np.inf), by_order[str(r["nmax"])])
    grid = list(REAL_BRIGHTNESSES)
    lines.append("\n## Median ideal samples per class against brightness, best cell, all orders: uniform against coherent\n")
    lines.append("| Class | Source | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    lines.append("|---|---|" + "---|" * len(grid))
    for c in CLASSES:
        ps = [p for p in pairs if p.rsplit("_", 1)[0] == c]
        for label, table in (("uniform", real), ("coherent", {p: circular[p]["coherent"] for p in circular}),
                             ("photons p=1", {p: circular[p]["photons p=1"] for p in circular})):
            cells = []
            for w in grid:
                v = [table[p][w] for p in ps if p in table and w in table[p] and np.isfinite(table[p][w])]
                cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
            lines.append(f"| {SHORT[c]} | {label} | " + " | ".join(cells) + " |")
    lines.append("\n## Uniform relative to coherent light at equal brightness, median per class (pairs both separate)\n")
    lines.append("| Class | " + " | ".join(f"w = {w:g}" for w in grid) + " |")
    lines.append("|---|" + "---|" * len(grid))
    for c in CLASSES + ["all"]:
        ps = pairs if c == "all" else [p for p in pairs if p.rsplit("_", 1)[0] == c]
        cells = []
        for w in grid:
            r = [real[p][w] / circular[p]["coherent"][w] for p in ps if p in real and p in circular
                 and np.isfinite(real[p].get(w, np.inf)) and np.isfinite(circular[p]["coherent"].get(w, np.inf))]
            cells.append(f"{median(r):.2f} ({len(r)})" if r else "-")
        lines.append(f"| {SHORT.get(c, c)} | " + " | ".join(cells) + " |")
    lines.append("\n## Pairs within a device-time budget at 1 GHz, uniform source\n")
    lines.append("| w | an hour | a day | a year |")
    lines.append("|---|---|---|---|")
    for w in grid:
        vals = [real[p].get(w, np.inf) for p in pairs if p in real]
        lines.append(f"| {w:g} | " + " | ".join(str(sum(v <= b for v in vals)) for b in (3.6e12, 8.64e13, 3.16e16)) + " |")
    pulls = [max(max(r["max_pull"][s].values()) for s in ("G", "H")) for r in runs]
    seconds = [r["seconds"] for r in runs]
    lines.append(f"\nMonte Carlo check of the exact rates: largest pull over sets, median {median(pulls):.1f}, max {max(pulls):.1f} over files; "
                 f"GPU seconds per file: median {median(seconds):.0f}, max {max(seconds):.0f}, total hours {sum(seconds) / 3600:.1f}")
    open(f"{OUT}/tables_real.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"fetch": fetch, "tables": tables}[sys.argv[1]]()
