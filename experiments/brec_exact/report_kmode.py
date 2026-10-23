"""Fetch and tabulate the K-source-mode run.

    python report_kmode.py fetch
    python report_kmode.py tables
"""
import base64
import gzip
import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_kmode import KMODE_PAIRS  # noqa: E402
from noise_app import KMODE_EXPERIMENT, KMODE_BRIGHTNESSES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODES = (1, 2, 4, 8)


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    names = [f"K{K}-N{N}-v2.json" for K in MODES for N in (3, 4)]
    blob = read_many.remote(KMODE_EXPERIMENT, list(KMODE_PAIRS), names)
    out = json.loads(gzip.decompress(base64.b64decode(blob)))
    json.dump(out, open(f"{OUT}/raw_kmode.json", "w"))
    print(len(out), "files of", len(KMODE_PAIRS) * len(MODES) * 4)


def tables():
    raw = json.load(open(f"{OUT}/raw_kmode.json"))
    runs = [r for r in raw.values() if isinstance(r, dict)]
    lines = [f"## K-mode chips: {len(runs)} pair-cell-K files\n"]

    def best(pair, K, placement, source, w):
        vals = [r["ideal"][placement][source][str(w)][str(r["nmax"])] for r in runs
                if r["pair"] == pair and r["local_modes"] == K and placement in r["ideal"]
                and source in r["ideal"][placement] and str(w) in r["ideal"][placement][source]]
        return min(vals) if vals else np.inf

    def med(values):
        v = [x for x in values if np.isfinite(x)]
        return f"{median(v):.2g} ({len(v)})" if v else "-"

    pairs = list(KMODE_PAIRS)
    # 1. the four inputs at flux K, per K: separate photons, separate lasers, Fock in the symmetric mode, laser in the symmetric mode
    for eta in KMODE_BRIGHTNESSES:
        lines.append(f"\n## Detected flux K x {eta} photons per vertex and tick: median ideal count over pairs, best cell\n")
        lines.append("| K | separate photons | separate lasers | Fock |K> symmetric | laser symmetric | Fock |K> single mode | laser single mode | thermal separate |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for K in MODES:
            w1, wK = eta, K * eta
            row = [med([best(p, K, "separate", "photons p=1", w1) for p in pairs]),
                   med([best(p, K, "separate", "coherent", w1) for p in pairs]),
                   med([best(p, K, "symmetric", f"fock {K}", wK) for p in pairs]) if K > 1 else med([best(p, 1, "separate", "photons p=1", w1) for p in pairs]),
                   med([best(p, K, "symmetric", "coherent", wK) for p in pairs]) if K > 1 else med([best(p, 1, "separate", "coherent", w1) for p in pairs]),
                   med([best(p, K, "single", f"fock {K}", wK) for p in pairs]) if K > 1 else "-",
                   med([best(p, K, "single", "coherent", wK) for p in pairs]) if K > 1 else "-",
                   med([best(p, K, "separate", "thermal", w1) for p in pairs])]
            lines.append(f"| {K} | " + " | ".join(row) + " |")
        lines.append("\n**Ratios within a placement, median over pairs: laser / antibunched at the same columns and flux**\n")
        lines.append("| K | separate: laser / photons (theory 4) | symmetric: laser / Fock (theory (1+1/K)^2) | single: laser / Fock | separate photons / symmetric Fock (same flux) |")
        lines.append("|---|---|---|---|---|")
        for K in MODES:
            w1, wK = eta, K * eta
            r1 = [best(p, K, "separate", "coherent", w1) / best(p, K, "separate", "photons p=1", w1) for p in pairs]
            if K > 1:
                r2 = [best(p, K, "symmetric", "coherent", wK) / best(p, K, "symmetric", f"fock {K}", wK) for p in pairs]
                r3 = [best(p, K, "single", "coherent", wK) / best(p, K, "single", f"fock {K}", wK) for p in pairs]
                r4 = [best(p, K, "separate", "photons p=1", w1) / best(p, K, "symmetric", f"fock {K}", wK) for p in pairs]
            else:
                r2 = r3 = r4 = []
            lines.append(f"| {K} | {med(r1)} | {med(r2)} ({(1 + 1 / K) ** 2:.2f}) | {med(r3)} | {med(r4)} |")
    # 2. per pair at eta = 0.05 (Poisson regime) and 1
    for eta in (0.05, 1.0):
        lines.append(f"\n## Per pair at detected flux K x {eta}: laser / photons (separate) and laser / Fock (symmetric), by K\n")
        lines.append("| pair | n | " + " | ".join(f"K={K} sep. | K={K} sym." for K in MODES) + " |")
        lines.append("|---|---|" + "---|---|" * len(MODES))
        for p in pairs:
            n = next((r["n"] for r in runs if r["pair"] == p), "?")
            cells = []
            for K in MODES:
                a = best(p, K, "separate", "coherent", eta) / best(p, K, "separate", "photons p=1", eta)
                b = best(p, K, "symmetric", "coherent", K * eta) / best(p, K, "symmetric", f"fock {K}", K * eta) if K > 1 else np.nan
                cells.append(f"{a:.2f}" if np.isfinite(a) else "-")
                cells.append(f"{b:.2f}" if np.isfinite(b) else "-")
            lines.append(f"| {p} | {n} | " + " | ".join(cells) + " |")
    seconds = [r["seconds"] for r in runs]
    if seconds:
        lines.append(f"\nGPU seconds per job: median {median(seconds):.0f}, max {max(seconds):.0f}, total hours {sum(seconds) / 3600:.1f}")
    open(f"{OUT}/tables_kmode.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"fetch": fetch, "tables": tables}[sys.argv[1]]()
