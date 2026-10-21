"""Aggregate the comparison with probabilistic graph networks.

    python report_gnn.py fetch
    python report_gnn.py tables
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
from gnn_app import BUDGETS  # noqa: E402
from launch import SEEDS, pairs_of  # noqa: E402
from report import CLASSES, SHORT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results"
MODELS = ("coherent-w100", "step-tanh", "step-lin", "init-tanh", "init-lin")
FILES = tuple(f"{m}-{kind}.json" for m in MODELS
              for kind in [f"T{T}" for T in BUDGETS] + [f"T{T}-retest" for T in BUDGETS]
              + ["exact", "hotelling", "fp", "ideal", "synthetic"])
LABEL = {"coherent-w100": "coherent light, w = 100",
         "step-tanh": "GNN, noise every step, tanh", "step-lin": "GNN, noise every step, linear",
         "init-tanh": "GNN, noise at init, tanh", "init-lin": "GNN, noise at init, linear"}
FLOOR = 1e-12
CELLS = 4
EXPERIMENT = os.environ.get("GNN_EXPERIMENT", "gnn-compare-v4")


def separated(r, T, studentised=False):
    """The protocol's decision at alpha / (J * cells): Bonferroni over the
    cells, so that 'any cell' keeps the nominal level.  With `studentised`
    the features include the studentised profile, from the retest of the
    checkpoint where there is one, else from the test itself when it has
    them; without, the original features only."""
    b = r.get("budgets", {}).get(str(T), {})
    t = b.get("retest") or b.get("test")
    if not t:
        return False
    p = t["pvalues"]
    if studentised:
        if not any(k.startswith("studentised") for k in p):
            return False
        return bool(min(p.values()) <= 0.05 / (t["J"] * CELLS))
    old = {k: v for k, v in p.items() if not k.startswith("studentised")}
    return bool(min(old.values()) <= 0.05 / (len(old) * CELLS))


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    pairs = list(pairs_of(None))
    results = {}
    chunks = [pairs[start:start + 5] for start in range(0, len(pairs), 5)]
    for blob in read_many.starmap([(EXPERIMENT, chunk, list(FILES)) for chunk in chunks]):
        results.update(json.loads(gzip.decompress(base64.b64decode(blob))))
        print(f"{len(results)} files", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/raw_gnn.json", "w"))
    print(len(results), "files")


def category_of(pair):
    return pair.rsplit("_", 1)[0]


def tables():
    raw = json.load(open(f"{OUT}/raw_gnn.json"))
    pairs = pairs_of(None)
    runs = {}   # (pair, seed, model) -> {"budgets": {T: {"test": ...}}, "exact": ..., "hotelling": ...}
    fps = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        _, pair, seed, name = key.split("/")
        model, _, kind = name[:-5].rpartition("-")
        if kind == "retest":
            model, _, tkind = model.rpartition("-")
            entry = runs.setdefault((pair, int(seed[1:]), model), {"budgets": {}, "n": value.get("n")})
            entry["budgets"].setdefault(tkind[1:], {})["retest"] = value["test"]
            continue
        if kind == "fp":
            fps[(pair, int(seed[1:]), model)] = value
            continue
        entry = runs.setdefault((pair, int(seed[1:]), model), {"budgets": {}, "n": value.get("n")})
        if kind.startswith("T"):
            entry["budgets"][kind[1:]] = value
            entry["n"] = value["n"]
            entry["seconds"] = entry.get("seconds", 0) + value["seconds"]
        else:
            entry[kind] = value
            entry["seconds"] = entry.get("seconds", 0) + value.get("seconds", 0)
    lines = []

    # 1. pairs separated against the budget, per model, any cell
    lines.append("## Pairs separated by the protocol against the budget (any cell, alpha / (J cells))\n")
    lines.append("| Model | " + " | ".join(f"T = {T:.0e}" for T in BUDGETS) + " | exact |")
    lines.append("|---|" + "---|" * (len(BUDGETS) + 1))
    per_model_budget = {}
    for model in MODELS:
        counts = []
        for T in BUDGETS:
            sep = {pair for (pair, seed, m), r in runs.items() if m == model and separated(r, T)}
            per_model_budget[(model, T)] = sep
            counts.append(len(sep))
        exact = "-"
        if model in ("step-lin", "init-lin"):
            exact = len({pair for (pair, seed, m), r in runs.items() if m == model
                         and any(v["max_rel_diff"] > FLOOR
                                 for v in (r.get("exact") or r.get("ideal", {}).get("exact", {})).values())})
        lines.append(f"| {LABEL[model]} | " + " | ".join(map(str, counts)) + f" | {exact} |")
    lines.append(f"| pairs run | " + " | ".join(
        str(len({p for (p, s, m) in runs if m == MODELS[0]})) for _ in BUDGETS) + " | |")

    # 1b. with the studentised profile, at the budgets where it was computed
    def has_studentised(r, T):
        b = r["budgets"].get(str(T), {})
        t = b.get("retest") or b.get("test") or {}
        return any(k.startswith("studentised") for k in t.get("pvalues", {}))

    lines.append("\n## Pairs separated with the studentised profile added (retest from checkpoints or native), any cell\n")
    lines.append("| Model | " + " | ".join(f"T = {T:.0e}" for T in BUDGETS) + " |")
    lines.append("|---|" + "---|" * len(BUDGETS))
    for model in MODELS:
        counts = []
        for T in BUDGETS:
            have = {pair for (pair, seed, m), r in runs.items() if m == model and has_studentised(r, T)}
            sep = {pair for (pair, seed, m), r in runs.items() if m == model and separated(r, T, True)}
            counts.append(f"{len(sep)} of {len(have)}")
        lines.append(f"| {LABEL[model]} | " + " | ".join(counts) + " |")

    # 2. per class at the largest budget
    T = BUDGETS[-1]
    lines.append(f"\n## Pairs separated per class at T = {T:.0e}, protocol, any cell\n")
    lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
    lines.append("|---|" + "---|" * (len(CLASSES) + 1))
    for model in MODELS:
        sep = per_model_budget[(model, T)]
        counts = [sum(category_of(p) == c for p in sep) for c in CLASSES]
        lines.append(f"| {LABEL[model]} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    lines.append("| pairs | " + " | ".join(str(sum(category_of(p) == c for p in pairs)) for c in CLASSES)
                 + f" | {len(pairs)} |")

    # 3. Hotelling at the Hotelling budget
    lines.append("\n## Pairs separated by BREC's test on sampled copies (1e5 samples per copy), any cell\n")
    lines.append("| Model | eps = 1e-7 | eps at the sampling noise | unreliable (cells) |")
    lines.append("|---|---|---|---|")
    for model in MODELS:
        v = {p for (p, s, m), r in runs.items() if m == model and r.get("hotelling", {}).get("verbatim", {}).get("separated")}
        sc = {p for (p, s, m), r in runs.items() if m == model and r.get("hotelling", {}).get("scaled", {}).get("separated")}
        un = sum(1 for (p, s, m), r in runs.items() if m == model and r.get("hotelling") and not r["hotelling"]["scaled"]["reliable"])
        lines.append(f"| {LABEL[model]} | {len(v)} | {len(sc)} | {un} |")

    # 4. smallest budget that separates, median per class, and ratios to coherent w = 100
    lines.append("\n## Smallest budget from which the pair stays separated at every larger budget (median over such pairs, best cell)\n")
    lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
    lines.append("|---|" + "---|" * len(CLASSES))
    first = {}
    for model in MODELS:
        for pair in pairs:
            best = None
            available = [T for T in BUDGETS if any(
                str(T) in runs.get((pair, seed, model), {}).get("budgets", {}) for seed in SEEDS)]
            for T in available:
                larger = [U for U in available if U >= T]
                if all(any(separated(runs.get((pair, seed, model), {}), U) for seed in SEEDS) for U in larger):
                    best = T
                    break
            first[(model, pair)] = best
        cells = []
        for c in CLASSES:
            vals = [first[(model, p)] for p in pairs if category_of(p) == c and first[(model, p)]]
            cells.append(f"{median(vals):.0e} ({len(vals)})" if vals else "-")
        lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")

    # 5. linear against tanh: pairs separated by one and not the other at the largest budget
    lines.append(f"\n## Linear against tanh at T = {T:.0e}\n")
    lines.append("| Noise | both | linear only | tanh only | neither |")
    lines.append("|---|---|---|---|---|")
    for lin, tanh, noise in (("step-lin", "step-tanh", "every step"), ("init-lin", "init-tanh", "at init")):
        a, b = per_model_budget[(lin, T)], per_model_budget[(tanh, T)]
        lines.append(f"| {noise} | {len(a & b)} | {len(a - b)} | {len(b - a)} | {len(set(pairs) - a - b)} |")
        only = sorted(b - a)
        if only:
            lines.append(f"\ntanh only: {', '.join(only)}\n")

    # 6. coherent light against the GNNs
    lines.append(f"\n## Coherent light at w = 100 against each network at T = {T:.0e}\n")
    lines.append("| Network | both | coherent only | network only |")
    lines.append("|---|---|---|---|")
    coh = per_model_budget[("coherent-w100", T)]
    for model in MODELS[1:]:
        other = per_model_budget[(model, T)]
        lines.append(f"| {LABEL[model]} | {len(coh & other)} | {len(coh - other)} | {len(other - coh)} |")

    # 6b. ideal counts: the optimal diagonal test, exact gap over sampled noise,
    # for the linear networks against coherent light at w = 100 from the full run
    full_path = f"{OUT}/raw_full.json"
    if os.path.exists(full_path):
        full = json.load(open(full_path))
        lines.append("\n## Ideal sample counts (exact gap, estimated noise, all orders), median per class, best cell\n")
        lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
        lines.append("|---|" + "---|" * len(CLASSES))
        ideal = {}
        for pair in pairs:
            vals = [full.get(f"brec-full-coherent/{pair}/s{seed}/g0.9.json") for seed in SEEDS]
            vals = [v for v in vals if isinstance(v, dict)]
            if vals:
                ideal[("coherent-w100", pair)] = min(
                    v["coherent_samples_by_brightness"]["100.0"][str(v["nmax"])] for v in vals)
            for model in ("step-lin", "init-lin"):
                vs = [runs.get((pair, seed, model), {}).get("ideal") for seed in SEEDS]
                vs = [v for v in vs if isinstance(v, dict)]
                if vs:
                    ideal[(model, pair)] = min(v["ideal_samples"][str(v["nmax"])] for v in vs)
        for model in ("coherent-w100", "step-lin", "init-lin"):
            cells = []
            for c in CLASSES:
                v = [ideal[(model, p)] for p in pairs if category_of(p) == c and (model, p) in ideal
                     and np.isfinite(ideal[(model, p)])]
                cells.append(f"{median(v):.2g} ({len(v)})" if v else "-")
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")
        lines.append("\n**Ratio of a network's ideal count to coherent light's, median over pairs both have finite**\n")
        lines.append("| Network | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
        lines.append("|---|" + "---|" * len(CLASSES))
        for model in ("step-lin", "init-lin"):
            cells = []
            for c in CLASSES:
                r = [ideal[(model, p)] / ideal[("coherent-w100", p)] for p in pairs
                     if category_of(p) == c and (model, p) in ideal and ("coherent-w100", p) in ideal
                     and np.isfinite(ideal[(model, p)]) and np.isfinite(ideal[("coherent-w100", p)])]
                cells.append(f"{median(r):.3g} ({len(r)})" if r else "-")
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")

    # 6c. synthetic batches: original against studentised statistic at every budget
    synth = {(p, s_, m): r["synthetic"] for (p, s_, m), r in runs.items() if "synthetic" in r}
    if synth:
        from gnn_app import SYNTHETIC_BUDGETS
        lines.append("\n## Synthetic batches from exact moments: pairs separated in at least half the draws, any cell, original statistic / with the studentised profile\n")
        lines.append("| Model | " + " | ".join(f"{T:.0e}" for T in SYNTHETIC_BUDGETS) + " |")
        lines.append("|---|" + "---|" * len(SYNTHETIC_BUDGETS))
        for model in ("coherent-w100", "step-lin", "init-lin"):
            cells = []
            for T in SYNTHETIC_BUDGETS:
                old = {p for (p, s_, m), r in synth.items() if m == model and r["budgets"].get(str(T), {}).get("old", 0) >= 0.5}
                new = {p for (p, s_, m), r in synth.items() if m == model and r["budgets"].get(str(T), {}).get("new", 0) >= 0.5}
                cells.append(f"{len(old)} / {len(new)}")
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")
        lines.append("\n**Median budget to separate (smallest T separated in half the draws and at every larger budget), per class, original / studentised**\n")
        lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
        lines.append("|---|" + "---|" * len(CLASSES))
        for model in ("coherent-w100", "step-lin", "init-lin"):
            cells = []
            for c in CLASSES:
                firsts = {"old": [], "new": []}
                for pair in pairs:
                    if category_of(pair) != c:
                        continue
                    for key in ("old", "new"):
                        best = None
                        for T in SYNTHETIC_BUDGETS:
                            ok = all(any(synth.get((pair, s_, model), {}).get("budgets", {}).get(str(U), {}).get(key, 0) >= 0.5
                                         for s_ in SEEDS) for U in SYNTHETIC_BUDGETS if U >= T)
                            if ok:
                                best = T
                                break
                        if best:
                            firsts[key].append(best)
                cells.append(f"{median(firsts['old']):.0e} ({len(firsts['old'])}) / {median(firsts['new']):.0e} ({len(firsts['new'])})"
                             if firsts["old"] or firsts["new"] else "-") if (firsts["old"] and firsts["new"]) else cells.append("-")
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")

    # 7. false positives
    if fps:
        lines.append("\n## False-positive rate of the protocol on G against relabelled G (10 runs per cell): at alpha / J, and at alpha / (J cells)\n")
        lines.append("| Model | " + " | ".join(f"T = {T:.0e}" for T in BUDGETS if T <= 1000000) + " |")
        lines.append("|---|" + "---|" * sum(T <= 1000000 for T in BUDGETS))
        for model in MODELS:
            cells = []
            for T in BUDGETS:
                if T > 1000000:
                    continue
                vals = [r["runs"].get(str(T)) for (p, s, m), r in fps.items() if m == model]
                vals = [v for v in vals if v is not None]
                strict = [p_ <= 0.05 / (J * CELLS) for (p, s, m), r in fps.items() if m == model
                          for p_, J in r.get("min_p", {}).get(str(T), [])]
                cells.append(f"{np.mean(vals):.3f} / {np.mean(strict):.3f} ({len(vals)})" if vals and strict
                             else (f"{np.mean(vals):.3f} ({len(vals)})" if vals else "-"))
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")

    # 8. timings
    secs = [(r.get("n"), m, r.get("seconds", 0)) for (p, s, m), r in runs.items()]
    lines.append("\n## GPU seconds per pair and cell, all budgets and the Hotelling copies\n")
    lines.append("| Model | median | max |\n|---|---|---|")
    for model in MODELS:
        v = [s for n, m, s in secs if m == model]
        if v:
            lines.append(f"| {LABEL[model]} | {median(v):.0f} | {max(v):.0f} |")
    text = "\n".join(lines)
    open(f"{OUT}/tables_gnn.md", "w").write(text)
    json.dump({"per_model_budget": {f"{m}|{T}": sorted(v) for (m, T), v in per_model_budget.items()},
               "first": {f"{m}|{p}": v for (m, p), v in first.items()}},
              open(f"{OUT}/results_gnn.json", "w"))
    print(text)


if __name__ == "__main__":
    {"fetch": fetch, "tables": tables}[sys.argv[1]]()
