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
FILES = ("step-lin.json", "step-tanh.json", "init-lin.json", "init-tanh.json",
         "coherent-w100.0.json", "coherent-w1.0.json",
         "step-lin-fp.json", "step-tanh-fp.json", "init-lin-fp.json", "init-tanh-fp.json",
         "coherent-w100.0-fp.json", "coherent-w1.0-fp.json")
MODELS = ("coherent-w100.0", "coherent-w1.0", "step-tanh", "step-lin", "init-tanh", "init-lin")
LABEL = {"coherent-w100.0": "coherent light, w = 100", "coherent-w1.0": "coherent light, w = 1",
         "step-tanh": "GNN, noise every step, tanh", "step-lin": "GNN, noise every step, linear",
         "init-tanh": "GNN, noise at init, tanh", "init-lin": "GNN, noise at init, linear"}
FLOOR = 1e-12


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    pairs = list(pairs_of(None))
    results = {}
    for start in range(0, len(pairs), 25):
        blob = read_many.remote("gnn-compare", pairs[start:start + 25], list(FILES))
        results.update(json.loads(gzip.decompress(base64.b64decode(blob))))
        print(f"{len(results)} files after {start + 25} pairs", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/raw_gnn.json", "w"))
    print(len(results), "files")


def category_of(pair):
    return pair.rsplit("_", 1)[0]


def tables():
    raw = json.load(open(f"{OUT}/raw_gnn.json"))
    pairs = pairs_of(None)
    runs = {}   # (pair, seed, model) -> result
    fps = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        _, pair, seed, name = key.split("/")
        model = name[:-5]
        if model.endswith("-fp"):
            fps[(pair, int(seed[1:]), model[:-3])] = value
        else:
            runs[(pair, int(seed[1:]), model)] = value
    lines = []

    # 1. pairs separated against the budget, per model, any cell
    lines.append("## Pairs separated by the protocol against the budget (any of four cells)\n")
    lines.append("| Model | " + " | ".join(f"T = {T:.0e}" for T in BUDGETS) + " | exact |")
    lines.append("|---|" + "---|" * (len(BUDGETS) + 1))
    per_model_budget = {}
    for model in MODELS:
        counts = []
        for T in BUDGETS:
            sep = {pair for (pair, seed, m), r in runs.items()
                   if m == model and r["budgets"].get(str(T), {}).get("test", {}).get("separated")}
            per_model_budget[(model, T)] = sep
            counts.append(len(sep))
        exact = "-"
        if model in ("step-lin", "init-lin"):
            exact = len({pair for (pair, seed, m), r in runs.items() if m == model and "exact" in r
                         and any(v["max_rel_diff"] > FLOOR for v in r["exact"].values())})
        lines.append(f"| {LABEL[model]} | " + " | ".join(map(str, counts)) + f" | {exact} |")
    lines.append(f"| pairs run | " + " | ".join(
        str(len({p for (p, s, m) in runs if m == MODELS[0]})) for _ in BUDGETS) + " | |")

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
    lines.append("\n## Smallest budget that separates (median over pairs the model separates at some budget, best cell)\n")
    lines.append("| Model | " + " | ".join(SHORT[c] for c in CLASSES) + " |")
    lines.append("|---|" + "---|" * len(CLASSES))
    first = {}
    for model in MODELS:
        for pair in pairs:
            best = None
            for T in BUDGETS:
                if any(runs.get((pair, seed, model), {}).get("budgets", {}).get(str(T), {}).get("test", {}).get("separated")
                       for seed in SEEDS):
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
    coh = per_model_budget[("coherent-w100.0", T)]
    for model in MODELS[1:]:
        other = per_model_budget[(model, T)]
        lines.append(f"| {LABEL[model]} | {len(coh & other)} | {len(coh - other)} | {len(other - coh)} |")

    # 7. false positives
    if fps:
        lines.append("\n## False-positive rate of the protocol on G against relabelled G (10 runs per cell)\n")
        lines.append("| Model | " + " | ".join(f"T = {T:.0e}" for T in BUDGETS if T <= 1000000) + " |")
        lines.append("|---|" + "---|" * sum(T <= 1000000 for T in BUDGETS))
        for model in MODELS:
            cells = []
            for T in BUDGETS:
                if T > 1000000:
                    continue
                vals = [r["runs"].get(str(T)) for (p, s, m), r in fps.items() if m == model]
                vals = [v for v in vals if v is not None]
                cells.append(f"{np.mean(vals):.3f} ({len(vals)})" if vals else "-")
            lines.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")

    # 8. timings
    secs = [(r["n"], m, r["seconds"]) for (p, s, m), r in runs.items()]
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
