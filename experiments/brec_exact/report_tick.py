"""Aggregate the forward-pass test on the pilot subset.

    python report_tick.py fetch
    python report_tick.py tables
"""

import base64
import gzip
import json
import os
import re
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
import single_tick as st  # noqa: E402
from launch import SEEDS, pairs_of  # noqa: E402
from report import CLASSES, SHORT  # noqa: E402
from tick_app import EXPERIMENT as DEFAULT_EXPERIMENT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results"
EXPERIMENT = os.environ.get("TICK_EXPERIMENT", DEFAULT_EXPERIMENT)
SUFFIX = "" if EXPERIMENT == DEFAULT_EXPERIMENT else "_" + EXPERIMENT.replace("tick-pilot-", "")
LABEL = {"coherent-w1": "coherent light, w = 1", "coherent-w100": "coherent light, w = 100",
         "coherent-inf": "coherent light, intensities", "thermal-inf": "thermal light, intensities"}
PROFILE_COLUMN = {"coherent-w1": "coherent", "coherent-w100": "coherent",
                  "coherent-inf": "coherent", "thermal-inf": "thermal"}
FLOOR = 1e-12


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    pairs = list(pairs_of(None))
    names = [f"{s}.json" for s in st.SOURCES] + ["anchors.json"]
    results = {}
    # one pair per call keeps the result inline: a larger one goes through
    # Modal's blob store, which the session's proxy does not reach
    for i, pair in enumerate(pairs):
        blob = read_many.remote(EXPERIMENT, [pair], names)
        results.update(json.loads(gzip.decompress(base64.b64decode(blob))))
        print(f"{len(results)} files after {i + 1} pairs", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(f"{OUT}/raw_tick{SUFFIX}.json.gz", "wt") as f:
        json.dump(results, f)
    print(len(results), "files")


def category_of(pair):
    return pair.rsplit("_", 1)[0]


CELLS = 4


def permutation_pvalue(X, Y, splits=999, seed=0):
    """The studentised distance sum_j (mean X_j - mean Y_j)^2 / s_j^2 between
    the two groups of copies, s_j^2 the pooled within-group variance, against
    its values over random reassignments of the 2q copies."""
    rng = np.random.default_rng(seed)
    allc = np.concatenate([X, Y])
    q = X.shape[0]
    masks = np.zeros((splits + 1, 2 * q))
    masks[0, :q] = 1
    for r in range(1, splits + 1):
        masks[r, rng.choice(2 * q, q, replace=False)] = 1
    a, b = masks / q, (1 - masks) / q
    mean_a, mean_b = a @ allc, b @ allc
    sq = allc * allc
    within = (np.maximum(a @ sq - mean_a ** 2, 0) + np.maximum(b @ sq - mean_b ** 2, 0)) / 2
    scale = np.mean(allc ** 2, axis=0, keepdims=True)
    d = np.sum((mean_a - mean_b) ** 2 / (within + 1e-12 * scale + 1e-300), axis=1)
    return float((1 + np.sum(d[1:] >= d[0])) / (splits + 1))


def exact_orders():
    """The exact separation order of every pair and source of the pilot's
    sorted profiles, read off results/tables_pilot.md."""
    orders = {}
    columns = ("photons", "distinguishable", "coherent", "thermal")
    for line in open(f"{HERE}/results/tables_pilot.md"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] in pairs_of(None):
            for column, cell in zip(columns, cells[3:7]):
                m = re.match(r"(\S+) / (\S+), (\S+)", cell)
                orders[(cells[0], column)] = (None if m.group(2) == "None" else int(m.group(2)),
                                              float(m.group(3)))
    return orders


def tables():
    with gzip.open(f"{OUT}/raw_tick{SUFFIX}.json.gz", "rt") as f:
        raw = json.load(f)
    pairs = pairs_of(None)
    runs, anchors = {}, {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        _, pair, seed, name = key.split("/")
        if name == "anchors.json":
            anchors[(pair, int(seed[1:]))] = value
        else:
            runs[(pair, int(seed[1:]), name[:-5])] = value
    orders = exact_orders()
    budgets = sorted({int(R) for r in runs.values() for R in r["decisions"]})
    lines = []

    def decision(d, rule, key=None):
        """BREC's rules as stored; 'hotelling': the scaled statistic with the
        factor q = 32 that BREC's code leaves out, so that 72.34 is the 5%
        point of Hotelling's T^2 it is named after; 'permutation': the
        studentised distance between the copy means of G and H against its
        distribution over 999 reassignments of the 64 copies, at level
        0.05 / 4 over the four cells, from the stored embeddings."""
        if rule == "permutation":
            return permutation_rules.setdefault(key, permutation_decision(*key))
        if rule != "hotelling":
            return d[rule]
        q = bp.NUM_RELABEL
        t2, t2r = q * d["scaled"]["T2_pair"], q * d["scaled"]["T2_reliability"]
        separated, reliable = bp.brec_decision(t2, t2r)
        return {"separated": separated, "reliable": reliable}

    permutation_rules = {}

    def permutation_decision(pair, seed, source, R):
        r = runs[(pair, seed, source)]
        q = bp.NUM_RELABEL
        E = lambda tags: np.array([r["embeddings"][t][str(R)] for t in tags])  # noqa: E731
        X, Y, Z = E([f"G{k}" for k in range(q)]), E([f"H{k}" for k in range(q)]), E([f"G{k}" for k in range(q, 2 * q)])
        return {"separated": permutation_pvalue(X, Y) <= 0.05 / CELLS,
                "reliable": permutation_pvalue(X, Z) > 0.05 / CELLS}

    def separated(pair, source, R, rule):
        return any(decision(runs[(pair, s, source)]["decisions"][str(R)], rule, (pair, s, source, R))["separated"]
                   for s in SEEDS if (pair, s, source) in runs)

    for rule in ("permutation", "hotelling", "scaled", "verbatim"):
        lines.append(f"## Pairs separated by BREC's test, rule {rule}, any of four cells\n")
        lines.append("| Source | " + " | ".join(f"R = {R:.0e}" for R in budgets) + " | profiles, exact |")
        lines.append("|---|" + "---|" * (len(budgets) + 1))
        for source in st.SOURCES:
            counts = [sum(separated(p, source, R, rule) for p in pairs) for R in budgets]
            exact = sum(orders.get((p, PROFILE_COLUMN[source]), (None,))[0] is not None for p in pairs)
            lines.append(f"| {LABEL[source]} | " + " | ".join(map(str, counts)) + f" | {exact} |")
        lines.append(f"| pairs run | " + " | ".join(
            str(sum(any((p, s, src) in runs for s in SEEDS for src in st.SOURCES) for p in pairs))
            for _ in budgets) + f" | {len(pairs)} |")
        lines.append("")
    R = budgets[-1]
    lines.append(f"## Pairs separated per class at R = {R:.0e}, permutation rule, any cell\n")
    lines.append("| Source | " + " | ".join(SHORT[c] for c in CLASSES) + " | Total |")
    lines.append("|---|" + "---|" * (len(CLASSES) + 1))
    for source in st.SOURCES:
        row = [sum(separated(p, source, R, "permutation") for p in pairs if category_of(p) == c) for c in CLASSES]
        lines.append(f"| {LABEL[source]} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    for column, name in (("coherent", "profiles, coherent, exact"), ("thermal", "profiles, thermal, exact")):
        row = [sum(orders.get((p, column), (None,))[0] is not None for p in pairs if category_of(p) == c)
               for c in CLASSES]
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    lines.append("| pairs | " + " | ".join(str(sum(category_of(p) == c for p in pairs)) for c in CLASSES)
                 + f" | {len(pairs)} |")
    lines.append("")
    lines.append(f"## Agreement with the exact profiles at R = {R:.0e}, permutation rule\n")
    lines.append("| Source | both | test only | profiles only | neither |")
    lines.append("|---|---|---|---|---|")
    for source in st.SOURCES:
        both = test_only = prof_only = neither = 0
        for p in pairs:
            t = separated(p, source, R, "permutation")
            e = orders.get((p, PROFILE_COLUMN[source]), (None,))[0] is not None
            both += t and e
            test_only += t and not e
            prof_only += e and not t
            neither += not t and not e
        lines.append(f"| {LABEL[source]} | {both} | {test_only} | {prof_only} | {neither} |")
    lines.append("")
    lines.append("## Reliability failures (pair-cell cases of 196) and median T² at the largest budget\n")
    lines.append("| Source | unreliable, hotelling | unreliable, permutation | unreliable, verbatim | median T² pair (BREC's, no factor q) | median T² reliability |")
    lines.append("|---|---|---|---|---|---|")
    for source in st.SOURCES:
        cases = [r["decisions"][str(R)] for (p, s, src), r in runs.items() if src == source]
        if not cases:
            continue
        lines.append(f"| {LABEL[source]} | {sum(not decision(c, 'hotelling')['reliable'] for c in cases)} | "
                     f"{sum(not permutation_decision(p, s, src, R)['reliable'] for (p, s, src) in runs if src == source)} | "
                     f"{sum(not c['verbatim']['reliable'] for c in cases)} | "
                     f"{median(c['scaled']['T2_pair'] for c in cases):.3g} | "
                     f"{median(c['scaled']['T2_reliability'] for c in cases):.3g} |")
    lines.append("")
    lines.append("## Which feature carries the gap: the feature with the largest gap over noise, counted over the pairs separated at the largest budget\n")
    lines.append("| Source | " + " | ".join(st.feature_names()) + " |")
    lines.append("|---|" + "---|" * len(st.feature_names()))
    for source in st.SOURCES:
        counts = np.zeros(len(st.feature_names()), dtype=int)
        for p in pairs:
            if not separated(p, source, R, "permutation"):
                continue
            best = max((np.array(runs[(p, s, source)]["decisions"][str(R)]["gap_over_noise"])
                        for s in SEEDS if (p, s, source) in runs), key=np.max)
            counts[int(np.argmax(best))] += 1
        lines.append(f"| {LABEL[source]} | " + " | ".join(map(str, counts)) + " |")
    lines.append("")
    lines.append("## Exact anchors: relative gap of the exact readout between G and H (cell 100), against the profile's exact order\n")
    lines.append("| Pair | n | " + " | ".join(k for k in sorted(next(iter(anchors.values())).keys()) if k != "seconds")
                 + " | profile order coherent / thermal |" if anchors else "")
    if anchors:
        keys = [k for k in sorted(next(iter(anchors.values())).keys()) if k != "seconds"]
        lines.append("|---|---|" + "---|" * (len(keys) + 1))
        for p in pairs:
            a = anchors.get((p, SEEDS[0]))
            if a is None:
                continue
            n = next((r["n"] for (pp, s, src), r in runs.items() if pp == p), "")
            cells = [f"{a[k]['relative_gap']:.1e}" if k in a else "-" for k in keys]
            oc, ot = orders.get((p, "coherent"), (None,))[0], orders.get((p, "thermal"), (None,))[0]
            lines.append(f"| {p} | {n} | " + " | ".join(cells) + f" | {oc} / {ot} |")
        lines.append("")
    lines.append("## Per pair: smallest budget that separates, permutation rule, any cell\n")
    lines.append("| Pair | n | slots | " + " | ".join(LABEL[s] for s in st.SOURCES) + " | profile order coherent / thermal |")
    lines.append("|---|---|---|" + "---|" * (len(st.SOURCES) + 1))
    for p in pairs:
        n = slots = ""
        cells = []
        for source in st.SOURCES:
            first = next((R for R in budgets if separated(p, source, R, "permutation")), None)
            cells.append("-" if first is None else f"{first:.0e}")
            r = next((runs[(p, s, source)] for s in SEEDS if (p, s, source) in runs), None)
            if r:
                n, slots = r["n"], r["slots"]
        oc, ot = orders.get((p, "coherent"), (None,))[0], orders.get((p, "thermal"), (None,))[0]
        lines.append(f"| {p} | {n} | {slots} | " + " | ".join(cells) + f" | {oc} / {ot} |")
    lines.append("")
    lines.append("## GPU seconds per pair and cell, all four sources at every budget\n")
    secs = {}
    for (p, s, src), r in runs.items():
        secs[(p, s)] = secs.get((p, s), 0) + r["seconds"]
    if secs:
        lines.append(f"median {median(secs.values()):.0f}, max {max(secs.values()):.0f}, "
                     f"total {sum(secs.values()) / 3600:.2f} GPU-hours over {len(secs)} pair-cell jobs")
    open(f"{OUT}/tables_tick{SUFFIX}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"fetch": fetch, "tables": tables}[sys.argv[1]]()
