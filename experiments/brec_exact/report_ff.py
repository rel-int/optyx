"""Fetch and tabulate the measurement-feedforward run.

    python report_ff.py fetch                  # results/raw_ff.json from the volume
    python report_ff.py tables [SEEDS]         # results/tables_ff.md, e.g. SEEDS=100

A pair is separated at budget T when the protocol separates it in at least
one of the cells at alpha / (J cells), J the number of features and cells
the number of seeds kept, and its order is the smallest order of a feature
below that threshold.
"""
import base64
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_ff import EXPERIMENT, configs, suffix  # noqa: E402
from launch_kmode import KMODE_PAIRS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEEDS = (100, 101, 102, 103)
ALPHA = 0.05
BUDGETS = [10 ** k for k in range(3, 13)]
LABEL = {"loop-coherent": "laser", "loop-thermal": "thermal"}


def fetch():
    import modal
    read_many = modal.Function.from_name("brec-photonic", "read_many")
    names = [f"{m}{suffix(k, d)}-T{T}.json" for m, k, d in configs() for T in BUDGETS]
    pairs = list(KMODE_PAIRS)
    out = {}
    for blob in read_many.starmap([(EXPERIMENT, pairs[i:i + 4], names) for i in range(0, len(pairs), 4)]):
        out.update(json.loads(gzip.decompress(base64.b64decode(blob))))
    json.dump(out, open(f"{OUT}/raw_ff.json", "w"))
    print(len(out), "files")


def decision(test, cells):
    """(separated, smallest separating order) at alpha / (J cells)."""
    threshold = ALPHA / (test["J"] * cells)
    orders = [int(name.rsplit("-", 1)[1]) for name, p in test["pvalues"].items() if p <= threshold]
    return (bool(orders), min(orders) if orders else None)


def collect(raw, seeds=SEEDS):
    """(pair, model, kappa, delay) -> T -> (separated in any cell, smallest order over cells)."""
    out = {}
    for r in raw.values():
        if not isinstance(r, dict) or r["seed"] not in seeds:
            continue
        key = (r["pair"], r["model"], float(r["kappa"]), int(r["delay"]))
        sep, order = decision(r["test"], len(seeds))
        cur = out.setdefault(key, {}).setdefault(r["T"], [False, None])
        cur[0] = cur[0] or sep
        if order is not None:
            cur[1] = order if cur[1] is None else min(cur[1], order)
    return out


def first_budget(by_T):
    hits = sorted(T for T, (sep, _) in by_T.items() if sep)
    return hits[0] if hits else None


def tables(seeds=SEEDS):
    raw = json.load(open(f"{OUT}/raw_ff.json"))
    res = collect(raw, seeds)
    cols = [c for c in configs() if any(key[1:] == c for key in res)]
    head = [f"{LABEL[m]} k={k:g}" + (f" d={d}" if k else "") for m, k, d in cols]
    top = max((T for v in res.values() for T in v), default=0)
    lines = [f"## Smallest budget (samples per graph) at which the protocol separates the pair, and the smallest separating order there\n",
             f"Brightness 100, gamma 0.9, cells {list(seeds)}, alpha / (J cells); '-' = not separated up to {top:.0e}.\n",
             "| pair | n | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(cols)]
    import brecphot as bp
    from gpu_app import BREC_RAW
    data = bp.load_brec(BREC_RAW)
    for pair in [p for p in KMODE_PAIRS if any(key[0] == p for key in res)]:
        c, i = pair.rsplit("_", 1)
        n = bp.adjacency(data[c][int(i)][0]).shape[0]
        cells = []
        for m, k, d in cols:
            by_T = res.get((pair, m, k, d), {})
            T = first_budget(by_T)
            cells.append(f"{T:.0e} (N={by_T[T][1]})" if T else ("-" if by_T else ""))
        lines.append(f"| {pair} | {n} | " + " | ".join(cells) + " |")
    lines.append("\n## Pairs separated at each budget\n")
    lines.append("| config | " + " | ".join(f"{T:.0e}" for T in BUDGETS if T <= top) + " |")
    lines.append("|---|" + "---|" * len([T for T in BUDGETS if T <= top]))
    for (m, k, d), h in zip(cols, head):
        counts = [sum(res.get((p, m, k, d), {}).get(T, [False])[0] for p in KMODE_PAIRS) for T in BUDGETS if T <= top]
        counts = [f"{c}/{sum((p, m, k, d) in res for p in KMODE_PAIRS)}" for c in counts]
        lines.append(f"| {h} | " + " | ".join(map(str, counts)) + " |")
    open(f"{OUT}/tables_ff.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch()
    else:
        tables(tuple(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else SEEDS)
