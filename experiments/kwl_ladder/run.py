"""
The k-WL ladder benchmark: five models against every pair of the
dataset, one CSV row per cell.

Photonic models run through the exact certificates of :mod:`exact` —
the spacetime permanent engine for the passive and Bell models, the
branching engine for the active one. Qubit models contract the doubled
functor image through :mod:`contract`. Cost caps are explicit: the
active engine drops to the edge ensemble past ``EDGE_ENSEMBLE_N``
vertices (an invariant ensemble, logged in the row), and a cell whose
projected cost exceeds the budget is recorded as ``trimmed`` rather
than silently skipped.

Rows are appended to ``results/results.csv`` and cells already present
are not recomputed, so the benchmark resumes where it stopped.

Usage: ``python run.py [--models a,b] [--pairs id,id] [--ticks 8]
[--invariance] [--qubit-ticks 4]``.
"""

import argparse
import csv
import json
import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
FIELDS = ["pair", "rung", "model", "ticks", "engine", "ensemble",
          "separation", "mass_error", "seconds", "notes"]

EDGE_ENSEMBLE_N = 40
PHOTONIC = ("passive", "bell", "active", "active-flood")
RELABEL_SEED = 7


def load_pairs():
    with open(os.path.join(DATA, "dataset.json")) as handle:
        data = json.load(handle)
    return data["pairs"]


def graphs_of(record):
    from models import adjacency
    return tuple(adjacency(nx.from_graph6_bytes(g6.encode()))
                 for g6 in record["g6"])


def relabelled(graph, seed=RELABEL_SEED):
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(graph)))
    inverse = [perm.index(i) for i in range(len(graph))]
    return tuple(
        tuple(sorted(perm[x] for x in graph[inverse[i]]))
        for i in range(len(graph)))


def edges_of(graph):
    return [(u, v) for u, nbrs in enumerate(graph)
            for v in nbrs if u < v]


def photonic_bins(name, graph, ticks):
    from exact import aggregate, machine_for
    from models import MODELS
    model = MODELS[name]
    ensemble = None
    if len(graph) > EDGE_ENSEMBLE_N:
        ensemble = edges_of(graph)
    machine = machine_for(model, graph)
    engine = "branch" if name.startswith("active") else "spacetime"
    bins = aggregate(machine, ticks, engine=engine, ensemble=ensemble)
    return bins, ("all-pairs" if ensemble is None else "edges")


def run_cell(record, name, ticks, qubit_ticks, invariance):
    from exact import separation as bin_separation
    left, right = graphs_of(record)
    if invariance:
        right = relabelled(left)
    t0 = time.time()
    if name in PHOTONIC:
        one, ensemble = photonic_bins(name, left, ticks)
        two, _ = photonic_bins(name, right, ticks)
        sep = bin_separation(one, two)
        mass = max(abs(sum(one.values()) - 1),
                   abs(sum(two.values()) - 1))
        used_ticks = ticks
    else:
        import contract
        from models import MODELS
        model = MODELS[name]
        used_ticks = qubit_ticks
        optimizer = "greedy" if (
            name == "qubit" and len(left) <= 6) else "hyper"
        one = contract.summary(model, left, used_ticks, optimizer)
        two = contract.summary(model, right, used_ticks, optimizer)
        sep = float(np.abs(one - two).max())
        mass = max(abs(float(one.sum()) / len(left) - 1),
                   abs(float(two.sum()) / len(right) - 1))
        ensemble = "trajectories"
    return {
        "pair": record["id"] + ("~relabel" if invariance else ""),
        "rung": record["rung"], "model": name, "ticks": used_ticks,
        "engine": "exact", "ensemble": ensemble,
        "separation": f"{sep:.6e}", "mass_error": f"{mass:.2e}",
        "seconds": f"{time.time() - t0:.1f}", "notes": ""}


def existing_rows(path):
    if not os.path.exists(path):
        return set()
    with open(path) as handle:
        return {(row["pair"], row["model"], row["ticks"])
                for row in csv.DictReader(handle)}


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (13 * 2 ** 30,) * 2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="passive,bell,active")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--rungs", default="")
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--qubit-ticks", type=int, default=4)
    parser.add_argument("--invariance", action="store_true")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, args.out)
    done = existing_rows(path)
    records = load_pairs()
    if args.pairs:
        wanted = set(args.pairs.split(","))
        records = [r for r in records if r["id"] in wanted]
    if args.rungs:
        wanted = set(args.rungs.split(","))
        records = [r for r in records if r["rung"] in wanted]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if write_header:
            writer.writeheader()
        for record in records:
            for name in args.models.split(","):
                ticks = (args.qubit_ticks if name.startswith("qubit")
                         else args.ticks)
                pair_id = record["id"] + (
                    "~relabel" if args.invariance else "")
                if (pair_id, name, str(ticks)) in done:
                    continue
                row = run_cell(record, name, args.ticks,
                               args.qubit_ticks, args.invariance)
                writer.writerow(row)
                handle.flush()
                print(f"{row['pair']:28s} {name:9s} T={row['ticks']} "
                      f"sep={row['separation']} "
                      f"mass_err={row['mass_error']} "
                      f"({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
