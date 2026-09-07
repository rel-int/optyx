"""
The invariant QMapNN benchmark: every cell against the controls and the
1-FWL-blind rung of the ladder dataset at ``T = 2, 3, 4``, one CSV row
per (pair, cell, ticks, certificate), the rook/Shrikhande sanity row at
``T = 2`` for the canonical cell, the rotation-system table and the
relabelling rows.

Pruning follows PR #69: a cell exactly zero on every pair of a rung does
not climb to the next. Rows are appended to ``results/results.csv`` and
``results/rotations.csv`` and existing rows are not recomputed, so the
benchmark resumes where it stopped.

Usage: ``python run.py [--cells a,b] [--pairs id,id] [--ticks 2,3,4]
[--certificates two-photon,one-photon,coherent] [--rotations]
[--invariance]``.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cells import CELLS, SEEDS, rotate  # noqa: E402
from engine import bins, read, slots_of  # noqa: E402
from ladder import graphs_of, load_pairs, relabelled, separation  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
FIELDS = ["pair", "rung", "cell", "ticks", "certificate", "separation",
          "l1", "mass_error", "seconds", "notes"]
ROTATION_FIELDS = ["pair", "rung", "side", "cell", "ticks", "spread",
                   "mass_error", "seconds"]
EXACT_ZERO = 1e-15
RUNGS = ("control", "1fwl-blind", "2fwl-blind")
SANITY, SANITY_CELL = "str-00", "canonical"
ROTATIONS = (1, 2, 3)
NAMED = ("canonical", "grover", "ladder", "generic")


def scored(cell, left, right, ticks, certificates):
    """For every certificate, the separation, its L1 counterpart and
    the mass error of two graphs under one cell and one number of
    ticks; the slot amplitudes are shared by the certificates."""
    slots = [slots_of(cell(graph), ticks) for graph in (left, right)]
    scores = {}
    for certificate in certificates:
        results = [read(*slot, certificate) for slot in slots]
        one, two = (bins(nodes, total) for nodes, total, _ in results)
        keys = set(one) | set(two)
        l1 = sum(abs(one.get(key, 0.) - two.get(key, 0.)) for key in keys)
        scores[certificate] = (separation(one, two), l1, max(
            abs(mass - 1) for _, _, mass in results))
    return scores


def existing(path, fields, value="separation"):
    """The rows already in a CSV, keyed by ``fields``, so that the
    benchmark resumes where it stopped and pruning reads the scores
    it skips."""
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return {tuple(row[field] for field in fields): float(row[value])
                for row in csv.DictReader(handle)}


def writer_for(path, fields):
    write_header = not os.path.exists(path)
    handle = open(path, "a", newline="")
    writer = csv.DictWriter(handle, fields)
    if write_header:
        writer.writeheader()
    return handle, writer


def run_rows(records, names, ticks, certificates, invariance):
    """Separation rows, climbing the rungs and pruning per cell."""
    path = os.path.join(RESULTS, "results.csv")
    done = existing(path, ("pair", "cell", "ticks", "certificate"))
    handle, writer = writer_for(path, FIELDS)
    for name in names:
        cell = CELLS[name]
        for rung in RUNGS:
            climbed = False
            for record in records:
                if record["rung"] != rung:
                    continue
                if rung == "2fwl-blind" and (
                        record["id"] != SANITY or invariance
                        or name != SANITY_CELL):
                    continue
                left, right = graphs_of(record)
                if invariance:
                    right = relabelled(left)
                pair = record["id"] + ("~relabel" if invariance else "")
                for n_ticks in ([2] if rung == "2fwl-blind" else ticks):
                    keys = {c: (pair, name, str(n_ticks), c)
                            for c in certificates}
                    climbed |= any(done[key] > EXACT_ZERO
                                   for key in keys.values() if key in done)
                    wanted = [c for c, key in keys.items() if key not in done]
                    if not wanted:
                        continue
                    t0 = time.time()
                    scores = scored(cell, left, right, n_ticks, wanted)
                    for certificate, (sep, l1, mass) in scores.items():
                        row = {
                            "pair": pair, "rung": rung, "cell": name,
                            "ticks": n_ticks, "certificate": certificate,
                            "separation": f"{sep:.6e}", "l1": f"{l1:.6e}",
                            "mass_error": f"{mass:.2e}",
                            "seconds": f"{time.time() - t0:.1f}",
                            "notes": (
                                "1 - mass is the Poisson weight beyond "
                                "two photons" if certificate == "coherent"
                                else "")}
                        writer.writerow(row)
                        handle.flush()
                        climbed |= sep > EXACT_ZERO
                        print(f"{pair:22s} {name:10s} T={n_ticks} "
                              f"{certificate:10s} sep={row['separation']} "
                              f"l1={row['l1']} mass_err={row['mass_error']}",
                              flush=True)
            if not climbed and not invariance and rung != "2fwl-blind":
                print(f"{name}: exactly zero on every {rung} pair, "
                      "not climbing", flush=True)
                break
    handle.close()


def run_rotations(records, names, ticks, certificate):
    """The rotation-system table: each graph of a pair under three
    shuffled rotation systems against its sorted one."""
    path = os.path.join(RESULTS, "rotations.csv")
    done = existing(path, ("pair", "side", "cell", "ticks"), "spread")
    handle, writer = writer_for(path, ROTATION_FIELDS)
    for record in records:
        if record["rung"] == "2fwl-blind":
            continue
        for side, graph in enumerate(graphs_of(record)):
            for name in names:
                for n_ticks in ticks:
                    key = (record["id"], str(side), name, str(n_ticks))
                    if key in done:
                        continue
                    t0 = time.time()
                    scores = [scored(CELLS[name], graph, rotate(graph, seed),
                                     n_ticks, [certificate])[certificate]
                              for seed in ROTATIONS]
                    row = {
                        "pair": record["id"], "rung": record["rung"],
                        "side": side, "cell": name, "ticks": n_ticks,
                        "spread": f"{max(s[0] for s in scores):.6e}",
                        "mass_error": f"{max(s[2] for s in scores):.2e}",
                        "seconds": f"{time.time() - t0:.1f}"}
                    writer.writerow(row)
                    handle.flush()
                    print(f"{record['id']:22s} side {side} {name:10s} "
                          f"T={n_ticks} spread={row['spread']}", flush=True)
    handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default=",".join(
        NAMED + tuple(f"seed-{seed:02d}" for seed in SEEDS)))
    parser.add_argument("--pairs", default="")
    parser.add_argument("--ticks", default="2,3,4")
    parser.add_argument("--certificates", default="two-photon")
    parser.add_argument("--rotations", action="store_true")
    parser.add_argument("--invariance", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    records = load_pairs()
    if args.pairs:
        wanted = set(args.pairs.split(","))
        records = [r for r in records if r["id"] in wanted]
    names = args.cells.split(",")
    ticks = [int(t) for t in args.ticks.split(",")]
    certificates = args.certificates.split(",")
    if args.rotations:
        run_rotations(records, names, ticks, certificates[0])
    else:
        run_rows(records, names, ticks, certificates, args.invariance)


if __name__ == "__main__":
    main()
