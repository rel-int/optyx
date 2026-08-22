"""Render ``results/results.csv`` as the markdown tables of the
report: one row per pair, one column per model, the separation in each
cell — and beside it the invariance table, the same cells for a graph
against a relabelling of itself, which for the exact engines must be
zero and for the compressed ones measures the truncation error."""

import csv
import os
from collections import OrderedDict

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = ("passive", "bell", "active", "qubit", "qubit-ff")
RUNGS = ("control", "1fwl-blind", "2fwl-blind", "3fwl-blind")


def rows(path=None):
    with open(path or os.path.join(RESULTS, "results.csv")) as handle:
        return list(csv.DictReader(handle))


def table(invariance=False, path=None):
    cells, order = {}, OrderedDict()
    for row in rows(path):
        is_inv = row["pair"].endswith("~relabel")
        if is_inv != invariance:
            continue
        pair = row["pair"].replace("~relabel", "")
        order.setdefault((RUNGS.index(row["rung"]), pair), None)
        note = row["notes"]
        text = f'{float(row["separation"]):.1e}'
        if note.startswith("trimmed"):
            text = "—"
        cells[pair, row["model"]] = text
    lines = ["| pair | rung | " + " | ".join(MODELS) + " |",
             "|" + "---|" * (len(MODELS) + 2)]
    for (rank, pair) in order:
        rung = RUNGS[rank]
        entries = [cells.get((pair, model), "") for model in MODELS]
        lines.append(f"| {pair} | {rung} | " + " | ".join(entries)
                     + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("## Separations\n")
    print(table())
    print("\n## Relabelling invariance\n")
    print(table(invariance=True))
