"""Render ``results/results.csv`` and ``results/rotations.csv`` as the
tables of the README: the canonical cell by ticks beside PR #69's
passive column, the seed ensemble, the three certificates, the
rotation-system spreads and the relabelling rows."""

import csv
import os
import statistics
from collections import OrderedDict

from ladder import LADDER_RESULTS

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
RUNGS = ("control", "1fwl-blind", "2fwl-blind")
EXACT_ZERO = 1e-15


def rows(name):
    with open(os.path.join(RESULTS, name)) as handle:
        return list(csv.DictReader(handle))


def cell_text(value):
    return "exact 0" if value <= EXACT_ZERO else f"{value:.1e}"


def pairs_of(table, suffix=""):
    order = OrderedDict()
    for row in table:
        if row["pair"].endswith("~relabel") == bool(suffix):
            order.setdefault(
                (RUNGS.index(row["rung"]), row["pair"]), row["rung"])
    return [(pair, rung) for (_, pair), rung in order.items()]


def markdown(header, lines):
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(line) + " |" for line in lines])


def ladder_passive():
    """PR #69's passive separations at T = 8, for the same pairs."""
    with open(LADDER_RESULTS) as handle:
        return {row["pair"]: float(row["separation"])
                for row in csv.DictReader(handle)
                if row["model"] == "passive" and row["ticks"] == "8"}


def main_table(table):
    reference = ladder_passive()
    columns = [("canonical", "2"), ("canonical", "3"), ("canonical", "4"),
               ("grover", "4"), ("ladder", "4"), ("generic", "4")]
    values = {(r["pair"], r["cell"], r["ticks"]): float(r["separation"])
              for r in table if r["certificate"] == "two-photon"}
    lines = []
    for pair, rung in pairs_of(table):
        entries = [cell_text(values[pair, cell, ticks])
                   if (pair, cell, ticks) in values else ""
                   for cell, ticks in columns]
        entries.append(cell_text(reference[pair]) if pair in reference
                       else "")
        lines.append([pair, rung] + entries)
    return markdown(
        ["pair", "rung"] + [f"{cell} T={ticks}" for cell, ticks in columns]
        + ["#69 passive T=8"], lines)


def ensemble_table(table, ticks="4"):
    lines = []
    for pair, rung in pairs_of(table):
        values = [float(r["separation"]) for r in table
                  if r["pair"] == pair and r["cell"].startswith("seed-")
                  and r["ticks"] == ticks
                  and r["certificate"] == "two-photon"]
        if not values:
            continue
        lines.append([
            pair, rung, str(len(values)),
            str(sum(v > EXACT_ZERO for v in values)),
            cell_text(max(values)), cell_text(statistics.median(values)),
            cell_text(min(values))])
    return markdown(["pair", "rung", "seeds", "separating", "max",
                     "median", "min"], lines)


def certificate_table(table, cell="canonical", ticks="4"):
    certificates = ("two-photon", "distinguishable", "one-photon",
                    "coherent")
    values = {(r["pair"], r["certificate"]): float(r["separation"])
              for r in table if r["cell"] == cell and r["ticks"] == ticks}
    lines = [[pair, rung] + [
        cell_text(values[pair, c]) if (pair, c) in values else ""
        for c in certificates] for pair, rung in pairs_of(table)
        if any((pair, c) in values for c in certificates)]
    return markdown(["pair", "rung"] + list(certificates), lines)


def rotation_table(table, ticks="4"):
    cells = ("canonical", "ladder", "generic")
    spread = {}
    for row in table:
        if row["ticks"] != ticks:
            continue
        key = (row["pair"], row["cell"])
        spread[key] = max(spread.get(key, 0.), float(row["spread"]))
    lines = [[pair, rung] + [
        cell_text(spread[pair, cell]) if (pair, cell) in spread else ""
        for cell in cells] for pair, rung in pairs_of(table)]
    return markdown(["pair", "rung"] + list(cells), lines)


def invariance_table(table):
    values = {(r["pair"], r["cell"]): float(r["separation"]) for r in table
              if r["pair"].endswith("~relabel") and r["ticks"] == "4"}
    cells = sorted({cell for _, cell in values})
    lines = [[pair.replace("~relabel", ""), rung] + [
        cell_text(values[pair, cell]) if (pair, cell) in values else ""
        for cell in cells] for pair, rung in pairs_of(table, "~relabel")]
    return markdown(["pair", "rung"] + cells, lines)


if __name__ == "__main__":
    table = rows("results.csv")
    for title, text in (
            ("Separations", main_table(table)),
            ("Seed ensemble (T = 4)", ensemble_table(table)),
            ("Certificates (canonical, T = 4)", certificate_table(table)),
            ("Rotation systems (T = 4)",
             rotation_table(rows("rotations.csv"))),
            ("Relabelling invariance (T = 4)", invariance_table(table))):
        print(f"## {title}\n\n{text}\n")
