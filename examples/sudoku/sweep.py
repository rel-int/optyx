"""Generate the sweep configurations for the Modal runs.

Usage::

    python sweep.py pilots > configs.json
    python sweep.py ladder quantum > configs.json
"""

import json
import sys


def pilots():
    """One representative per family at two ticks, with one-tick
    ablations: recurrence off."""
    shared = {"n_steps": 600, "eval_every": 100}
    return [
        {"family": "quantum", "cell_depth": 8, "cons_depth": 8,
         "feedback": 2, "ticks": 2, **shared},
        {"family": "quantum", "cell_depth": 4, "cons_depth": 4,
         "feedback": 2, "ticks": 2, **shared},
        {"family": "quantum", "cell_depth": 8, "cons_depth": 8,
         "feedback": 2, "ticks": 1, **shared},
        {"family": "square", "bond": 6, "feedback": 2, "ticks": 2,
         **shared},
        {"family": "square", "bond": 6, "feedback": 1, "ticks": 2,
         **shared},
        {"family": "exp", "bond": 6, "feedback": 2, "ticks": 2, **shared},
        {"family": "square", "bond": 6, "feedback": 2, "ticks": 1,
         **shared},
        {"family": "born", "depth": 32, "ticks": 2, **shared},
    ]


def ladder(family):
    """Progressively increasing parameter counts for one family."""
    shared = {"ticks": 2, "n_steps": 2000, "eval_every": 200}
    if family == "quantum":
        return [
            {"family": "quantum", "cell_depth": depth,
             "cons_depth": depth, "feedback": 2, **shared}
            for depth in (2, 4, 8, 16)]
    if family == "born":
        return [
            {"family": "born", "depth": depth, **shared}
            for depth in (8, 32, 128, 256)]
    return [
        {"family": family, "bond": bond, "feedback": 2, **shared}
        for bond in (4, 6, 8, 12, 14)]


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "pilots"
    configs = pilots() if kind == "pilots" else ladder(sys.argv[2])
    print(json.dumps(configs, indent=2))
