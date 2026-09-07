"""
The pieces of ``experiments/kwl_ladder`` this experiment reuses: its
graph encoding and functor pattern, its port-symmetric interferometer,
its dataset and its scoring — so that numbers are comparable with
PR #69. The ladder's ``run.py`` is loaded by path under another name
because this directory has a ``run.py`` of its own.
"""

import importlib.util
import os
import sys

LADDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kwl_ladder")
sys.path.insert(0, LADDER)

from exact import separation  # noqa: E402
from models import Model, adjacency, coupler  # noqa: E402


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(LADDER, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder_run = load_module("kwl_ladder_run", "run.py")
load_pairs, graphs_of, relabelled = (
    ladder_run.load_pairs, ladder_run.graphs_of, ladder_run.relabelled)
LADDER_RESULTS = os.path.join(LADDER, "results", "results.csv")

__all__ = ["Model", "adjacency", "coupler", "separation", "load_pairs",
           "graphs_of", "relabelled", "LADDER_RESULTS"]
