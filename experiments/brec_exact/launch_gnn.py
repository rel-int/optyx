"""Spawn the comparison jobs on the deployed brec-gnn app.

    python launch_gnn.py compare [--pairs A,B] [--seeds 100,101,102,103]
                                 [--models step-lin,...] [--budgets 1000,...]
    python launch_gnn.py fp      [--pairs ...]   # false-positive validation
    python launch_gnn.py status
"""

import argparse
import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gnn_app import BREC_RAW, BUDGETS, HOTELLING_BUDGET  # noqa: E402
from launch import pairs_of  # noqa: E402

MODELS = ("step-lin", "step-tanh", "init-lin", "init-tanh", "coherent-100", "coherent-1")
FP_PAIRS = ("Basic_0", "Extension_0", "Strongly_regular_0", "CFI_12",
            "Distance_regular_1", "4-vertex_condition_0")


def spec_of(model, pair, g, h, seed, budgets, nmax, exact_nmax, hotelling):
    name, _, brightness = model.partition("-") if model.startswith("coherent") else (model, "", "")
    spec = {"pair": pair, "seed": seed, "model": name, "G": g, "H": h,
            "budgets": list(budgets), "nmax": nmax, "exact_nmax": exact_nmax,
            "hotelling": hotelling}
    if name == "coherent":
        spec["brightness"] = float(brightness)
        spec["suffix"] = f"-w{brightness}"
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    parser.add_argument("--nmax", type=int, default=3)
    parser.add_argument("--exact-nmax", type=int, default=4)
    parser.add_argument("--hotelling", type=int, default=HOTELLING_BUDGET)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    if args.command == "status":
        status = modal.Function.from_name("brec-photonic", "status")
        print(json.dumps(status.remote("gnn-compare"), indent=1))
        return
    names = set(args.pairs.split(",")) if args.pairs else None
    if args.command == "fp" and names is None:
        names = set(FP_PAIRS)
    pairs = pairs_of(names)
    compare = modal.Function.from_name("brec-gnn", "compare_job")
    fp = modal.Function.from_name("brec-gnn", "false_positive_job")
    count = 0
    for pair, (g, h) in pairs.items():
        for seed in seeds:
            for model in args.models.split(","):
                spec = spec_of(model, pair, g, h, seed, budgets, args.nmax,
                               args.exact_nmax, args.hotelling)
                if args.command == "compare":
                    compare.spawn(spec)
                else:
                    spec["runs"] = args.runs
                    spec["budgets"] = [b for b in budgets if b <= 1000000]
                    fp.spawn(spec)
                count += 1
    print("spawned", count)


if __name__ == "__main__":
    main()
