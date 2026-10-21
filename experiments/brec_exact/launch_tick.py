"""Spawn the forward-pass test on the deployed brec-tick app.

    python launch_tick.py run [--pairs A,B] [--seeds 100,101,102,103]
                              [--sources coherent-w1,...] [--budgets 1000,...]
    python launch_tick.py status
"""

import argparse
import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_tick as st  # noqa: E402
from launch import SEEDS, pairs_of  # noqa: E402
from tick_app import EXPERIMENT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--sources", default=",".join(st.SOURCES))
    parser.add_argument("--budgets", default=",".join(map(str, st.BUDGETS)))
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--app", default="brec-tick")
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--wait", action="store_true", help="run one job and print its summary")
    args = parser.parse_args()
    if args.command == "status":
        status = modal.Function.from_name("brec-photonic", "status")
        print(json.dumps(status.remote(args.experiment), indent=1))
        return
    names = set(args.pairs.split(",")) if args.pairs else None
    job = modal.Function.from_name(args.app, "tick_job")
    count = 0
    for pair, (g, h) in pairs_of(names).items():
        for seed in map(int, args.seeds.split(",")):
            spec = {"experiment": args.experiment, "pair": pair, "seed": seed, "G": g, "H": h,
                    "sources": args.sources.split(","),
                    "budgets": [int(b) for b in args.budgets.split(",")],
                    "anchors": not args.no_anchors}
            if args.wait:
                print(json.dumps(job.remote(spec), indent=1))
            else:
                job.spawn(spec)
            count += 1
    print("spawned", count)


if __name__ == "__main__":
    main()
