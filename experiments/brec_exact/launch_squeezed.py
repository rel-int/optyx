"""Spawn the squeezed-light jobs: the 16-pair subset by default, or every
pair with --all, at order 4 up to `--order4` vertices and 3 above.

    python launch_squeezed.py [--all] [--pairs A,B] [--seeds 100,101,102,103]
"""
import argparse
import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402
from launch_kmode import KMODE_PAIRS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--order4", type=int, default=63, help="largest graph run at order 4, order 3 above")
    parser.add_argument("--out", default=None, help="write the results here as JSON")
    args = parser.parse_args()
    data = bp.load_brec(BREC_RAW)
    if args.pairs:
        names = args.pairs.split(",")
    elif args.all:
        names = [f"{c}_{i}" for c in data for i in range(len(data[c]))]
    else:
        names = list(KMODE_PAIRS)
    job = modal.Function.from_name("brec-squeezed", "squeezed_job")
    calls = []
    for pair in names:
        category, i = pair.rsplit("_", 1)
        g, h = data[category][int(i)]
        nmax = 4 if bp.adjacency(g).shape[0] <= args.order4 else 3
        for seed in map(int, args.seeds.split(",")):
            calls.append(job.spawn({"pair": pair, "seed": seed, "gamma": args.gamma, "nmax": nmax, "G": g, "H": h}))
    print("spawned", len(calls), flush=True)
    results = {}
    for call in calls:
        r = call.get()
        results[f"{r['pair']}/s{r['seed']}"] = r
        print(r["pair"], r["seed"], r["n"], r["nmax"], f"{r['seconds']:.1f}s", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
