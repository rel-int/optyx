"""Spawn the exact-noise jobs.

    python launch_noise.py [--all] [--pairs A,B] [--seeds 100,101,102,103]
                           [--order4-max-n 40] [--order3-max-n 110]
"""
import argparse
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402
from launch import pairs_of  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--order4-max-n", type=int, default=40)
    parser.add_argument("--order3-max-n", type=int, default=110)
    args = parser.parse_args()
    names = set(args.pairs.split(",")) if args.pairs else None
    if args.all:
        data = bp.load_brec(BREC_RAW)
        pairs = {f"{c}_{i}": pair for c in data for i, pair in enumerate(data[c])
                 if names is None or f"{c}_{i}" in names}
    else:
        pairs = pairs_of(names)
    job = modal.Function.from_name("brec-photon-noise", "noise_job")
    count = 0
    for pair, (g, h) in pairs.items():
        n = bp.adjacency(g).shape[0]
        nmax = 4 if n <= args.order4_max_n else 3 if n <= args.order3_max_n else 2
        for seed in map(int, args.seeds.split(",")):
            job.spawn({"pair": pair, "seed": seed, "gamma": args.gamma, "nmax": nmax, "G": g, "H": h})
            count += 1
    print("spawned", count)


if __name__ == "__main__":
    main()
