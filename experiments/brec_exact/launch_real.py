"""Spawn the phase-locked uniform source on every pair, at the order of the
exact-noise run and with Monte Carlo samples set by the graph size.

    python launch_real.py [--pairs A,B] [--seeds 100,101,102,103] [--missing]
"""
import argparse
import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def samples_for(n):
    return 1000000 if n <= 32 else 300000 if n <= 40 else 100000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--missing", action="store_true")
    args = parser.parse_args()
    names = set(args.pairs.split(",")) if args.pairs else None
    noise = json.load(open(f"{HERE}/results/raw_noise.json"))
    nmax = {}
    for r in noise.values():
        if isinstance(r, dict):
            nmax[r["pair"]] = max(nmax.get(r["pair"], 0), r["nmax"])
    done = set()
    if args.missing and os.path.exists(f"{HERE}/results/raw_real.json"):
        raw = json.load(open(f"{HERE}/results/raw_real.json"))
        done = {(r["pair"], r["seed"]) for r in raw.values() if isinstance(r, dict)}
    data = bp.load_brec(BREC_RAW)
    job = modal.Function.from_name("brec-photon-noise-v2", "real_job")
    count = 0
    for c in data:
        for i, (g, h) in enumerate(data[c]):
            pair = f"{c}_{i}"
            if names is not None and pair not in names:
                continue
            n = bp.adjacency(g).shape[0]
            for seed in map(int, args.seeds.split(",")):
                if (pair, seed) in done:
                    continue
                job.spawn({"pair": pair, "seed": seed, "gamma": args.gamma, "nmax": nmax[pair],
                           "samples": samples_for(n), "G": g, "H": h})
                count += 1
    print("spawned", count)


if __name__ == "__main__":
    main()
