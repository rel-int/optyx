"""Spawn BREC's test on the exact profiles of the full run.

    python launch_full_test.py [--pairs A,B] [--experiment brec-full-coherent]
"""
import argparse
import json
import os
import sys
from statistics import median

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def pilot_noise():
    """Median rounding variance of a projected feature per order, over the
    pilot's relabelled copies (test.json files fetched by report.py)."""
    raw = json.load(open(f"{HERE}/results/raw.json"))
    per_order = {}
    for path, t in raw.items():
        if not path.endswith("test.json") or not isinstance(t, dict):
            continue
        for N, entry in t["orders"].items():
            for source in bp.SOURCES:
                per_order.setdefault(N, []).append(entry[source]["noise_variance"])
    return {N: median(v) for N, v in per_order.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--experiment", default="brec-full-coherent")
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()
    noise = pilot_noise()
    print("pilot rounding variance per order:", noise)
    data = bp.load_brec(BREC_RAW)
    names = None if args.pairs is None else set(args.pairs.split(","))
    job = modal.Function.from_name("brec-photonic", "full_test")
    count = 0
    for category in data:
        for i in range(len(data[category])):
            pair = f"{category}_{i}"
            if names is not None and pair not in names:
                continue
            for seed in map(int, args.seeds.split(",")):
                job.spawn({"experiment": args.experiment, "pair": pair, "seed": seed,
                           "gamma": args.gamma, "noise": noise})
                count += 1
    print("spawned", count)


if __name__ == "__main__":
    main()
