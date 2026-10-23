"""Spawn the jobs of one experiment on the deployed app.

    python launch.py column [--pairs Basic_0,CFI_5] [--copies 32] [--chunk 8]
    python launch.py sweep  [--pairs ...] [--gammas 0.5,0.7,...]
    python launch.py test   [--pairs ...]        # BREC's test on finished copies
    python launch.py status column|loss-sweep
    python launch.py smoke                        # one tiny job of each kind
"""

import os
import argparse
import json
import sys

import modal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402

BREC = "/home/user/graphpku/brec/customize/Data/raw"
SEEDS = (100, 101, 102, 103)
GAMMAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)


def pairs_of(names):
    data = bp.load_brec(BREC)
    out = {}
    for category, indices in bp.SUBSET.items():
        for i in indices:
            name = f"{category}_{i}"
            if names is None or name in names:
                out[name] = data[category][i]
    return out


def copy_tags(pair, n, copies):
    """32 copies of G and of H for the pair test, 32 more of G for the
    reliability check; copy 0 of each is the unrelabelled graph."""
    rng = np.random.default_rng(sum(map(ord, pair)) + 1)
    tags = []
    for side, count in (("G", 2 * copies), ("H", copies)):
        for k in range(count):
            perm = None if k == 0 else rng.permutation(n).tolist()
            tags.append((f"{side}{k}", side, perm))
    return tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--copies", type=int, default=bp.NUM_RELABEL)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--gammas", default=",".join(map(str, GAMMAS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--nmax", type=int, default=4)
    args = parser.parse_args()
    names = None if args.pairs is None else set(args.pairs.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    profile_job = modal.Function.from_name("brec-photonic", "profile_job")
    sweep_job = modal.Function.from_name("brec-photonic", "sweep_job")
    column_test = modal.Function.from_name("brec-photonic", "column_test")
    status = modal.Function.from_name("brec-photonic", "status")

    if args.command == "status":
        print(json.dumps(status.remote(args.pairs or "column-pilot"), indent=1))
        return

    pairs = pairs_of(names)
    spawned = 0
    if args.command in ("column", "smoke"):
        copies = 2 if args.command == "smoke" else args.copies
        for pair, (g, h) in pairs.items():
            n = bp.adjacency(g).shape[0]
            tags = copy_tags(pair, n, copies)
            for seed in seeds:
                for start in range(0, len(tags), args.chunk):
                    chunk = [(tag, g if side == "G" else h, perm)
                             for tag, side, perm in tags[start:start + args.chunk]]
                    spec = {"experiment": "column-pilot", "pair": pair,
                            "seed": seed, "gamma": 0.9, "nmax": args.nmax,
                            "copies": chunk}
                    if args.command == "smoke":
                        print(profile_job.remote(spec))
                        print(json.dumps(column_test.remote(
                            {"pair": pair, "seed": seed,
                             "tags": [t for t, _, _ in tags]}), indent=1)[:3000])
                    else:
                        profile_job.spawn(spec)
                    spawned += 1
    if args.command in ("sweep", "smoke"):
        gammas = [0.5] if args.command == "smoke" else [float(x) for x in args.gammas.split(",")]
        for pair, (g, h) in pairs.items():
            for seed in seeds:
                for gamma in gammas:
                    spec = {"pair": pair, "seed": seed, "gamma": gamma,
                            "nmax": args.nmax, "G": g, "H": h}
                    if args.command == "smoke":
                        print(json.dumps(sweep_job.remote(spec))[:1500])
                    else:
                        sweep_job.spawn(spec)
                    spawned += 1
    if args.command == "test":
        for pair, (g, h) in pairs.items():
            n = bp.adjacency(g).shape[0]
            tags = [t for t, _, _ in copy_tags(pair, n, args.copies)]
            for seed in seeds:
                column_test.spawn({"pair": pair, "seed": seed, "tags": tags})
                spawned += 1
    print("spawned", spawned)


if __name__ == "__main__":
    main()
