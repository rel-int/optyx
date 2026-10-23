"""Spawn the K-source-mode chip jobs on the 15-pair subset.

    python launch_kmode.py [--modes 1,2,4,8] [--seeds 100,101,102,103] [--pairs A,B]
"""
import argparse
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402

# pair -> largest order (3 where the pair separates at order 3 and is large)
KMODE_PAIRS = {"Basic_0": 4, "Basic_13": 4, "Regular_47": 4, "Regular_10": 4, "Extension_11": 4, "Extension_13": 4,
               "Strongly_regular_0": 4, "Strongly_regular_2": 4, "Strongly_regular_1": 4,
               "CFI_5": 4, "CFI_12": 4, "CFI_17": 3, "4-vertex_condition_0": 3,
               "Distance_regular_1": 4, "Distance_regular_18": 4, "Distance_regular_19": 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="1,2,4,8")
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--chunk", type=int, default=4000, help="sets per GPU chunk, lower it on many slots")
    parser.add_argument("--key-batch", type=int, default=1024)
    parser.add_argument("--missing", action="store_true", help="skip the (pair, seed, K) already in results/raw_kmode.json")
    parser.add_argument("--max-inflight", type=int, default=0, help="wait so that at most this many jobs run at once (0: spawn all)")
    args = parser.parse_args()
    done = set()
    if args.missing:
        import json
        raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw_kmode.json")))
        done = {(r["pair"], r["seed"], r["local_modes"]) for r in raw.values() if isinstance(r, dict)}
    inflight = []
    data = bp.load_brec(BREC_RAW)
    names = set(args.pairs.split(",")) if args.pairs else set(KMODE_PAIRS)
    job = modal.Function.from_name("brec-photon-noise-v2", "kmode_job")
    count = 0
    for pair, nmax in KMODE_PAIRS.items():
        if pair not in names:
            continue
        category, i = pair.rsplit("_", 1)
        g, h = data[category][int(i)]
        for K in map(int, args.modes.split(",")):
            for seed in map(int, args.seeds.split(",")):
                if (pair, seed, K) in done:
                    continue
                if args.max_inflight and len(inflight) >= args.max_inflight:
                    inflight.pop(0).get()
                inflight.append(job.spawn({"pair": pair, "seed": seed, "gamma": args.gamma, "nmax": nmax, "local_modes": K,
                                           "G": g, "H": h, "chunk": args.chunk, "key_batch": args.key_batch}))
                count += 1
                print("spawned", pair, seed, K, flush=True)
    for call in inflight:
        call.get()
    print("spawned", count)


if __name__ == "__main__":
    main()
