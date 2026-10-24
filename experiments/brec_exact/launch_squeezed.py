"""Spawn the squeezed-light jobs: the 16-pair subset by default, or every
pair with --all, at the orders of the photons' exact-noise run: 4 up to
`--order4` vertices, 3 up to `--order3` and 2 above.  With --higher every
pair runs at order 4 with variances, and at order 5 with rates (so gaps)
up to `--rates5` vertices and variances up to `--var5`.

    python launch_squeezed.py [--all] [--higher] [--pairs A,B] [--seeds 100,101,102,103]
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
    parser.add_argument("--order4", type=int, default=63, help="largest graph run at order 4")
    parser.add_argument("--order3", type=int, default=110, help="largest graph run at order 3, order 2 above")
    parser.add_argument("--higher", action="store_true")
    parser.add_argument("--rates5", type=int, default=110, help="largest graph whose order-5 rates are computed")
    parser.add_argument("--var5", type=int, default=40, help="largest graph whose order-5 variances are computed")
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
        n = bp.adjacency(g).shape[0]
        nmax = 4 if n <= args.order4 else 3 if n <= args.order3 else 2
        vmax = nmax
        if args.higher:
            nmax = 5 if n <= args.rates5 else 4
            vmax = 5 if n <= args.var5 else 4
        for seed in map(int, args.seeds.split(",")):
            calls.append(job.spawn({"pair": pair, "seed": seed, "gamma": args.gamma, "nmax": nmax, "vmax": vmax,
                                    "G": g, "H": h, "chunk": 16384 if n > 63 else 4096}))
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
