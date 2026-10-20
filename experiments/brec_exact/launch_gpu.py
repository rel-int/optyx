"""Spawn the full-BREC coherent-light jobs on the GPU app.

    python launch_gpu.py [--seeds 100,101,102,103] [--gamma 0.9] [--samples 100000]
                         [--order4-max-n 110] [--categories CFI,Basic] [--dry]
"""

import os
import argparse
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW, BRIGHTNESSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--order4-max-n", type=int, default=110)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--experiment", default="brec-full-coherent")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    data = bp.load_brec(BREC_RAW)
    job = modal.Function.from_name("brec-photonic-gpu", "pair_job")
    categories = args.categories.split(",") if args.categories else [c for c, _ in bp.CATEGORIES]
    count, order4 = 0, 0
    for category in categories:
        for i, (g, h) in enumerate(data[category]):
            n = bp.adjacency(g).shape[0]
            nmax = 4 if n <= args.order4_max_n else 3
            order4 += nmax == 4
            for seed in map(int, args.seeds.split(",")):
                spec = {"experiment": args.experiment, "pair": f"{category}_{i}",
                        "seed": seed, "gamma": args.gamma, "nmax": nmax,
                        "samples": args.samples, "brightnesses": list(BRIGHTNESSES),
                        "G": g, "H": h}
                if not args.dry:
                    job.spawn(spec)
                count += 1
    print(f"spawned {count} jobs, {order4} pairs at order 4")


if __name__ == "__main__":
    main()
