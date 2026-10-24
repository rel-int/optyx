"""Spawn the measurement-feedforward jobs on the 16-pair subset: laser and
thermal light in the chip run tick by tick, the count of every detector
setting the phase of its chip's memory mode `delay` ticks later, against
the same chip without feedforward (kappa = 0).

    python launch_ff.py [--pairs A,B] [--seeds 100,101,102,103] [--top 100000000]
"""
import argparse
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402
from gpu_app import BREC_RAW  # noqa: E402
from launch_kmode import KMODE_PAIRS  # noqa: E402

EXPERIMENT = "ff-compare"
MODELS = ("loop-coherent", "loop-thermal")
BRIGHTNESS = 100.0
KAPPAS = (0.0, 0.5, 2.0)
DELAYS = (1, 5)


def configs():
    """(model, kappa, delay): kappa = 0 once per model, the others at every delay."""
    for model in MODELS:
        yield model, 0.0, 1
        for kappa in KAPPAS[1:]:
            for delay in DELAYS:
                yield model, kappa, delay


def suffix(kappa, delay, brightness=BRIGHTNESS):
    return f"-w{brightness:g}-k{kappa:g}-d{delay}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--top", type=int, default=10 ** 8, help="largest budget, samples per graph")
    args = parser.parse_args()
    budgets = [10 ** k for k in range(3, 20) if 10 ** k <= args.top]
    data = bp.load_brec(BREC_RAW)
    names = args.pairs.split(",") if args.pairs else list(KMODE_PAIRS)
    job = modal.Function.from_name("brec-gnn-v5", "compare_job")
    calls = []
    for pair in names:
        category, i = pair.rsplit("_", 1)
        g, h = data[category][int(i)]
        for seed in map(int, args.seeds.split(",")):
            for model, kappa, delay in configs():
                calls.append(job.spawn({"pair": pair, "seed": seed, "model": model, "brightness": BRIGHTNESS,
                                        "kappa": kappa, "delay": delay, "G": g, "H": h, "nmax": KMODE_PAIRS[pair],
                                        "budgets": budgets, "experiment": EXPERIMENT,
                                        "suffix": suffix(kappa, delay)}))
    print("spawned", len(calls), flush=True)
    for call in calls:
        r = call.get()
        print(r["pair"], r["seed"], r["model"], r["done"], flush=True)


if __name__ == "__main__":
    main()
