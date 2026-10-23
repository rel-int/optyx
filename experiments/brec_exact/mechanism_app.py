"""Which photon blocks carry the difference between two graphs: the gap of
synthetic sources with a single non-zero block weight, next to the real ones.

    modal run mechanism_app.py
"""
import json
import os

import modal

app = modal.App("brec-mechanism")
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scipy", "torch")
         .add_local_python_source("brecphot", "coherent_gpu"))

PAIRS = {"Distance_regular_1": 4, "Distance_regular_18": 4, "Distance_regular_19": 4, "Distance_regular_10": 4,
         "Distance_regular_0": 3, "Strongly_regular_0": 4, "Strongly_regular_1": 4, "4-vertex_condition_0": 3,
         "CFI_17": 3, "CFI_12": 4, "Basic_0": 4, "Extension_11": 4}


def lam_from_f(f2=0.0, f3=0.0, f4=0.0):
    """Moments from block weights: lam(m) = sum over partitions of prod f(|B|), f(1) = 1."""
    table = {1: 1.0, 2: 1 + f2, 3: 1 + 3 * f2 + f3, 4: 1 + 6 * f2 + 3 * f2 ** 2 + 4 * f3 + f4}
    return lambda m: table.get(m, 0.0)


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=3600)
def gaps(spec):
    import numpy as np
    import brecphot as bp
    import coherent_gpu as cg
    bp.LAMBDA.update({"pair": lam_from_f(f2=-1.0), "triple": lam_from_f(f3=1.0), "quad": lam_from_f(f4=1.0),
                      "laser f2 only": lam_from_f(f2=-0.5), "laser f3 only": lam_from_f(f3=2 / 3), "laser f4 only": lam_from_f(f4=-35 / 8),
                      "photons no f3": lam_from_f(f2=-1.0, f3=0.0, f4=-6.0), "photons no f4": lam_from_f(f2=-1.0, f3=2.0, f4=0.0),
                      "photons f2 only": lam_from_f(f2=-1.0), "laser no f3": lam_from_f(f2=-0.5, f3=0.0, f4=-35 / 8)})
    sources = ["thermal", "pair", "triple", "quad", "photons p=1", "photons p=0", "coherent", "laser f2 only", "laser f3 only", "laser f4 only",
               "photons no f3", "photons no f4", "photons f2 only", "laser no f3"]
    out = {}
    for side in ("G", "H"):
        adj = bp.adjacency(spec[side])
        W = cg.to_torch(bp.path_matrix(bp.network(adj, bp.cell(spec["seed"]), 0.9)), "cuda")
        out[side], _ = cg.profiles_torch(W, spec["nmax"], 1.0, sources)
    result, counts = {}, {}
    d2 = {s: 0.0 for s in sources}
    for N in range(1, spec["nmax"] + 1):
        result[N], counts[N] = {}, {}
        for i, s in enumerate(sources):
            g, h = out["G"][N][i], out["H"][N][i]
            scale = max(np.abs(g).max(), np.abs(h).max(), 1e-300)
            result[N][s] = {"gap": float(np.max(np.abs(g - h)) / scale), "scale": float(scale)}
            # Poisson-limit ideal count at w = 1: the variance of a coincidence is its rate
            noise = g + h
            mask = noise > 0
            d2[s] += float(np.sum((g - h)[mask] ** 2 / noise[mask]))
            counts[N][s] = 4 * np.log(20) / d2[s] if d2[s] > 0 else float("inf")
    return {"pair": spec["pair"], "seed": spec["seed"], "n": int(adj.shape[0]), "gaps": result, "poisson_counts": counts}


@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0, HERE)
    import brecphot as bp
    data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
    specs = []
    for pair, nmax in PAIRS.items():
        c, i = pair.rsplit("_", 1)
        g, h = data[c][int(i)]
        for seed in (100, 101):
            specs.append({"pair": pair, "seed": seed, "nmax": nmax, "G": g, "H": h})
    results = list(gaps.map(specs))
    json.dump(results, open(f"{HERE}/results/raw_mechanism.json", "w"))
    for r in results:
        print(r["pair"], "seed", r["seed"], "n", r["n"])
        top = max(r["poisson_counts"], key=int)
        ph = r["poisson_counts"][top]["photons p=1"]
        print("   counts / photons at the top order: " + "  ".join(f"{s}={v / ph:.2f}" for s, v in r["poisson_counts"][top].items()))
