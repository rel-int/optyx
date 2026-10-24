"""Gaussian squeezed light on the GPU: exact rates, noise and ideal sample
counts, with thermal light and the phase-locked real Gaussian amplitude from
the same moments as checks against the earlier runs.

    modal deploy squeezed_app.py
    modal run squeezed_app.py::bench --pairs Strongly_regular_2,CFI_12 --nmax 4
    python launch_squeezed.py               # the 16-pair subset
"""
import json
import os
import time

import modal

app = modal.App("brec-squeezed")
vol = modal.Volume.from_name("photonic-brec-rates")
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "torch")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("brecphot", "squeezed"))

BRIGHTNESSES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0)
KINDS = ("squeezed", "gaussian real", "thermal")
EXPERIMENT = "brec-squeezed"


def run(spec, device):
    import numpy as np
    import torch
    import brecphot as bp
    import squeezed as sq
    nmax = spec["nmax"]
    t0 = time.time()
    timing = {}
    mom = {}
    for side in ("G", "H"):
        t = time.time()
        W = bp.path_matrix(bp.network(bp.adjacency(spec[side]), bp.cell(spec["seed"]), spec["gamma"]))
        Wt = torch.as_tensor(W, device=device)
        timing[f"path {side}"] = time.time() - t
        G, T = sq.gram(Wt)
        mom[side] = {}
        for N in range(1, nmax + 1):
            t = time.time()
            for d, m in sq.moments(G, T, N, chunk=spec.get("chunk", 4096), key_batch=spec.get("key_batch", 1 << 16),
                                   by_size=True, to_numpy=False).items():
                mom[side][(N, d)] = m
            if device == "cuda":
                torch.cuda.synchronize()
            timing[f"N{N} {side}"] = time.time() - t
        del Wt, G, T
    result = {"pair": spec["pair"], "seed": spec["seed"], "gamma": spec["gamma"], "nmax": nmax,
              "n": bp.adjacency(spec["G"]).shape[0], "ideal": {}, "gaps": {}}
    for kind in KINDS:
        result["ideal"][kind], result["gaps"][kind] = {}, {}
        for w in BRIGHTNESSES:
            counts = sq.ideal_counts(mom["G"], mom["H"], nmax, w, kind)
            result["ideal"][kind][str(w)] = {str(N): c for N, c in counts.items()}
            # the rates of squeezed light mix powers of w; the others are w^N times a constant
            if kind == "squeezed" or w == 1.0:
                gaps = {str(N): g for N, g in sq.gaps(mom["G"], mom["H"], nmax, w, kind).items()}
                result["gaps"][kind][str(w)] = gaps
    # the j-components separately: each one is a graph invariant on its own
    result["component_gaps"] = {}
    for N in range(1, nmax + 1):
        rows = [sq.relative_gap(mom["G"][(N, ())][j], mom["H"][(N, ())][j]) for j in range(N + 1)]
        result["component_gaps"][str(N)] = rows
    del mom
    result["timing"] = timing
    result["seconds"] = time.time() - t0
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=86400, max_containers=10, retries=1)
def squeezed_job(spec):
    folder = f"/vol/{EXPERIMENT}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/g{spec['gamma']}-N{spec['nmax']}.json"
    try:
        vol.reload()
    except RuntimeError:
        pass
    if os.path.exists(path) and not spec.get("force"):
        return json.load(open(path))
    result = run(spec, "cuda")
    if not spec.get("dry"):
        os.makedirs(folder, exist_ok=True)
        json.dump(result, open(path, "w"))
        vol.commit()
    return result


@app.local_entrypoint()
def bench(pairs: str = "Strongly_regular_2", nmax: int = 4, seed: int = 100, chunk: int = 4096, key_batch: int = 65536):
    import sys
    sys.path.insert(0, HERE)
    import brecphot as bp
    from gpu_app import BREC_RAW
    data = bp.load_brec(BREC_RAW)
    calls = []
    for pair in pairs.split(","):
        category, i = pair.rsplit("_", 1)
        g, h = data[category][int(i)]
        calls.append(squeezed_job.spawn({"pair": pair, "seed": seed, "gamma": 0.9, "nmax": nmax, "G": g, "H": h,
                                         "chunk": chunk, "key_batch": key_batch, "dry": True, "force": True}))
    for call in calls:
        r = call.get()
        print(r["pair"], r["n"], f"{r['seconds']:.1f}s", {k: round(v, 2) for k, v in r["timing"].items()})
