"""GPU runs for coherent light: benchmark, and full-BREC jobs at any
brightness.

    modal deploy gpu_app.py
    modal run gpu_app.py::bench_entry
"""

import json
import os
import time

import modal

app = modal.App("brec-photonic-gpu")
vol = modal.Volume.from_name("photonic-brec-rates")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "torch")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("brecphot", "coherent_gpu"))

BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"
BRIGHTNESSES = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=3600)
def bench(specs, samples=20000):
    import torch
    import brecphot as bp
    import coherent_gpu as cg
    device = "cuda"
    out = {"gpu": torch.cuda.get_device_name(0), "cases": []}
    for graph6, nmax in specs:
        adj = bp.adjacency(graph6)
        res = cg.benchmark(adj, bp.cell(100), 0.9, nmax, device)
        W = cg.to_torch(bp.path_matrix(bp.network(adj, bp.cell(100), 0.9)), device)
        torch.cuda.synchronize()
        t0 = time.time()
        moments = cg.intensity_moments(W, nmax, samples, device)
        torch.cuda.synchronize()
        res["moments_seconds"] = time.time() - t0
        res["moments_samples"] = samples
        # the relative noise of the top-order variance estimate, from two halves
        half = cg.intensity_moments(W, nmax, samples // 2, device, seed=1)
        v1, _ = cg.coherent_variance(moments, nmax, 1.0)
        v2, _ = cg.coherent_variance(half, nmax, 1.0)
        res["variance_rel_scatter"] = float(np.median(np.abs(v1 - v2) / np.maximum(v1, 1e-300)))
        out["cases"].append(res)
    return json.dumps(out)


import numpy as np  # noqa: E402  (used in bench, imported after modal for the image)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=4 * 3600, max_containers=20, retries=1)
def pair_job(spec):
    """Exact profiles of G and H for all sources, coherent noise moments, and
    the ideal sample counts of coherent light at several brightnesses."""
    import torch
    import brecphot as bp
    import coherent_gpu as cg
    device = "cuda"
    folder = f"/vol/{spec['experiment']}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/g{spec['gamma']}.json"
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    cells = bp.cell(spec["seed"])
    prof, order, moments = {}, {}, {}
    for side in ("G", "H"):
        adj = bp.adjacency(spec[side])
        W = cg.to_torch(bp.path_matrix(bp.network(adj, cells, spec["gamma"])), device)
        prof[side], order[side] = cg.profiles_torch(W, spec["nmax"], bp.W)
        moments[side] = cg.intensity_moments(W, spec["nmax"], spec["samples"], device)
        del W
        torch.cuda.empty_cache()
    n = bp.adjacency(spec["G"]).shape[0]
    result = {"pair": spec["pair"], "seed": spec["seed"], "gamma": spec["gamma"],
              "n": n, "nmax": spec["nmax"], "seconds": time.time() - t0, "orders": {},
              "coherent_samples_by_brightness": {}}
    for N in prof["G"]:
        entry = {}
        for i, source in enumerate(bp.SOURCES):
            fg, fh = prof["G"][N][i], prof["H"][N][i]
            scale = max(fg.max(), fh.max())
            entry[source] = {"gap": bp.gap(fg, fh),
                             "max_rel_diff": float(np.max(np.abs(fg - fh)) / scale),
                             "ideal_samples_poisson": bp.ideal_samples(fg, fh)}
        result["orders"][str(N)] = entry
    i = bp.SOURCES.index("coherent")
    pg = {N: prof["G"][N][i] for N in prof["G"]}
    ph = {N: prof["H"][N][i] for N in prof["H"]}
    og = {N: order["G"][N][i] for N in order["G"]}
    oh = {N: order["H"][N][i] for N in order["H"]}
    for w in spec["brightnesses"]:
        result["coherent_samples_by_brightness"][str(w)] = {
            str(N): cg.ideal_samples_at(pg, og, moments["G"], ph, oh, moments["H"], N, w)
            for N in range(1, spec["nmax"] + 1)}
    os.makedirs(folder, exist_ok=True)
    np.savez(f"{folder}/g{spec['gamma']}.npz",
             **{f"F{s}{N}": prof[s][N] for s in prof for N in prof[s]},
             **{f"M{s}{N}": moments[s][N] for s in moments for N in moments[s]})
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.local_entrypoint()
def bench_entry():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import brecphot as bp
    data = bp.load_brec(BREC_RAW)
    specs = [(data["Strongly_regular"][0][0], 4),
             (data["CFI"][76][0], 4),
             (data["4-vertex_condition"][0][0], 4),
             (data["CFI"][99][0], 3)]
    print(bench.remote(specs))
