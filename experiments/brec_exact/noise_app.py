"""Exact noise of every source at any brightness on the GPU: ideal sample
counts for single photons, Fock states, coherent and thermal light.

    modal deploy noise_app.py
    modal run noise_app.py::validate          # brute-force checks, CPU
    modal run noise_app.py::bench             # timing on one pair
    python launch_noise.py [--all]            # the jobs
"""
import json
import os
import time

import modal

app = modal.App("brec-photon-noise")
vol = modal.Volume.from_name("photonic-brec-rates")
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "torch")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("brecphot", "coherent_gpu", "photon_noise")
         .add_local_dir(f"{HERE}/noise_tables", remote_path="/root/noise_tables")
         .add_local_file(f"{HERE}/test_photon_noise.py", remote_path="/root/test_photon_noise.py"))

BRIGHTNESSES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
CLASSICAL_BRIGHTNESSES = (2.0, 5.0, 10.0, 20.0, 100.0)
FOCK = (2, 5, 10)
EXPERIMENT = "brec-photon-noise"


@app.function(image=image, cpu=8, memory=16384, timeout=3600)
def validate():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "/root/test_photon_noise.py"], capture_output=True, text=True, cwd="/root")
    return out.stdout + out.stderr


def all_moments(W, nmax, lam, device):
    """moments[(N, doubled)] -> source -> array, for every pattern."""
    import photon_noise as pn
    from itertools import combinations
    out = {}
    for N in range(1, nmax + 1):
        for size in range(0, N + 1):
            for doubled in combinations(range(N), size):
                out[(N, doubled)] = pn.moments(W, N, doubled, lam, device)
    return out


def combine(mom, nmax, source, w):
    """Rates and variances of one source at brightness w, per order."""
    import numpy as np
    out = {}
    for N in range(1, nmax + 1):
        rates = w ** N * mom[(N, ())][source]
        second = np.zeros_like(rates)
        for (M, doubled), m in mom.items():
            if M == N:
                second += w ** (N + len(doubled)) * m[source]
        out[N] = (rates, np.maximum(second - rates ** 2, 0.0))
    return out


def counts_for(mom_g, mom_h, nmax, source, w):
    import photon_noise as pn
    rv_g = {N: ({source: r}, {source: v}) for N, (r, v) in combine(mom_g, nmax, source, w).items()}
    rv_h = {N: ({source: r}, {source: v}) for N, (r, v) in combine(mom_h, nmax, source, w).items()}
    return pn.ideal_counts(rv_g, rv_h, nmax)[source]


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=7200, max_containers=10, retries=1)
def noise_job(spec):
    """One pair and cell: exact rates and variances of every source for
    orders up to spec['nmax'], ideal counts at every brightness."""
    import numpy as np
    import torch
    import brecphot as bp
    import coherent_gpu as cg
    import photon_noise as pn
    folder = f"/vol/{EXPERIMENT}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/g{spec['gamma']}.json"
    if os.path.exists(path):
        return json.load(open(path))
    device = "cuda"
    t0 = time.time()
    lam = pn.sources_lambda(fock=FOCK)
    nmax = spec["nmax"]
    mom = {}
    for side in ("G", "H"):
        adj = bp.adjacency(spec[side])
        W = cg.to_torch(bp.path_matrix(bp.network(adj, bp.cell(spec["seed"]), spec["gamma"])), device)
        mom[side] = all_moments(W, nmax, lam, device)
        del W
        torch.cuda.empty_cache()
    result = {"pair": spec["pair"], "seed": spec["seed"], "gamma": spec["gamma"], "nmax": nmax,
              "n": bp.adjacency(spec["G"]).shape[0], "ideal": {}, "gaps": {}}
    for source in bp.SOURCES:
        grid = BRIGHTNESSES + (CLASSICAL_BRIGHTNESSES if source in ("coherent", "thermal") else ())
        result["ideal"][source] = {str(w): {str(N): c for N, c in counts_for(mom["G"], mom["H"], nmax, source, w).items()}
                                   for w in grid}
        result["gaps"][source] = {}
        for N in range(1, nmax + 1):
            fg, fh = np.sort(mom["G"][(N, ())][source]), np.sort(mom["H"][(N, ())][source])
            result["gaps"][source][str(N)] = float(np.max(np.abs(fg - fh)) / max(fg.max(), fh.max()))
    for K in FOCK:
        source = f"fock {K}"
        result["ideal"][source] = {str(float(K)): {str(N): c for N, c in counts_for(mom["G"], mom["H"], nmax, source, float(K)).items()}}
    result["seconds"] = time.time() - t0
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.local_entrypoint()
def bench(cat: str = "Strongly_regular", i: int = 0, nmax: int = 4, seed: int = 100):
    import sys
    sys.path.insert(0, HERE)
    import brecphot as bp
    data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
    g, h = data[cat][i]
    r = noise_job.remote({"pair": f"{cat}_{i}-bench", "seed": seed, "gamma": 0.9, "nmax": nmax, "G": g, "H": h})
    print(json.dumps({k: v for k, v in r.items() if k != "ideal"}, indent=1))
    for s, by_w in r["ideal"].items():
        print(s, {w: f"{c[str(nmax)]:.2e}" for w, c in by_w.items()})
