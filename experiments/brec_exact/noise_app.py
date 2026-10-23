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

app = modal.App("brec-photon-noise-v2")
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


def all_moments(W, nmax, lam, device, chunk=4000, key_batch=1024):
    """moments[(N, doubled)] -> source -> array, for every pattern."""
    import photon_noise as pn
    out = {}
    for N in range(1, nmax + 1):
        for doubled, by_source in pn.moments_all(W, N, lam, device, chunk, key_batch).items():
            out[(N, doubled)] = by_source
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
    path = f"{folder}/g{spec['gamma']}-N{spec['nmax']}.json"
    try:
        vol.reload()
    except RuntimeError:
        pass
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


KMODE_EXPERIMENT = "brec-kmode"
KMODE_BRIGHTNESSES = (0.05, 0.2, 1.0)        # per source mode, for the separate placement


def rates_variances(mom, nmax, source, w):
    return combine(mom, nmax, source, w)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=7200, max_containers=10, retries=1)
def kmode_job(spec):
    """A chip with K source modes per vertex, one cell for every input.
    Three placements of the light: `separate`, one unit per source mode
    (K units per vertex and tick); `symmetric`, one unit in the symmetric
    combination of the K source modes; `single`, one unit in source mode 0.
    For every placement, every unit at every brightness of its grid, with
    the ideal counts cumulated over the orders.  `chunk` and `key_batch`
    bound the GPU memory, which grows with the number of slots."""
    import numpy as np
    import torch
    import brecphot as bp
    import coherent_gpu as cg
    import photon_noise as pn
    K = spec["local_modes"]
    folder = f"/vol/{KMODE_EXPERIMENT}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/K{K}-N{spec['nmax']}-v2.json"
    try:
        vol.reload()
    except RuntimeError:
        pass
    if os.path.exists(path):
        return json.load(open(path))
    device = "cuda"
    t0 = time.time()
    nmax = spec["nmax"]
    lam = pn.sources_lambda(fock=(K,) if K > 1 else ())
    cells = bp.cell_k(spec["seed"], K)
    mom = {}
    slots = {}
    for side in ("G", "H"):
        adj = bp.adjacency(spec[side])
        n = adj.shape[0]
        W = bp.path_matrix(bp.network_k(adj, cells, spec["gamma"], K))
        placements = {"separate": W, "symmetric": bp.symmetric_columns(W, n, K), "single": bp.single_columns(W, n, K)}
        if K == 1:
            placements = {"separate": W}
        for name, Wp in placements.items():
            slots[name] = Wp.shape[1]
            Wt = cg.to_torch(Wp, device)
            mom[(side, name)] = all_moments(Wt, nmax, lam, device, spec.get("chunk", 4000), spec.get("key_batch", 1024))
            del Wt
            torch.cuda.empty_cache()
    result = {"pair": spec["pair"], "seed": spec["seed"], "gamma": spec["gamma"], "nmax": nmax, "local_modes": K,
              "n": bp.adjacency(spec["G"]).shape[0], "slots": slots, "ideal": {}, "gaps": {}}
    grids = {"separate": {s: KMODE_BRIGHTNESSES for s in bp.SOURCES},
             # one unit per vertex and tick; brightness = detected photons per tick, i.e. K photons
             # injected times a detector efficiency of 0.05, 0.2 or 1 for the Fock pulse
             "symmetric": {"photons p=1": KMODE_BRIGHTNESSES, "photons p=0": KMODE_BRIGHTNESSES,
                           "coherent": tuple(K * w for w in KMODE_BRIGHTNESSES),
                           "thermal": tuple(K * w for w in KMODE_BRIGHTNESSES),
                           f"fock {K}": tuple(K * w for w in KMODE_BRIGHTNESSES)},
             "single": {"photons p=1": KMODE_BRIGHTNESSES, "coherent": tuple(K * w for w in KMODE_BRIGHTNESSES),
                        "thermal": tuple(K * w for w in KMODE_BRIGHTNESSES), f"fock {K}": tuple(K * w for w in KMODE_BRIGHTNESSES)}}
    if K == 1:
        grids = {"separate": {s: KMODE_BRIGHTNESSES for s in bp.SOURCES}}
    for name, by_source in grids.items():
        result["ideal"][name] = {}
        result["gaps"][name] = {}
        for source, ws in by_source.items():
            result["ideal"][name][source] = {}
            for w in ws:
                rv_g = {N: ({source: r}, {source: v}) for N, (r, v) in combine(mom[("G", name)], nmax, source, w).items()}
                rv_h = {N: ({source: r}, {source: v}) for N, (r, v) in combine(mom[("H", name)], nmax, source, w).items()}
                result["ideal"][name][source][str(w)] = {str(N): c for N, c in pn.ideal_counts(rv_g, rv_h, nmax)[source].items()}
            result["gaps"][name][source] = {}
            for N in range(1, nmax + 1):
                fg = np.sort(mom[("G", name)][(N, ())][source]); fh = np.sort(mom[("H", name)][(N, ())][source])
                result["gaps"][name][source][str(N)] = float(np.max(np.abs(fg - fh)) / max(fg.max(), fh.max()))
    result["seconds"] = time.time() - t0
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result
