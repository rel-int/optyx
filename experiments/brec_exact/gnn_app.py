"""Modal jobs for the comparison with probabilistic graph networks.

    modal run gnn_app.py::validate          # checks on small graphs
    modal deploy gnn_app.py
    python launch_gnn.py ...
"""

import json
import os
import time

import modal

app = modal.App("brec-gnn")
vol = modal.Volume.from_name("photonic-brec-rates")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "torch")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("brecphot", "gnn_bench"))

BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"
BUDGETS = (1000, 10000, 100000, 1000000, 10000000)
HOTELLING_BUDGET = 100000


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600)
def validate(big_graph6=None):
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    device = "cuda"
    out = {}

    def cycle(n):
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1
        return adj

    def two_triangles():
        adj = np.zeros((6, 6), dtype=np.int8)
        for block in ([0, 1, 2], [3, 4, 5]):
            for i in block:
                for j in block:
                    adj[i, j] = i != j
        return adj

    g, h = cycle(6), two_triangles()
    # 1. sampled means of the linear models against Wick
    for model in ("step-lin", "init-lin"):
        stats, counts = gb.sampled_statistics(model, g, 3, 400000, 3, device)
        exact = gb.exact_linear_profiles(model, g, 3, 3)
        out[f"{model}: sampled vs exact"] = {
            N: float(np.max(np.abs(np.sort(stats[N][0].mean(axis=0)) - exact[N]) / exact[N].max()))
            for N in exact}
    # 2. the coherent sampler against the exact rates at w = 1 and w = 20
    W = bp.path_matrix(bp.network(g, bp.cell(3), 0.9))
    for w in (1.0, 20.0):
        stats, counts = gb.sampled_statistics("coherent", g, 3, 400000, 3, device, brightness=w)
        for N in (1, 2, 3):
            exact = np.sort(bp.rate_tensors(W, N)["coherent"][tuple(
                np.array(list(gb.combinations(range(6), N))).T)] * w ** N)
            est = np.sort(stats[N][0].mean(axis=0))
            out[f"coherent w={w} order {N}: max rel err"] = float(np.max(np.abs(est - exact) / exact.max()))
    # 3. the protocol: false positives on G against relabelled G, and power on G vs H
    for model, kw in (("step-tanh", {}), ("coherent", {"brightness": 20.0}), ("init-tanh", {})):
        n = g.shape[0]
        index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T) for N in (1, 2, 3)}
        fp, power = 0, 0
        runs = 10
        for r in range(runs):
            rng = np.random.default_rng(r)
            sg, cg = gb.sampled_statistics(model, g, 3, 20000, 3, device, noise_seed=3 * r, **kw)
            sg2, cg2 = gb.sampled_statistics(model, bp.relabel(g, rng.permutation(n)), 3, 20000, 3, device, noise_seed=3 * r + 1, **kw)
            sh, ch = gb.sampled_statistics(model, h, 3, 20000, 3, device, noise_seed=3 * r + 2, **kw)
            fg = gb.batch_features(sg, cg, index, 3)
            fg2 = gb.batch_features(sg2, cg2, index, 3)
            fh = gb.batch_features(sh, ch, index, 3)
            fp += gb.permutation_test(fg, fg2, 3, splits=499, seed=r)["separated"]
            power += gb.permutation_test(fg, fh, 3, splits=499, seed=r)["separated"]
        out[f"{model}: protocol false positives / power at 2e4 samples"] = (fp / runs, power / runs)
    # 4. sampled Hotelling smoke test
    t0 = time.time()
    out["hotelling step-tanh 1e3"] = gb.sampled_hotelling("step-tanh", g, h, 3, 1000, 2, device)
    out["hotelling seconds"] = time.time() - t0
    # 5. timing at BREC sizes: 63 vertices, orders up to 3, 1e5 samples
    adj = bp.adjacency(big_graph6) if big_graph6 else None
    if adj is not None:
        for model, kw in (("step-tanh", {}), ("init-tanh", {}), ("coherent", {"brightness": 100.0})):
            torch.cuda.synchronize()
            t0 = time.time()
            gb.sampled_statistics(model, adj, 3, 100000, 3, device, **kw)
            torch.cuda.synchronize()
            out[f"{model} n={adj.shape[0]} 1e5 samples order 3: seconds"] = time.time() - t0
        t0 = time.time()
        gb.exact_linear_profiles("step-lin", adj, 3, 4)
        out["exact step-lin order 4 seconds"] = time.time() - t0
    return json.dumps(out, default=str)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=4 * 3600, max_containers=30, retries=1)
def compare_job(spec):
    """One pair, seed and model: the protocol at every budget, the exact
    linear column, and the sampled Hotelling test."""
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    device = "cuda"
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/{spec['model']}{spec.get('suffix', '')}.json"
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    adj_g, adj_h = bp.adjacency(spec["G"]), bp.adjacency(spec["H"])
    n = adj_g.shape[0]
    nmax = spec["nmax"]
    kw = {"brightness": spec["brightness"]} if spec["model"] == "coherent" else {}
    index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T)
             for N in range(1, nmax + 1)}
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"],
              "brightness": spec.get("brightness"), "n": n, "nmax": nmax, "budgets": {}}
    for T in spec["budgets"]:
        t1 = time.time()
        sg, cg = gb.sampled_statistics(spec["model"], adj_g, spec["seed"], T, nmax, device,
                                       noise_seed=2 * len(result["budgets"]), **kw)
        sh, ch = gb.sampled_statistics(spec["model"], adj_h, spec["seed"], T, nmax, device,
                                       noise_seed=2 * len(result["budgets"]) + 1, **kw)
        fg = gb.batch_features(sg, cg, index, nmax)
        fh = gb.batch_features(sh, ch, index, nmax)
        test = gb.permutation_test(fg, fh, nmax, seed=spec["seed"])
        gaps = {str(N): float(np.linalg.norm(np.sqrt(np.sort(sg[N][0].mean(0)))
                                             - np.sqrt(np.sort(sh[N][0].mean(0)))))
                for N in range(1, nmax + 1)}
        result["budgets"][str(T)] = {"test": test, "sampled_gap": gaps,
                                     "seconds": time.time() - t1}
    if spec["model"] in ("step-lin", "init-lin"):
        eg = gb.exact_linear_profiles(spec["model"], adj_g, spec["seed"], spec["exact_nmax"])
        eh = gb.exact_linear_profiles(spec["model"], adj_h, spec["seed"], spec["exact_nmax"])
        result["exact"] = {str(N): {
            "gap": bp.gap(eg[N], eh[N]),
            "max_rel_diff": float(np.max(np.abs(eg[N] - eh[N])) / max(eg[N].max(), eh[N].max()))}
            for N in eg}
    if spec.get("hotelling"):
        t1 = time.time()
        result["hotelling"] = gb.sampled_hotelling(spec["model"], adj_g, adj_h, spec["seed"],
                                                   spec["hotelling"], nmax, device, **kw)
        result["hotelling"]["seconds"] = time.time() - t1
    result["seconds"] = time.time() - t0
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=4 * 3600, max_containers=30, retries=1)
def false_positive_job(spec):
    """The protocol on G against relabelled copies of G, at every budget."""
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    device = "cuda"
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/{spec['model']}{spec.get('suffix', '')}-fp.json"
    if os.path.exists(path):
        return json.load(open(path))
    adj = bp.adjacency(spec["G"])
    n = adj.shape[0]
    nmax = spec["nmax"]
    kw = {"brightness": spec["brightness"]} if spec["model"] == "coherent" else {}
    index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T)
             for N in range(1, nmax + 1)}
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "runs": {}}
    for T in spec["budgets"]:
        hits = 0
        for r in range(spec["runs"]):
            rng = np.random.default_rng(1000 * spec["seed"] + r)
            sg, cg = gb.sampled_statistics(spec["model"], adj, spec["seed"], T, nmax, device,
                                           noise_seed=100 + 2 * r, **kw)
            sg2, cg2 = gb.sampled_statistics(spec["model"], bp.relabel(adj, rng.permutation(n)),
                                              spec["seed"], T, nmax, device, noise_seed=101 + 2 * r, **kw)
            fg = gb.batch_features(sg, cg, index, nmax)
            fg2 = gb.batch_features(sg2, cg2, index, nmax)
            hits += gb.permutation_test(fg, fg2, nmax, seed=r)["separated"]
        result["runs"][str(T)] = hits / spec["runs"]
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.local_entrypoint()
def validate_entry():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import brecphot as bp
    data = bp.load_brec(BREC_RAW)
    print(validate.remote(data["4-vertex_condition"][0][0]))
