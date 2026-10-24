"""Modal jobs for the comparison with probabilistic graph networks.

    modal run gnn_app.py::validate          # checks on small graphs
    modal deploy gnn_app.py
    python launch_gnn.py ...
"""

import json
import os
import time

import modal

app = modal.App("brec-gnn-v5")
vol = modal.Volume.from_name("photonic-brec-rates")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "torch")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("brecphot", "gnn_bench", "coherent_gpu"))

BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"
BUDGETS = (1000, 10000, 100000, 1000000, 10000000, 100000000)
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
    """One pair, seed and model, resumable.

    The two graphs are sampled once up to the largest budget, with the
    protocol run and a result file written at every budget on the way;
    budgets whose file exists are skipped, and a checkpoint of the
    accumulators and of the samplers' states lets a later call continue
    from the last budget to a higher one.  The exact linear column and the
    sampled Hotelling test are cached in their own files.
    """
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    device = "cuda"
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    stem = f"{folder}/{spec['model']}{spec.get('suffix', '')}"
    os.makedirs(folder, exist_ok=True)
    adj = {"G": bp.adjacency(spec["G"]), "H": bp.adjacency(spec["H"])}
    n = adj["G"].shape[0]
    nmax = spec["nmax"]
    kw = {"brightness": spec["brightness"]} if spec["model"] in ("coherent", "loop-coherent", "loop-thermal") else {}
    if spec["model"].startswith("loop"):
        kw.update(kappa=spec.get("kappa", 0.0), delay=spec.get("delay", 1))
    stationary = spec["model"].startswith(("step", "loop"))
    budgets = sorted(spec["budgets"])
    todo = [T for T in budgets if not os.path.exists(f"{stem}-T{T}.json")]
    summary = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"],
               "n": n, "nmax": nmax, "done": []}
    if todo:
        index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T)
                 for N in range(1, nmax + 1)}
        samplers, accs = {}, {}
        for i, side in enumerate(("G", "H")):
            samplers[side] = gb.Sampler(spec["model"], adj[side], spec["seed"], device,
                                        noise_seed=i, chunk=gb.chunk_size(n, nmax), **kw)
            accs[side] = gb.Accumulator(n, nmax, device=device)
        t_done = 0
        if os.path.exists(f"{stem}-ckpt.npz"):
            with np.load(f"{stem}-ckpt.npz") as z:
                ck = {k: z[k] for k in z.files}
            t_done = int(ck["t_done"])
            for side in ("G", "H"):
                accs[side].load_state_dict({k[2:]: v for k, v in ck.items() if k.startswith(f"{side}:")}, device)
                samplers[side].load_state_dict({k[2:]: v for k, v in ck.items() if k.startswith(f"{side}:")}, device)
        for T in budgets:
            if T <= t_done:
                continue
            t1 = time.time()
            for side in ("G", "H"):
                for y in samplers[side].chunks(T - t_done):
                    accs[side].add(y, stationary)
            t_done = T
            stats = {side: accs[side].stats() for side in ("G", "H")}
            feats = {side: gb.batch_features(stats[side][0], stats[side][1], index, nmax)
                     for side in ("G", "H")}
            test = gb.permutation_test(feats["G"], feats["H"], nmax, seed=spec["seed"])
            gaps = {str(N): float(np.linalg.norm(
                np.sqrt(np.sort(stats["G"][0][N][0].mean(0))) - np.sqrt(np.sort(stats["H"][0][N][0].mean(0)))))
                for N in range(1, nmax + 1)}
            result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"],
                      "brightness": spec.get("brightness"), "kappa": spec.get("kappa"),
                      "delay": spec.get("delay"), "n": n, "nmax": nmax, "T": T,
                      "test": test, "sampled_gap": gaps, "seconds": time.time() - t1}
            json.dump(result, open(f"{stem}-T{T}.json", "w"))
            summary["done"].append(T)
            if spec.get("checkpoint", True):
                ck = {"t_done": np.array(t_done)}
                for side in ("G", "H"):
                    ck.update({f"{side}:{k}": v for k, v in accs[side].state_dict().items()})
                    ck.update({f"{side}:{k}": v for k, v in samplers[side].state_dict().items()})
                np.savez(f"{stem}-ckpt.npz", **ck)
            vol.commit()
    if spec["model"] in ("step-lin", "init-lin") and not os.path.exists(f"{stem}-exact.json"):
        eg = gb.exact_linear_profiles(spec["model"], adj["G"], spec["seed"], spec["exact_nmax"])
        eh = gb.exact_linear_profiles(spec["model"], adj["H"], spec["seed"], spec["exact_nmax"])
        json.dump({str(N): {"gap": bp.gap(eg[N], eh[N]),
                            "max_rel_diff": float(np.max(np.abs(eg[N] - eh[N])) / max(eg[N].max(), eh[N].max()))}
                   for N in eg}, open(f"{stem}-exact.json", "w"))
        vol.commit()
    if spec.get("hotelling") and not os.path.exists(f"{stem}-hotelling.json"):
        t1 = time.time()
        hot = gb.sampled_hotelling(spec["model"], adj["G"], adj["H"], spec["seed"],
                                   spec["hotelling"], nmax, device, **kw)
        hot["seconds"] = time.time() - t1
        json.dump(hot, open(f"{stem}-hotelling.json", "w"))
        vol.commit()
    return summary


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
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "runs": {},
              "min_p": {}}
    for T in spec["budgets"]:
        hits = 0
        result["min_p"][str(T)] = []
        for r in range(spec["runs"]):
            rng = np.random.default_rng(1000 * spec["seed"] + r)
            sg, cg = gb.sampled_statistics(spec["model"], adj, spec["seed"], T, nmax, device,
                                           noise_seed=100 + 2 * r, **kw)
            sg2, cg2 = gb.sampled_statistics(spec["model"], bp.relabel(adj, rng.permutation(n)),
                                              spec["seed"], T, nmax, device, noise_seed=101 + 2 * r, **kw)
            fg = gb.batch_features(sg, cg, index, nmax)
            fg2 = gb.batch_features(sg2, cg2, index, nmax)
            test = gb.permutation_test(fg, fg2, nmax, seed=r)
            hits += test["separated"]
            result["min_p"][str(T)].append([min(test["pvalues"].values()), test["J"]])
        result["runs"][str(T)] = hits / spec["runs"]
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=3600, max_containers=30, retries=1)
def ideal_job(spec):
    """The optimal diagonal test's sample count for a linear network: the
    exact sorted profiles by Wick's theorem over the variance of every set
    estimated from `samples` draws, cumulated over the orders, like the
    counts of coherent light in the full run."""
    import numpy as np
    import brecphot as bp
    import gnn_bench as gb
    device = "cuda"
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    stem = f"{folder}/{spec['model']}{spec.get('suffix', '')}"
    path = f"{stem}-ideal.json"
    if os.path.exists(path):
        return json.load(open(path))
    adj = {"G": bp.adjacency(spec["G"]), "H": bp.adjacency(spec["H"])}
    nmax = spec["nmax"]
    exact, var = {}, {}
    for i, side in enumerate(("G", "H")):
        exact[side] = gb.exact_linear_profiles(spec["model"], adj[side], spec["seed"], nmax)
        stats, counts = gb.sampled_statistics(spec["model"], adj[side], spec["seed"], spec["samples"],
                                              nmax, device, noise_seed=50 + i)
        # the variance of every set, in the order of the exact sorted profile
        var[side] = {}
        for N in range(1, nmax + 1):
            order = np.argsort(stats[N][0].mean(axis=0))
            var[side][N] = (stats[N][1].mean(axis=0) + stats[N][0].var(axis=0) * counts.mean())[order]
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "nmax": nmax,
              "n": adj["G"].shape[0], "samples": spec["samples"], "ideal_samples": {}, "exact": {}}
    for N in range(1, nmax + 1):
        scale = max(exact["G"][N].max(), exact["H"][N].max())
        result["exact"][str(N)] = {
            "gap": bp.gap(exact["G"][N], exact["H"][N]),
            "max_rel_diff": float(np.max(np.abs(exact["G"][N] - exact["H"][N])) / scale)}
        d2 = 0.0
        for M in range(1, N + 1):
            diff = exact["G"][M] - exact["H"][M]
            noise = var["G"][M] + var["H"][M]
            mask = noise > 0
            d2 += float(np.sum(diff[mask] ** 2 / noise[mask]))
        result["ideal_samples"][str(N)] = 4 * np.log(20) / d2 if d2 > 0 else float("inf")
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


SYNTHETIC_BUDGETS = (1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
                     10000000000, 100000000000, 1000000000000, 10000000000000)


@app.function(image=image, cpu=8, memory=32768, volumes={"/vol": vol},
              timeout=3600, max_containers=50, retries=1)
def synthetic_job(spec):
    """The protocol on synthetic batches drawn from exact moments, as in the
    draft's BREC run: every set's batch estimate is Gaussian with the exact
    mean and the estimated variance over T / 16 samples, sets independent.
    Coherent light takes its exact rates and phase-Monte-Carlo variances;
    a linear network its Wick moments and sampled variances.  Records, at
    every budget and over several draws, the decision with the original
    features and with the studentised profile added."""
    import numpy as np
    import torch
    import brecphot as bp
    import coherent_gpu as cg
    import gnn_bench as gb
    device = "cpu"  # exact moments and a short Monte Carlo: no GPU, so no queue
    torch.set_num_threads(8)
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    stem = f"{folder}/{spec['model']}{spec.get('suffix', '')}"
    path = f"{stem}-synthetic.json"
    if os.path.exists(path):
        return json.load(open(path))
    adj = {"G": bp.adjacency(spec["G"]), "H": bp.adjacency(spec["H"])}
    n = adj["G"].shape[0]
    nmax = spec["nmax"]
    mean, var = {}, {}
    for i, side in enumerate(("G", "H")):
        if spec["model"] == "coherent":
            w = spec["brightness"]
            W = cg.to_torch(bp.path_matrix(bp.network(adj[side], bp.cell(spec["seed"]), 0.9)), device)
            tensors = {N: cg.rate_tensors_torch(W, N, ("coherent",))["coherent"] for N in range(1, nmax + 1)}
            moments = cg.intensity_moments(W, nmax, spec["samples"], device, seed=50 + i)
            mean[side], var[side] = {}, {}
            for N in range(1, nmax + 1):
                idx = torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T, device=device)
                mean[side][N] = (tensors[N][tuple(idx)] * w ** N).cpu().numpy()
                var[side][N], _ = cg.coherent_variance(moments, N, w)
            del W, tensors
        else:
            C = gb.linear_covariance(spec["model"], adj[side], spec["seed"])
            stats, counts = gb.sampled_statistics(spec["model"], adj[side], spec["seed"], spec["samples"],
                                                  nmax, device, noise_seed=50 + i)
            mean[side], var[side] = {}, {}
            for N in range(1, nmax + 1):
                sets = np.array(list(gb.combinations(range(n), N)))
                mean[side][N] = gb.hafnians_of_doubled_sets(C, sets)
                var[side][N] = stats[N][1].mean(axis=0) + stats[N][0].var(axis=0) * counts.mean()
    index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T) for N in range(1, nmax + 1)}
    rng = np.random.default_rng(spec["seed"])
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "n": n,
              "nmax": nmax, "runs": spec["runs"], "budgets": {}}
    B = gb.BATCHES
    for T in spec["budgets"]:
        per_batch = T / B
        hits_old, hits_new, pmin = 0, 0, []
        for r in range(spec["runs"]):
            feats = {}
            for side in ("G", "H"):
                stats = {}
                for N in range(1, nmax + 1):
                    noise = rng.normal(size=(B, mean[side][N].shape[0])) * np.sqrt(var[side][N] / per_batch)
                    stats[N] = (np.maximum(mean[side][N][None, :] + noise, 0), np.tile(var[side][N], (B, 1)))
                feats[side] = gb.batch_features(stats, np.full(B, per_batch), index, nmax)
            test = gb.permutation_test(feats["G"], feats["H"], nmax, seed=r)
            hits_old += test["separated_old"]
            hits_new += test["separated"]
            pmin.append(min(test["pvalues"].values()))
        result["budgets"][str(T)] = {"old": hits_old / spec["runs"], "new": hits_new / spec["runs"],
                                     "min_p": pmin}
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.function(image=image, cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=3600, max_containers=50, retries=1)
def retest_job(spec):
    """Rerun the protocol at the checkpointed budget from the saved
    accumulators, with the studentised profile among the features; no
    sample is drawn."""
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    stem = f"{folder}/{spec['model']}{spec.get('suffix', '')}"
    try:
        vol.reload()
    except RuntimeError:
        pass
    if not os.path.exists(f"{stem}-ckpt.npz"):
        return {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "skipped": "no checkpoint"}
    with np.load(f"{stem}-ckpt.npz") as z:
        ck = {k: z[k] for k in z.files}
    T = int(ck["t_done"])
    path = f"{stem}-T{T}-retest.json"
    if os.path.exists(path):
        return json.load(open(path))
    n = bp.adjacency(spec["G"]).shape[0]
    nmax = spec["nmax"]
    index = {N: torch.as_tensor(np.array(list(gb.combinations(range(n), N))).T) for N in range(1, nmax + 1)}
    t0 = time.time()
    feats = {}
    for side in ("G", "H"):
        acc = gb.Accumulator(n, nmax, device="cpu")
        acc.load_state_dict({k[2:]: v for k, v in ck.items() if k.startswith(f"{side}:")}, "cpu")
        stats, counts = acc.stats()
        feats[side] = gb.batch_features(stats, counts, index, nmax)
    test = gb.permutation_test(feats["G"], feats["H"], nmax, seed=spec["seed"])
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "n": n, "nmax": nmax,
              "T": T, "test": test, "seconds": time.time() - t0}
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


@app.function(image=image, cpu=4, memory=32768, volumes={"/vol": vol},
              timeout=3600, max_containers=50, retries=1)
def gnn_hotelling_job(spec):
    """BREC's test on the exact profiles of a linear network, as `full_test`
    does for the photonic sources: the profiles of orders 1..N, each divided
    by the N-th power of the mean order-1 rate of G (the unit that w^N is
    for photons), concatenated, projected by the pair's random matrix, and
    the projected gap squared over the regularisation, verbatim and scaled
    to the pilot's rounding noise."""
    import numpy as np
    import brecphot as bp
    import gnn_bench as gb
    folder = f"/vol/{spec.get('experiment', 'gnn-compare')}/{spec['pair']}/s{spec['seed']}"
    stem = f"{folder}/{spec['model']}{spec.get('suffix', '')}"
    path = f"{stem}-hotelling-exact.json"
    if os.path.exists(path):
        return json.load(open(path))
    adj = {"G": bp.adjacency(spec["G"]), "H": bp.adjacency(spec["H"])}
    nmax = spec["nmax"]
    exact = {side: gb.exact_linear_profiles(spec["model"], adj[side], spec["seed"], nmax)
             for side in ("G", "H")}
    unit = float(exact["G"][1].mean())
    rng = np.random.default_rng(sum(map(ord, spec["pair"])))
    result = {"pair": spec["pair"], "seed": spec["seed"], "model": spec["model"], "nmax": nmax,
              "n": adj["G"].shape[0], "unit": unit, "orders": {}}
    for N in range(1, nmax + 1):
        dim = sum(exact["G"][M].shape[0] for M in range(1, N + 1))
        projection = rng.normal(size=(bp.OUTPUT_DIM, dim)) / np.sqrt(dim)
        d_mean = bp.features({M: exact["G"][M] for M in range(1, N + 1)}, unit, projection) \
            - bp.features({M: exact["H"][M] for M in range(1, N + 1)}, unit, projection)
        norm2 = float(d_mean @ d_mean)
        scale = max(exact["G"][N].max(), exact["H"][N].max())
        result["orders"][str(N)] = {
            "T2_verbatim": norm2 / bp.EPSILON_MATRIX,
            "T2_scaled": norm2 / spec["noise"][str(N)],
            "separated_verbatim": bool(norm2 / bp.EPSILON_MATRIX > bp.THRESHOLD),
            "separated_scaled": bool(norm2 / spec["noise"][str(N)] > bp.THRESHOLD),
            "projected_gap": float(np.sqrt(norm2)),
            "max_rel_diff": float(np.max(np.abs(exact["G"][N] - exact["H"][N])) / scale)}
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result
