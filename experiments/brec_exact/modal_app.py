"""Modal fan-out for the exact-profile experiments.

    modal deploy modal_app.py
    python launch.py column   # 32 relabelled copies at gamma = 0.9
    python launch.py sweep    # one copy per graph over a grid of losses

Every job writes its result under /vol/<experiment>/ on the volume
``photonic-brec-rates`` and returns a one-line summary; the aggregator
``column_test`` reads the copies of one pair and seed and runs BREC's test.
"""

import json
import os
import time

import modal

app = modal.App("brec-photonic")
vol = modal.Volume.from_name("photonic-brec-rates")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .env({"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
         .add_local_python_source("brecphot"))

W = 0.05
GAMMA = 0.9


def _save(path, **arrays):
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768,
              timeout=4 * 3600, max_containers=100, retries=1)
def profile_job(spec):
    """Profiles of relabelled copies: spec has pair, seed, gamma, nmax and a
    list of copies (tag, graph6, permutation or None)."""
    import numpy as np
    import brecphot as bp
    cells = bp.cell(spec["seed"])
    folder = f"/vol/{spec['experiment']}/{spec['pair']}/s{spec['seed']}"
    done, t0 = [], time.time()
    for tag, graph6, perm in spec["copies"]:
        path = f"{folder}/{tag}.npz"
        if os.path.exists(path):
            done.append(tag)
            continue
        adj = bp.adjacency(graph6)
        if perm is not None:
            adj = bp.relabel(adj, np.array(perm))
        t1 = time.time()
        prof, slots = bp.network_profiles(adj, cells, spec["gamma"], W,
                                          spec["nmax"])
        _save(path, slots=slots, seconds=time.time() - t1,
              **{f"F{N}": prof[N] for N in prof})
        done.append(tag)
    vol.commit()
    return {"pair": spec["pair"], "seed": spec["seed"],
            "copies": len(done), "seconds": time.time() - t0}


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=65536,
              timeout=6 * 3600, max_containers=100, retries=1)
def sweep_job(spec):
    """Profiles of G and H at one loss, and the gaps between them."""
    import numpy as np
    import brecphot as bp
    cells = bp.cell(spec["seed"])
    gamma = spec["gamma"]
    folder = f"/vol/loss-sweep/{spec['pair']}/s{spec['seed']}"
    path = f"{folder}/g{gamma}.json"
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    prof, slots = {}, {}
    for side, graph6 in (("G", spec["G"]), ("H", spec["H"])):
        adj = bp.adjacency(graph6)
        prof[side], slots[side] = bp.network_profiles(
            adj, cells, gamma, W, spec["nmax"])
    n = bp.adjacency(spec["G"]).shape[0]
    result = {"pair": spec["pair"], "seed": spec["seed"], "gamma": gamma,
              "n": n, "slots": slots, "seconds": time.time() - t0,
              "mu": W * n / (1 - gamma), "orders": {}}
    for N in prof["G"]:
        fg, fh = prof["G"][N], prof["H"][N]
        entry = {}
        for i, source in enumerate(bp.SOURCES):
            scale = max(fg[i].max(), fh[i].max())
            entry[source] = {
                "gap": bp.gap(fg[i], fh[i]),
                "max_rel_diff": float(np.max(np.abs(fg[i] - fh[i])) / scale),
                "ideal_samples": bp.ideal_samples(fg[i], fh[i]),
                "ideal_samples_cumulative": bp.ideal_samples(
                    np.concatenate([prof["G"][M][i] for M in range(1, N + 1)]),
                    np.concatenate([prof["H"][M][i] for M in range(1, N + 1)])),
            }
        result["orders"][str(N)] = entry
    _save(f"{folder}/g{gamma}.npz",
          **{f"F{s}{N}": prof[s][N] for s in prof for N in prof[s]})
    os.makedirs(folder, exist_ok=True)
    json.dump(result, open(path, "w"))
    vol.commit()
    return result


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=32768,
              timeout=3600, max_containers=50)
def column_test(spec):
    """BREC's test on the profiles of the relabelled copies of one pair and
    seed, cumulatively over the orders."""
    import numpy as np
    import brecphot as bp
    folder = f"/vol/column-pilot/{spec['pair']}/s{spec['seed']}"
    try:
        vol.reload()
    except RuntimeError:
        pass  # a warm container with a stale handle still sees its mount
    copies = {}
    for tag in spec["tags"]:
        with np.load(f"{folder}/{tag}.npz") as z:
            copies[tag] = {int(k[1:]): np.array(z[k])
                           for k in z.files if k.startswith("F")}
    nmax = max(copies[spec["tags"][0]])
    rng = np.random.default_rng(sum(map(ord, spec["pair"])))
    result = {"pair": spec["pair"], "seed": spec["seed"], "orders": {}}
    g_tags = [t for t in spec["tags"] if t.startswith("G")]
    h_tags = [t for t in spec["tags"] if t.startswith("H")]
    g_pair, g_rel = g_tags[:len(h_tags)], g_tags[len(h_tags):]
    for N in range(1, nmax + 1):
        dim = sum(copies[g_tags[0]][M].shape[1] for M in range(1, N + 1))
        projection = rng.normal(size=(bp.OUTPUT_DIM, dim)) / np.sqrt(dim)
        entry = {}
        for i, source in enumerate(bp.SOURCES):
            def feats(tags):
                return np.array([bp.features(
                    {M: copies[t][M][i] for M in range(1, N + 1)}, W,
                    projection) for t in tags]).T
            X, Y, Z = feats(g_pair), feats(h_tags), feats(g_rel)
            noise = float(np.mean(np.diag(np.cov(X - Z))))
            out = {"noise_variance": noise}
            for label, eps in (("verbatim", bp.EPSILON_MATRIX),
                               ("scaled", bp.EPSILON_MATRIX * noise)):
                t2_pair = bp.hotelling(X, Y, eps)
                t2_rel = bp.hotelling(X, Z, eps)
                separated, reliable = bp.brec_decision(t2_pair, t2_rel)
                out[label] = {"T2_pair": t2_pair, "T2_reliability": t2_rel,
                              "separated": separated, "reliable": reliable}
            # the exact test of the draft on the unrelabelled profiles
            fg, fh = copies["G0"][N][i], copies["H0"][N][i]
            scale = max(fg.max(), fh.max())
            spread = max(np.max(np.abs(copies[t][N][i] - fg)) for t in g_tags)
            out["exact"] = {
                "gap": bp.gap(fg, fh),
                "max_rel_diff": float(np.max(np.abs(fg - fh)) / scale),
                "relabel_floor": float(spread / scale),
                "separated_1e-12": bool(np.max(np.abs(fg - fh)) / scale > 1e-12),
                "ideal_samples": bp.ideal_samples(fg, fh),
            }
            entry[source] = out
        result["orders"][str(N)] = entry
    json.dump(result, open(f"{folder}/test.json", "w"))
    vol.commit()
    return result


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def status(experiment):
    """How many result files each pair and seed has."""
    vol.reload()
    import brecphot as bp
    root = f"/vol/{experiment}"
    out = {"brecphot_version": getattr(bp, "VERSION", 1)}
    if not os.path.isdir(root):
        return out
    for pair in sorted(os.listdir(root)):
        for seed in sorted(os.listdir(f"{root}/{pair}")):
            files = os.listdir(f"{root}/{pair}/{seed}")
            out[f"{pair}/{seed}"] = {
                "npz": sum(f.endswith(".npz") for f in files),
                "json": sorted(f for f in files if f.endswith(".json"))}
    return out


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def read_json(paths):
    """The contents of small result files, for the driver."""
    vol.reload()
    out = {}
    for p in paths:
        try:
            out[p] = json.load(open(f"/vol/{p}"))
        except Exception as e:  # noqa: BLE001
            out[p] = f"ERR {e!r}"
    return out
