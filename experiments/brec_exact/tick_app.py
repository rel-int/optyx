"""Modal jobs for the forward-pass test on the pilot subset.

    modal run tick_app.py::validate      # checks on small graphs, timings
    modal deploy tick_app.py
    python launch_tick.py run [--pairs ...] [--seeds ...] [--sources ...]
"""

import json
import os
import time

import modal

app = modal.App("brec-tick")
vol = modal.Volume.from_name("photonic-brec-rates")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "torch")
         .add_local_python_source("brecphot", "single_tick"))

BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"
EXPERIMENT = "tick-pilot"
GAMMA = 0.9


def _graphs():
    import numpy as np
    cycle = np.zeros((6, 6), dtype=np.int8)
    for i in range(6):
        cycle[i, (i + 1) % 6] = cycle[(i + 1) % 6, i] = 1
    two = np.zeros((6, 6), dtype=np.int8)
    for block in ([0, 1, 2], [3, 4, 5]):
        for i in block:
            for j in block:
                two[i, j] = i != j
    return cycle, two


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600)
def validate(big_graph6=None):
    """Self-checks, invariance of the embedding under relabelling, the false
    positive rate on G against relabelled G, the power on C6 against 2C3, the
    exact anchors, and timings at BREC sizes."""
    import numpy as np
    import torch
    import brecphot as bp
    import single_tick as st
    device = "cuda"
    out = {"self_check": st.self_check()}
    g, h = _graphs()
    cells = bp.cell(100)
    matrices = st.copies_of({"G": g, "H": h}, cells, GAMMA, "smoke", copies=4)
    # the exact mean intensity is the same on every copy, the sampled one too
    out["mean intensity G0, G1, G5"] = [st.mean_intensity(matrices[t], None) for t in ("G0", "G1", "G5")]
    for source in st.SOURCES:
        e0, _ = st.copy_embeddings(matrices["G0"], source, 1, [100000], device)
        e1, _ = st.copy_embeddings(matrices["G1"], source, 2, [100000], device)
        out[f"{source}: max rel diff of G0 vs relabelled G1 at 1e5 draws"] = float(
            np.max(np.abs(e0[100000] - e1[100000]) / np.abs(e0[100000])))
    # the sampled coherent embedding against the exact one on the walk
    for light in ("coherent", "thermal"):
        e, _ = st.copy_embeddings(matrices["G0"], f"{light}-inf", 3, [400000], device)
        names = st.feature_names()
        for N, k in ((3, 1), (3, 2)):
            exact = st.exact_readout(matrices["G0"], light, N, k, device)
            out[f"{light} e{N}(I^{k}): sampled / exact - 1"] = float(e[400000][names.index(f"e{N}(I^{k})")] / exact - 1)
    # false positives and power, 32 copies, at small budgets
    matrices = st.copies_of({"G": g, "H": h}, cells, GAMMA, "smoke")
    for source in ("coherent-w100", "thermal-inf"):
        t0 = time.time()
        res = st.forward_pass_test(matrices, source, 7, [1000, 10000], device)
        out[f"{source}: C6 vs 2C3 decisions"] = {R: {k: d[k]["separated"] for k in ("verbatim", "scaled")}
                                                for R, d in res["decisions"].items()}
        out[f"{source}: reliability"] = {R: d["scaled"]["T2_reliability"] for R, d in res["decisions"].items()}
        out[f"{source}: seconds"] = time.time() - t0
    out["anchors C6 vs 2C3"] = st.exact_anchors(matrices, device=device)
    if big_graph6:
        adj = bp.adjacency(big_graph6)
        W = bp.path_matrix(bp.network(adj, cells, GAMMA), tol=st.PATH_TOL)
        out["big: n, slots"] = list(W.shape)
        for source in ("coherent-w100", "thermal-inf"):
            torch.cuda.synchronize()
            t0 = time.time()
            st.copy_embeddings(W, source, 1, [1000000], device)
            torch.cuda.synchronize()
            out[f"big {source}: seconds per copy at 1e6 draws"] = time.time() - t0
        t0 = time.time()
        out["big thermal e4(I^3) exact"] = st.exact_readout(W, "thermal", 4, 3, device)
        out["big thermal exact seconds"] = time.time() - t0
    return json.dumps(out, default=str)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=4 * 3600, max_containers=40, retries=1)
def tick_job(spec):
    """One pair and cell: the path matrices of the 96 copies, then the test
    for every source and the exact anchors, each cached in its own file."""
    import numpy as np
    import brecphot as bp
    import single_tick as st
    device = "cuda"
    folder = f"/vol/{spec.get('experiment', EXPERIMENT)}/{spec['pair']}/s{spec['seed']}"
    os.makedirs(folder, exist_ok=True)
    todo = [s for s in spec["sources"] if not os.path.exists(f"{folder}/{s}.json")]
    anchors = spec.get("anchors", True) and not os.path.exists(f"{folder}/anchors.json")
    summary = {"pair": spec["pair"], "seed": spec["seed"], "done": []}
    if not todo and not anchors:
        return summary
    adj = {"G": bp.adjacency(spec["G"]), "H": bp.adjacency(spec["H"])}
    t0 = time.time()
    matrices = st.copies_of(adj, bp.cell(spec["seed"]), spec.get("gamma", GAMMA), spec["pair"])
    summary["matrix_seconds"] = time.time() - t0
    summary["n, slots"] = list(matrices["G0"].shape)
    for source in todo:
        t1 = time.time()
        result = st.forward_pass_test(matrices, source, spec["seed"], spec["budgets"], device)
        result.update({"pair": spec["pair"], "seed": spec["seed"], "source": source,
                       "n": int(adj["G"].shape[0]), "slots": int(matrices["G0"].shape[1]),
                       "features": st.feature_names(), "version": st.VERSION,
                       "seconds": time.time() - t1})
        json.dump(result, open(f"{folder}/{source}.json", "w"))
        vol.commit()
        summary["done"].append(source)
    if anchors:
        t1 = time.time()
        result = st.exact_anchors(matrices, device=device)
        result["seconds"] = time.time() - t1
        json.dump(result, open(f"{folder}/anchors.json", "w"))
        vol.commit()
        summary["done"].append("anchors")
    summary["seconds"] = time.time() - t0
    return summary


@app.local_entrypoint()
def validate_entry():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import brecphot as bp
    data = bp.load_brec(BREC_RAW)
    print(validate.remote(data["4-vertex_condition"][0][0]))
