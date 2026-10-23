"""Level 1 of the sampler validation on Modal: the devices of the subset run
tick by tick, against the exact rates, variances and autocovariances.

    modal run validation_app.py                       # the subset, all sources
    modal run validation_app.py --pairs Regular_47 --ticks 256

Every job is one graph, one source and one brightness; its record is written
to the volume under sampler-validation/ and returned, and the entrypoint
gathers the records into results/raw_validation.json.gz.
"""
import gzip
import json
import os

import modal

app = modal.App("brec-sampler-validation")
vol = modal.Volume.from_name("photonic-brec-rates")
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "torch")
         .add_local_python_source("brecphot", "photon_noise", "recurrent")
         .add_local_dir(f"{HERE}/noise_tables", remote_path="/root/noise_tables"))

BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"
EXPERIMENT = "sampler-validation"
PAIRS = {"Regular_47": 4, "Regular_10": 4, "Basic_0": 4, "Extension_11": 4,
         "Strongly_regular_0": 3, "CFI_5": 3}
RUNS = {"coherent": (0.05, 1.0, 100.0), "thermal": (0.05, 1.0, 100.0),
        "photons p=0": (0.05, 1.0)}
SEED, GAMMA = 100, 0.9
CHAINS, TICKS, BATCHES, BLOCK = 65536, 2048, 64, 64
LAGS_SAMPLED, LAGS_EXACT, TRACKED = 32, 120, 6
EXACT_TAU_ORDER = 3  # the two-time moments of order 4 need the tables of 8 distinct detectors


def tracked_sets(F, count=TRACKED):
    """`count` sets spread evenly over the sorted exact rates."""
    import numpy as np
    order = np.argsort(F, kind="stable")
    return sorted({int(order[i]) for i in np.linspace(0, len(F) - 1, count).round().astype(int)})


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=7200,
              volumes={"/vol": vol}, max_containers=10, retries=1)
def level1_job(spec):
    import time
    import numpy as np
    import brecphot as bp
    import recurrent as rc
    start = time.time()
    adj = bp.adjacency(spec["graph6"])
    n = adj.shape[0]
    device = rc.Device(bp.network(adj, bp.cell(spec["seed"]), spec["gamma"]), "cuda")
    source, w, nmax = spec["source"], spec["w"], spec["nmax"]
    exact = rc.rates_variances(device.W, nmax, source, w, "cuda")
    tracked = {N: tracked_sets(F) for N, (F, _) in exact.items()}
    covariances = {}
    for N in range(1, min(nmax, EXACT_TAU_ORDER) + 1):
        sets = [tuple(int(a) for a in rc.set_index(n, N)[:, i]) for i in tracked[N]]
        covariances[N] = rc.autocovariances(device.W, n, sets, spec["lags_exact"], source, w, "cuda")
    exact_time = time.time() - start
    moments = rc.Moments(n, {N: F for N, (F, _) in exact.items()}, spec["chains"],
                         spec["batches"], spec["block"], spec["lags_sampled"], tracked, "cuda")
    for y in device.counts(source, w, spec["chains"], spec["ticks"], seed=spec["noise_seed"]):
        moments.add(y)
    sums = moments.result()
    record = {k: v for k, v in spec.items() if k != "graph6"}
    record.update(n=n, slots=device.W.shape[1], exact_seconds=exact_time,
                  seconds=time.time() - start, orders={})
    for N, (F, var) in exact.items():
        s = rc.summary(F, var, sums[N], spec["block"])
        s["tracked"] = tracked[N]
        if N in covariances:
            s.update(autocov=covariances[N], tau_exact=rc.tau_exact(covariances[N], var[tracked[N]]))
        record["orders"][str(N)] = {k: np.asarray(v).tolist() if not np.isscalar(v) else v
                                    for k, v in s.items()}
    name = f"{spec['pair']}_{spec['side']}_{source.replace(' ', '').replace('=', '')}_{w}"
    os.makedirs(f"/vol/{EXPERIMENT}", exist_ok=True)
    with open(f"/vol/{EXPERIMENT}/{name}.json", "w") as f:
        json.dump(record, f)
    vol.commit()
    return record


def specs(pairs, sources, ticks, chains):
    import brecphot as bp
    data = bp.load_brec(BREC_RAW)
    out = []
    for pair in pairs:
        category, i = pair.rsplit("_", 1)
        for side, graph6 in zip("GH", data[category][int(i)]):
            for source in sources:
                for w in RUNS[source]:
                    out.append({"pair": pair, "side": side, "graph6": graph6, "source": source,
                                "w": w, "nmax": PAIRS[pair], "seed": SEED, "gamma": GAMMA,
                                "chains": chains, "ticks": ticks, "batches": BATCHES,
                                "block": BLOCK, "lags_sampled": LAGS_SAMPLED,
                                "lags_exact": LAGS_EXACT, "noise_seed": len(out)})
    return out


@app.local_entrypoint()
def main(pairs: str = ",".join(PAIRS), sources: str = ",".join(RUNS),
         ticks: int = TICKS, chains: int = CHAINS, out: str = "results/raw_validation.json.gz"):
    import sys
    sys.path.insert(0, HERE)
    todo = specs(pairs.split(","), sources.split(","), ticks, chains)
    path = os.path.join(HERE, out)
    records = {}
    if os.path.exists(path):
        records = json.load(gzip.open(path, "rt"))
    for record in level1_job.map(todo, return_exceptions=True, order_outputs=False):
        if isinstance(record, Exception):
            print("failed:", record)
            continue
        key = f"{record['pair']}|{record['side']}|{record['source']}|{record['w']}"
        records[key] = record
        print(key, f"{record['seconds']:.0f}s")
        with gzip.open(path, "wt") as f:
            json.dump(records, f)
