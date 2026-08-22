"""
Modal fan-out for the expensive active-model cells.

The two-herald ensemble is an average over injection pairs, so it
shards perfectly: each worker receives the graph, the frozen step-matrix
blocks (precomputed locally, so workers only need numpy) and a chunk of
injection pairs, runs the branching engine on its chunk and returns the
unnormalised bins; the driver sums the chunks and divides once.

``python modal_app.py --pair cfi-83 --ticks 6`` runs one dataset pair
remotely and stores the two ensembles under ``results/bins/``.
"""

import argparse
import json
import os
import pickle
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = modal.App("kwl-ladder")
image = (modal.Image.debian_slim()
         .pip_install("numpy")
         .add_local_python_source("exact"))


@app.function(image=image, cpu=2.0, memory=8192, timeout=3600,
              max_containers=100, retries=2)
def run_chunk(payload):
    import numpy as np
    from exact import FrozenLayout, Machine, aggregate
    spec = pickle.loads(payload)
    layout = FrozenLayout(spec["n_drives"], spec["blocks"],
                          spec.get("reference"))
    machine = Machine(
        spec["graph"], layout, tap=spec["tap"],
        kicked_tap=spec["kicked_tap"], broadcast=spec["broadcast"])
    bins = aggregate(machine, spec["ticks"], engine=spec["engine"],
                     ensemble=spec["pairs"])
    scaled = {key: p * len(spec["pairs"]) for key, p in bins.items()}
    return pickle.dumps(scaled)


def shard(graph, model_name, ticks, pairs, chunk_size, engine):
    from exact import FrozenLayout, machine_for
    from models import MODELS
    model = MODELS[model_name]
    machine = machine_for(model, graph, frozen=True)
    layout = machine.layout
    spec = {
        "graph": graph, "n_drives": layout.n_drives,
        "blocks": layout.cache,
        "reference": getattr(layout, "reference", None),
        "tap": machine.tap, "kicked_tap": machine.kicked_tap,
        "broadcast": machine.broadcast, "ticks": ticks,
        "engine": engine}
    chunks = [pairs[i:i + chunk_size]
              for i in range(0, len(pairs), chunk_size)]
    return [pickle.dumps({**spec, "pairs": chunk}) for chunk in chunks]


def remote_aggregate(graph, model_name, ticks, pairs, chunk_size=4,
                     engine="branch"):
    payloads = shard(graph, model_name, ticks, pairs, chunk_size,
                     engine)
    total = {}
    with app.run():
        for result in run_chunk.map(payloads):
            for key, p in pickle.loads(result).items():
                total[key] = total.get(key, 0.0) + p
    return {key: p / len(pairs) for key, p in total.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--model", default="active")
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--ensemble", default="edges",
                        choices=("edges", "all"))
    args = parser.parse_args()

    from run import RESULTS, edges_of, graphs_of, load_pairs
    record, = [r for r in load_pairs() if r["id"] == args.pair]
    os.makedirs(os.path.join(RESULTS, "bins"), exist_ok=True)
    for side, graph in zip("lr", graphs_of(record)):
        from itertools import combinations
        pairs = (edges_of(graph) if args.ensemble == "edges"
                 else list(combinations(range(len(graph)), 2)))
        bins = remote_aggregate(
            graph, args.model, args.ticks, pairs, args.chunk)
        path = os.path.join(
            RESULTS, "bins",
            f"{args.pair}-{side}-{args.model}-T{args.ticks}"
            f"-{args.ensemble}.pkl")
        with open(path, "wb") as handle:
            pickle.dump(bins, handle)
        print(f"{path}: {len(bins)} bins, "
              f"mass {sum(bins.values()):.9f}", flush=True)


if __name__ == "__main__":
    main()
