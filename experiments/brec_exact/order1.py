"""Order 1 on full BREC under the two conditions of exact equilibrium
propagation: a reciprocal chip and a dephased memory.

The order-1 rate of every source with no phase reference is the stationary
intensity at the detectors, diag(C K C^dag + D D^dag) per unit brightness,
with K the stationary one-particle density matrix of the loop modes:

    K = Deph(A K A^dag + B B^dag),

Deph zeroing the coherences of the dephased modes with every other mode.
Without dephasing K = sum_k A^k B B^dag A^k^dag, which gives diag(W W^dag),
the order-1 profile of the reach run.

Variants, each with the cells of seeds 100-103:
    haar            the reach run's Haar U(3) cell
    reciprocal      a symmetric unitary cell, U = V V^T with V Haar, so that
                    every chip and the arc wiring are reciprocal
    haar+memory     the Haar cell, the memory mode dephased every tick
    reciprocal+memory
    reciprocal+loop every fed-back mode dephased, ports included

    python order1.py [--variants a,b] [--out results/raw_order1.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402

SEEDS = (100, 101, 102, 103)
GAMMA = 0.9
THRESHOLD = 1e-12
BREC_RAW = "/home/user/graphpku/brec/customize/Data/raw"   # gpu_app.BREC_RAW, without importing modal
VARIANTS = ("haar", "reciprocal", "haar+memory", "reciprocal+memory", "reciprocal+loop")


def reciprocal_cell(seed):
    """Symmetric unitaries U(3) and U(2): U = V V^T, V Haar (the circular
    orthogonal ensemble). The port block acts on the uniform Fourier mode,
    a real vector, so the chip's unitary is symmetric too."""
    rng = np.random.default_rng(seed)
    out = []
    for d in (3, 2):
        v = bp.haar_unitary(rng, d)
        out.append(v @ v.T)
    return out


def dephased_modes(adj, kind):
    """Indices of the loop modes dephased every tick."""
    arcs = int(adj.sum())
    n = adj.shape[0]
    if kind == "memory":
        return np.arange(arcs, arcs + n)
    return np.arange(arcs + n)


def stationary(A, B, dephased=None, tol=1e-17, max_ticks=5000):
    """K = Deph(A K A^dag + B B^dag); by doubling without dephasing."""
    Q = B @ B.conj().T
    if dephased is None:
        K, P = Q.copy(), A.copy()
        for _ in range(64):
            step = P @ K @ P.conj().T
            K = K + step
            if np.abs(step).max() < tol * np.abs(K).max():
                return K
            P = P @ P
        raise RuntimeError("doubling did not converge")
    mask = np.ones(A.shape, dtype=bool)
    mask[dephased, :] = False
    mask[:, dephased] = False
    mask[dephased, dephased] = True  # the populations of the dephased modes
    K = np.zeros_like(Q)
    for _ in range(max_ticks):
        new = A @ K @ A.conj().T + Q
        new *= mask
        if np.abs(new - K).max() < 1e-2 * (1 - GAMMA) * THRESHOLD * np.abs(new).max():
            return new
        K = new
    raise RuntimeError("dephased loop did not converge")


def order1(adj, variant, seed):
    """The sorted order-1 profile of one graph."""
    cells = reciprocal_cell(seed) if variant.startswith("reciprocal") else bp.cell(seed)
    A, B, C, D = bp.network(adj, cells, GAMMA)
    kind = variant.split("+")[1] if "+" in variant else None
    K = stationary(A, B, dephased_modes(adj, kind) if kind else None)
    rates = np.real(np.einsum("ij,jk,ik->i", C, K, C.conj()) + np.einsum("ij,ij->i", D, D.conj()))
    return np.sort(rates)


def relative_gap(fg, fh):
    scale = max(np.abs(fg).max(), np.abs(fh).max())
    return float(np.abs(fg - fh).max() / scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--classes", default=None)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw_order1.json"))
    args = parser.parse_args()
    data = bp.load_brec(BREC_RAW)
    classes = args.classes.split(",") if args.classes else list(data)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for variant in args.variants.split(","):
        start = time.time()
        for c in classes:
            for i, (g, h) in enumerate(data[c]):
                key = f"{variant}/{c}_{i}"
                if key in out:
                    continue
                ag, ah = bp.adjacency(g), bp.adjacency(h)
                out[key] = {s: relative_gap(order1(ag, variant, s), order1(ah, variant, s)) for s in SEEDS}
            json.dump(out, open(args.out, "w"))
            done = [v for k, v in out.items() if k.startswith(f"{variant}/{c}_")]
            print(variant, c, sum(max(v.values()) > THRESHOLD for v in done), "of", len(done),
                  f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
