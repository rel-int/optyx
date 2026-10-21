"""Ad hoc check on Modal: Wick means and the synthetic variance of a linear
model against its sampler, on one pair."""
import modal
from gnn_app import image

app = modal.App("brec-gnn-check")


@app.function(image=image, cpu=8, memory=16384, timeout=1800)
def check(g6, h6, model, seed=100, nmax=4, samples=50000):
    from math import log
    import numpy as np
    import torch
    import brecphot as bp
    import gnn_bench as gb
    torch.set_num_threads(8)
    lines, keep = [], {}
    for side, g in (("G", g6), ("H", h6)):
        adj = bp.adjacency(g); n = adj.shape[0]
        C = gb.linear_covariance(model, adj, seed)
        stats, counts = gb.sampled_statistics(model, adj, seed, samples, nmax, "cpu", noise_seed=50)
        counts = np.asarray(counts)
        lines.append(f"{side}: counts per batch {counts[:3]}, stats[4] shapes {[np.asarray(s).shape for s in stats[4]]}")
        keep[side] = {}
        for N in range(1, nmax + 1):
            sets = np.array(list(gb.combinations(range(n), N)))
            wick = gb.hafnians_of_doubled_sets(C, sets)
            m0 = np.asarray(stats[N][0]); v0 = np.asarray(stats[N][1])
            samp = m0.mean(0)
            var_used = v0.mean(0) + m0.var(0) * counts.mean()
            keep[side][N] = (wick, var_used, samp)
            lines.append(f"  order {N}: wick mean {wick.mean():.4g}, sampled mean {samp.mean():.4g}, "
                         f"mean |wick-samp|/wick {np.mean(np.abs(wick - samp)) / wick.mean():.2e}, "
                         f"var used/mean^2 {np.mean(var_used / wick ** 2):.3g} "
                         f"(within {np.mean(v0.mean(0) / wick ** 2):.3g}, between*count {np.mean(m0.var(0) * counts.mean() / wick ** 2):.3g})")
    for N in range(1, nmax + 1):
        wg, vg, sg = keep["G"][N]; wh, vh, sh = keep["H"][N]
        fg, fh = np.sort(wg), np.sort(wh)
        d2 = np.sum((fg - fh) ** 2 / (vg[np.argsort(wg)] + vh[np.argsort(wh)]))
        lines.append(f"order {N}: sorted rel gap wick {np.max(np.abs(fg - fh)) / fg.mean():.2e}, sampled {np.max(np.abs(np.sort(sg) - np.sort(sh))) / fg.mean():.2e}, "
                     f"ideal count {4 * log(20) / d2 if d2 > 0 else np.inf:.2e}, noise/level per entry at T=1e3 {np.sqrt(vg.mean() / 1e3) / fg.mean():.2e}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(cat: str = "Strongly_regular", i: int = 0, model: str = "step-lin"):
    import sys
    sys.path.insert(0, ".")
    import brecphot as bp
    data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
    g6, h6 = data[cat][i]
    print(check.remote(g6, h6, model))
