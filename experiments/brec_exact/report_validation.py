"""Tables of Level 1 of the sampler validation, from
results/raw_validation.json.gz.

    python report_validation.py [results/raw_validation.json.gz]

For every pair, graph, source, brightness and order: how many sets carry at
least MIN_COUNTS expected coincidences (the others are left out of the
pulls, whose normal law needs them); over those sets, the mean and spread
of the pulls (F_hat - F) / sqrt(tau Var / T), with tau the exact
autocorrelation time where it is computed and the sampled one otherwise;
the median and extreme ratios of the sample to the exact variance, and the
spread and extreme of their z-scores against the batch standard error; the
exact and sampled autocorrelation times.
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_COUNTS = 100


def order_rows(record):
    for N, o in sorted(record["orders"].items()):
        F, var, mean, sample_var, var_se = (np.array(o[k]) for k in
                                            ("rate", "var", "mean", "sample_var", "var_se"))
        T = o["T"]
        counted = F * T >= MIN_COUNTS if record["w"] < 1 or record["source"] == "photons p=0" \
            else np.ones(len(F), bool)
        tau = np.array(o["tau_blocks"])
        pull = (mean - F) / np.sqrt(np.maximum(tau, 0.5) * var / T)
        z = (sample_var - var) / var_se
        ratio = sample_var / var
        yield int(N), {
            "sets": len(F), "counted": int(counted.sum()), "T": T,
            "pull_mean": pull[counted].mean() if counted.any() else np.nan,
            "pull_sd": pull[counted].std() if counted.sum() > 1 else np.nan,
            "ratio_median": np.median(ratio[counted]) if counted.any() else np.nan,
            "ratio_min": ratio[counted].min() if counted.any() else np.nan,
            "ratio_max": ratio[counted].max() if counted.any() else np.nan,
            "z_sd": z[counted].std() if counted.sum() > 1 else np.nan,
            "z_max": np.abs(z[counted]).max() if counted.any() else np.nan,
            "rel_se": np.median(var_se[counted] / var[counted]) if counted.any() else np.nan,
            "tau_exact": (min(o["tau_exact"]), max(o["tau_exact"])) if "tau_exact" in o else None,
            "tau_blocks": float(np.median(tau)),
        }


def tables(records):
    lines = ["| pair | graph | source | w | N | sets (counted) | pull mean | pull sd | "
             "var ratio median [min, max] | rel. s.e. | z sd | max abs z | tau exact | tau sampled |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(records, key=lambda k: (records[k]["pair"], records[k]["side"],
                                              records[k]["source"], records[k]["w"])):
        r = records[key]
        for N, row in order_rows(r):
            tau = "" if row["tau_exact"] is None else f"{row['tau_exact'][0]:.3f}-{row['tau_exact'][1]:.3f}"
            lines.append(
                f"| {r['pair']} | {r['side']} | {r['source']} | {r['w']:g} | {N} | {row['sets']} "
                f"({row['counted']}) | {row['pull_mean']:+.2f} | {row['pull_sd']:.2f} | "
                f"{row['ratio_median']:.4f} [{row['ratio_min']:.4f}, {row['ratio_max']:.4f}] | "
                f"{row['rel_se']:.1e} | {row['z_sd']:.2f} | {row['z_max']:.1f} | {tau} | "
                f"{row['tau_blocks']:.3f} |")
    return "\n".join(lines)


def worst(records):
    """Per source, the largest |z| of a variance and the ratio behind it."""
    out = {}
    for r in records.values():
        for N, o in r["orders"].items():
            F, var, sample_var, var_se = (np.array(o[k]) for k in ("rate", "var", "sample_var", "var_se"))
            counted = (F * o["T"] >= MIN_COUNTS) | (r["w"] >= 1) & (r["source"] != "photons p=0")
            if not counted.any():
                continue
            z = np.abs(sample_var - var) / var_se
            i = int(np.argmax(np.where(counted, z, -1)))
            best = out.get(r["source"])
            if best is None or z[i] > best[0]:
                out[r["source"]] = (z[i], sample_var[i] / var[i], r["pair"], r["side"], r["w"], N)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "raw_validation.json.gz")
    records = json.load(gzip.open(path, "rt"))
    print(tables(records))
    print()
    for source, (z, ratio, pair, side, w, N) in worst(records).items():
        print(f"{source}: largest |z| {z:.1f}, ratio {ratio:.4f} ({pair} {side}, w={w:g}, order {N})")


if __name__ == "__main__":
    main()
