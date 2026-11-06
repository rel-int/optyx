"""BREC's protocol on the 46-pair subset of brec_exact, four processes."""
import json
import sys
import time
from multiprocessing import Pool

import brec_ep as B

SUBSET = {"Basic": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54],
          "Regular": [0, 10, 20, 30, 40, 47],
          "Strongly_regular": [0, 5, 10, 20, 30, 40],
          "Extension": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
          "CFI": [5, 11, 12, 17, 18, 31, 57, 73, 76],
          "4-vertex_condition": [0, 10],
          "Distance_regular": [1, 2, 4, 5, 18, 19]}
OFFSET = {"Basic": 0, "Regular": 60, "Strongly_regular": 110,
          "Extension": 160, "CFI": 260, "4-vertex_condition": 360,
          "Distance_regular": 380}
DATA = None


def job(spec):
    global DATA
    if DATA is None:
        DATA = B.load(sys.argv[1])
    category, index = spec
    start = time.time()
    result = B.run_pair(DATA, OFFSET[category] + index)
    result.update(category=category, seconds=time.time() - start)
    return result


if __name__ == "__main__":
    jobs = [(c, i) for c, idx in SUBSET.items() for i in idx]
    jobs.sort(key=lambda j: j[0] in ("CFI", "Distance_regular"))
    with Pool(4) as pool:
        results = []
        for r in pool.imap_unordered(job, jobs):
            results.append(r)
            print(r["category"], r["pair"], r["vertices"],
                  "untrained", r["untrained"]["decision"],
                  round(r["untrained"]["test"], 1),
                  "trained", r["trained"]["decision"],
                  round(r["trained"]["test"], 1),
                  round(r["trained"]["reliability"], 1),
                  "epochs", len(r["history"]), round(r["seconds"]), "s",
                  flush=True)
    with open(sys.argv[2], "w") as f:
        json.dump(results, f, indent=1)
