"""Dataset builder for the k-WL-ladder benchmark.

Convention: k-FWL means the folklore k-dimensional Weisfeiler-Leman
algorithm on k-tuples of vertices, which is equivalent in distinguishing
power to the tuple (k+1)-WL algorithm; 1-FWL means colour refinement.

Pairs are drawn from the BREC benchmark (Wang & Zhang, 2023), which must
be cloned so that its raw ``.npy`` files sit at ``BREC_PATH``, plus a few
constructed classic pairs.  Running ``build()`` measures the smallest k
in 1..3 at which each candidate pair is distinguished and writes the
selection to ``data/dataset.json`` next to this module.

Certificates are canonical across graphs: each one is the round-by-round
sorted multiset of refinement signatures, where signatures are written
over rank colour codes.  Round zero uses raw structural codes, so ranks
mean the same structure in any two graphs whose certificates agree on
every earlier round; the first disagreeing round makes the certificates
differ, hence certificate equality coincides with k-FWL equivalence.
"""

import json
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

BREC_PATH = "/home/user/graphpku/brec/customize/Data/raw"
BREC_FILES = ("basic", "regular", "str", "extension", "cfi", "4vtx", "dr")
DATA_PATH = Path(__file__).parent / "data" / "dataset.json"
CONVENTION = "k-FWL == tuple (k+1)-WL; 1-FWL == colour refinement"
FWL3_MAX_N = 110
CFI_MAX_N = 110
VF2_MAX_N = 30


def wl1(g):
    """Colour refinement certificate of a graph.

    Starting from uniform colours, each round recolours a vertex by the
    rank of its signature, the pair of its old colour and the sorted
    colours of its neighbours, iterated until the partition stabilises.
    The certificate is the tuple, one entry per round, of the sorted
    multiset of signatures; its last entry determines the stable sorted
    colour multiset.
    """
    nodes = sorted(g.nodes)
    colours = dict.fromkeys(nodes, 0)
    cert = [(len(nodes), )]
    n_colours = 1
    while True:
        sigs = {v: (colours[v], tuple(sorted(colours[u] for u in g[v])))
                for v in nodes}
        ranking = {s: i for i, s in enumerate(sorted(set(sigs.values())))}
        cert.append(tuple(sorted(sigs.values())))
        colours = {v: ranking[sigs[v]] for v in nodes}
        if len(ranking) == n_colours:
            return tuple(cert)
        n_colours = len(ranking)


def _recode(rows):
    """Rank the rows of an int64 matrix via a structured view.

    Returns the rank of each row under bytewise order, the unique rows
    as bytes and their counts as bytes, all computed with ``np.unique``
    so that the result is deterministic and never uses Python ``hash``.
    """
    void = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    view = np.ascontiguousarray(rows).view(void)[:, 0]
    uniq, inverse, counts = np.unique(
        view, return_inverse=True, return_counts=True)
    return (inverse.reshape(-1).astype(np.int64),
            uniq.tobytes(), counts.tobytes())


def _base_colours(adjacency, k):
    """Ordered isomorphism type of every k-tuple of vertices.

    The type of a tuple packs, for every pair of positions, whether the
    two entries are equal and whether they are adjacent.
    """
    n = len(adjacency)
    axes = [np.arange(n).reshape((1, ) * i + (n, ) + (1, ) * (k - 1 - i))
            for i in range(k)]
    base = np.zeros((n, ) * k, dtype=np.int64)
    for i, j in combinations(range(k), 2):
        base = 4 * base + 2 * (axes[i] == axes[j])\
            + adjacency[axes[i], axes[j]]
    return base


def _replacement(colours, i):
    """Colour of a tuple with position ``i`` set to w, indexed by (t, w)."""
    return np.expand_dims(np.moveaxis(colours, i, -1), i)


def _fwl(g, k, chunk=16):
    """Folklore k-WL certificate of a graph.

    Tuples start from their ordered isomorphism type; each round maps a
    tuple to the rank of its signature, the pair of its old colour and
    the sorted multiset over vertices w of the k colours obtained by
    substituting w at each position, iterated until the partition of
    tuples stabilises.  Colours are int64 codes recoded each round with
    ``np.unique`` on structured views, and the substitution colours are
    materialised in chunks over w to bound memory.  The certificate is
    the tuple of per-round sorted signature multisets, whose last entry
    determines the stable sorted colour multiset.
    """
    nodes = sorted(g.nodes)
    n = len(nodes)
    adjacency = nx.to_numpy_array(g, nodelist=nodes, dtype=np.int64)
    colours, uniq, counts = _recode(_base_colours(adjacency, k).reshape(-1, 1))
    cert = [(uniq, counts)]
    n_colours = len(counts) // np.dtype(np.int64).itemsize
    while True:
        replaced = [_replacement(colours.reshape((n, ) * k), i)
                    for i in range(k)]
        codes = np.empty((n ** k, n), dtype=np.int64)
        for lo in range(0, n, chunk):
            acc = np.zeros((1, ) * k + (1, ), dtype=np.int64)
            for r in replaced:
                acc = acc * n_colours + r[..., lo:lo + chunk]
            codes[:, lo:lo + chunk] = acc.reshape(n ** k, -1)
        codes.sort(axis=1)
        sigs = np.concatenate([colours.reshape(-1, 1), codes], axis=1)
        colours, uniq, counts = _recode(sigs)
        cert.append((uniq, counts))
        new_n_colours = len(counts) // np.dtype(np.int64).itemsize
        if new_n_colours == n_colours:
            return tuple(cert)
        n_colours = new_n_colours


def fwl2(g):
    """Folklore 2-WL certificate, equivalent to tuple 3-WL."""
    return _fwl(g, 2)


def fwl3(g):
    """Folklore 3-WL certificate, equivalent to tuple 4-WL."""
    return _fwl(g, 3)


def distinguishes(cert_fn, g, h):
    """Whether the certificates of ``g`` and ``h`` differ."""
    return cert_fn(g) != cert_fn(h)


def level(g, h, cap=3):
    """Smallest k in 1..cap such that k-FWL distinguishes ``g`` and ``h``.

    Returns ``None`` if no k up to ``cap`` distinguishes them; recall
    k-FWL == tuple (k+1)-WL and 1-FWL == colour refinement.
    """
    for k, cert_fn in list(enumerate((wl1, fwl2, fwl3), 1))[:cap]:
        if distinguishes(cert_fn, g, h):
            return k
    return None


def _graph6(entry):
    """Normalise a str/bytes/np.str_/np.bytes_ entry to a graph6 string."""
    return entry.decode() if isinstance(entry, bytes) else str(entry)


def from_graph6(string):
    """The graph encoded by a graph6 string."""
    return nx.from_graph6_bytes(string.encode())


def to_graph6(g):
    """The graph6 string encoding a graph."""
    return nx.to_graph6_bytes(g, header=False).decode().strip()


def load_brec():
    """Load the BREC benchmark as a dict from category to graph pairs.

    Categories stored flat pair up consecutive entries; the others store
    explicit shape-(2, ) rows of graph6 strings or bytes.
    """
    result = {}
    for name in BREC_FILES:
        entries = np.load(f"{BREC_PATH}/{name}.npy", allow_pickle=True)
        rows = (zip(entries[0::2], entries[1::2])
                if entries.ndim == 1 else entries)
        result[name] = [(from_graph6(_graph6(x)), from_graph6(_graph6(y)))
                        for x, y in rows]
    return result


def _cost(g, h):
    """Selection cost ``n * (max_degree + 3)`` of a pair."""
    max_degree = max(d for _, d in list(g.degree) + list(h.degree))
    return g.number_of_nodes() * (max_degree + 3)


def _timed_fwl3(g, label, log):
    """Run ``fwl3`` and append its wall time to the log."""
    start = time.perf_counter()
    cert = fwl3(g)
    log.append(f"fwl3 {label} (n={g.number_of_nodes()}):"
               f" {time.perf_counter() - start:.2f}s")
    return cert


def _shallow(g, h):
    """The wl1 and fwl2 verdicts of a pair, skipping fwl2 when moot."""
    w1 = "differ" if distinguishes(wl1, g, h) else "same"
    w2 = w1 if w1 == "differ" else (
        "differ" if distinguishes(fwl2, g, h) else "same")
    return w1, w2


def _candidate(pair_id, category, g, h, w1, w2):
    """A candidate record holding the graphs and the shallow verdicts."""
    return {"id": pair_id, "category": category, "graphs": (g, h),
            "cost": _cost(g, h), "w1": w1, "w2": w2}


def _deep(candidate, log):
    """Run fwl3 on a candidate if small enough and record the verdict."""
    g, h = candidate["graphs"]
    if g.number_of_nodes() > FWL3_MAX_N:
        candidate["w3"] = "skipped"
    else:
        certs = tuple(_timed_fwl3(graph, f"{candidate['id']}[{side}]", log)
                      for side, graph in enumerate((g, h)))
        candidate["w3"] = "differ" if certs[0] != certs[1] else "same"
    verdicts = [candidate["w1"], candidate["w2"], candidate["w3"]]
    candidate["level"] = next(
        (k for k, v in enumerate(verdicts, 1) if v == "differ"), None)
    return candidate


def _record(candidate, rung, log):
    """The JSON record of a selected candidate, with a vf2 check."""
    g, h = candidate["graphs"]
    n = g.number_of_nodes()
    if n <= VF2_MAX_N:
        verdict = "isomorphic (!)" if nx.is_isomorphic(g, h)\
            else "non-isomorphic"
        log.append(f"vf2 {candidate['id']}: {verdict}")
    else:
        log.append(f"vf2 {candidate['id']}: skipped (n={n} > {VF2_MAX_N})")
    return {
        "id": candidate["id"], "rung": rung,
        "category": candidate["category"],
        "g6": [to_graph6(g), to_graph6(h)], "n": n,
        "edges": max(g.number_of_edges(), h.number_of_edges()),
        "max_degree": max(d for _, d in list(g.degree) + list(h.degree)),
        "level": candidate["level"],
        "verified": {"wl1": candidate["w1"], "fwl2": candidate["w2"],
                     "fwl3": candidate["w3"]}}


def _constructed_pairs():
    """The constructed control and classic pairs."""
    c6, p6 = nx.cycle_graph(6), nx.path_graph(6)
    return [
        ("control-c6-vs-p6", "classic", c6, p6),
        ("control-k15-vs-p6", "classic", nx.star_graph(5), p6),
        ("classic-2c3-vs-c6", "classic",
         nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)), c6)]


def _scan(brec, categories, log):
    """Shallow-scan the given BREC categories into candidate records."""
    candidates = []
    for category in categories:
        for i, (g, h) in enumerate(brec[category]):
            w1, w2 = _shallow(g, h)
            candidates.append(
                _candidate(f"{category}-{i:02d}", category, g, h, w1, w2))
    return candidates


def _select_rung1(candidates, classic, log):
    """The five 1fwl-blind pairs: cheapest level-2 candidates.

    Four come from at least two BREC categories, the fifth is the
    classic pair when it measures at level 2.
    """
    eligible = sorted(
        (c for c in candidates if (c["w1"], c["w2"]) == ("same", "differ")),
        key=lambda c: (c["cost"], c["id"]))
    chosen = eligible[:3]
    others = [c for c in eligible[3:]
              if len({d["category"] for d in chosen + [c]}) >= 2]
    chosen += others[:1] if others else eligible[3:4]
    if len({c["category"] for c in chosen}) < 2:
        log.append("1fwl-blind: only one BREC category had level-2 pairs")
    if classic["level"] == 2:
        chosen.append(classic)
    else:
        log.append("1fwl-blind: classic 2C3-vs-C6 measured level"
                   f" {classic['level']}, not included")
        chosen += eligible[4:5]
    if len(chosen) < 5:
        log.append(f"1fwl-blind: only {len(chosen)} of 5 level-2 pairs found")
    return chosen


def _select_rung2(candidates, log):
    """The five 2fwl-blind pairs: cheapest fwl2-blind candidates that
    fwl3 distinguishes, running fwl3 in cost order until five confirm,
    with at most three pairs per BREC category for diversity."""
    eligible = sorted(
        (c for c in candidates if (c["w1"], c["w2"]) == ("same", "same")),
        key=lambda c: (c["cost"], c["id"]))
    chosen, spillover = [], []
    for candidate in eligible:
        if len(chosen) == 5:
            break
        if sum(c["category"] == candidate["category"]
               for c in chosen) >= 3:
            log.append(f"2fwl-blind scan: {candidate['id']} skipped for"
                       " category diversity")
            continue
        _deep(candidate, log)
        if candidate["level"] == 3:
            chosen.append(candidate)
        else:
            spillover.append(candidate)
            log.append(f"2fwl-blind scan: {candidate['id']} measured level"
                       f" {candidate['level']}, not level 3")
    return chosen, spillover


def _select_rung3(brec, log):
    """The five 3fwl-blind pairs: smallest candidates fwl3 confirms same.

    Checks the small cfi pairs and every dr pair, then 4vtx pairs only
    while the rung is underfull; measured levels are honoured, so a dr
    or cfi pair that fwl3 distinguishes is reported instead of selected.
    """
    small_cfi = [c for c in _scan(brec, ("cfi", ), log)
                 if c["graphs"][0].number_of_nodes() <= CFI_MAX_N]
    pool = small_cfi + sorted(
        _scan(brec, ("dr", ), log), key=lambda c: (c["cost"], c["id"]))
    chosen, leftovers = [], []
    for candidate in pool + sorted(
            _scan(brec, ("4vtx", ), log), key=lambda c: (c["cost"], c["id"])):
        if len(chosen) >= 5 and candidate["category"] == "4vtx":
            break
        if (candidate["w1"], candidate["w2"]) != ("same", "same"):
            candidate["w3"] = "skipped"
            candidate["level"] = 1 if candidate["w1"] == "differ" else 2
            leftovers.append(candidate)
            log.append(f"3fwl-blind scan: {candidate['id']} measured level"
                       f" {candidate['level']} before fwl3")
            continue
        _deep(candidate, log)
        if candidate["level"] is None and candidate["w3"] == "same":
            chosen.append(candidate)
        else:
            leftovers.append(candidate)
            log.append(f"3fwl-blind scan: {candidate['id']} measured level"
                       f" {candidate['level']}")
    chosen.sort(key=lambda c: (c["cost"], c["id"]))
    return chosen[:5], leftovers


def _fill(rung, chosen, leftovers, wanted_level, log):
    """Top an underfull rung up from measured leftovers of that level."""
    fillers = sorted(
        (c for c in leftovers if c.get("level") == wanted_level),
                     key=lambda c: (c["cost"], c["id"]))
    for filler in fillers[:5 - len(chosen)]:
        chosen.append(filler)
        log.append(f"{rung}: filled with {filler['id']}")
    if len(chosen) < 5:
        log.append(f"{rung}: only {len(chosen)} of 5 pairs found, no further"
                   f" measured candidates at the required level")
    return chosen


def build():
    """Build the dataset, write it to ``DATA_PATH`` and return it."""
    start = time.perf_counter()
    log = []
    brec = load_brec()
    constructed = [_deep(_candidate(pair_id, category, g, h, *_shallow(g, h)),
                         log)
                   for pair_id, category, g, h in _constructed_pairs()]
    controls, classic = constructed[:2], constructed[2]
    for control in controls:
        if control["w1"] != "differ":
            log.append(f"control: {control['id']} is not wl1-distinguishable")
    rung1 = _select_rung1(
        _scan(brec, ("basic", "regular", "extension"), log), classic, log)
    for candidate in rung1:
        if "w3" not in candidate:
            _deep(candidate, log)
    rung2, spill2 = _select_rung2(_scan(brec, ("str", ), log), log)
    rung3, leftovers = _select_rung3(brec, log)
    rung2 = _fill("2fwl-blind", rung2, leftovers + spill2, 3, log)
    rung3 = _fill("3fwl-blind", rung3, spill2, None, log)
    pairs = [_record(c, rung, log) for rung, chosen in (
        ("control", controls), ("1fwl-blind", rung1),
        ("2fwl-blind", rung2), ("3fwl-blind", rung3))
        for c in chosen]
    log.append(f"total build time: {time.perf_counter() - start:.2f}s")
    dataset = {"convention": CONVENTION, "pairs": pairs, "log": log}
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset


if __name__ == "__main__":
    for line in build()["log"]:
        print(line)
