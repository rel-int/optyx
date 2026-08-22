"""
Exact few-photon certificates for the photonic models.

Every cell is driven identically at every tick, so the record statistics
of the attenuated, two-herald post-selected ensemble are the uniform
average over injection pairs of the exact two-photon conditioned
dynamics — see ``docs/notebooks/beyond_3wl``. The step matrix of a graph
is assembled cell by cell from each box's own path matrix and glued by
the read/write routing of the :class:`optyx.interaction.CMap`, which
grounds it in the functor image; the assembly is checked against
``CMap.step.to_path()`` on small graphs by the tests.

Two engines compute the same ensembles:

- :func:`run_pair` branches over classical records, photons evolving
  under the (possibly kicked) assembled matrix of each branch — the only
  engine for the active model, whose feed-forward changes the matrix;
- :func:`passive_pair_bins` unrolls the fixed passive step into one
  spacetime transfer matrix and reads every two-photon record off
  permanents, which reaches sixty-vertex graphs in seconds.

The Bell model reduces to the same machinery: detecting the local
reference half of a pair collapses the network half onto one of two
fixed drive-rail superpositions, so its ensemble is the mixture, over
reference outcomes, of runs with the collapsed injections, keyed by the
outcomes.
"""

from itertools import combinations
from math import comb

import numpy as np

EINSUM = {1: "a,ai->i", 2: "ab,ai,bj->ij"}


class Layout:
    """How a model's cells sit inside the assembled step matrix.

    Rows are the drive rails of every cell, then one memory slot per
    paired port, then the internal memories; columns are the drive
    reflections and the taps (both measured), then the memory written
    for the next tick. ``block`` selects the physical rows and the
    network columns of one cell's path matrix."""

    n_drives = 1

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def block(self, degree, tap):
        key = (degree, round(tap, 12))
        if key not in self.cache:
            model = type(self.model)(tap=tap)
            array = np.asarray(
                model.vertex(degree).channel.to_path().array,
                dtype=complex)
            self.cache[key] = self.select(array, degree)
        return self.cache[key]

    def select(self, array, degree):
        return array[:degree + 2, :]


class FrozenLayout(Layout):
    """A layout carrying precomputed blocks, so a machine can run
    without optyx — on a worker that only has numpy."""

    def __init__(self, n_drives, blocks, reference=None):
        self.n_drives, self.cache = n_drives, dict(blocks)
        if reference is not None:
            self.reference = np.asarray(reference)

    def block(self, degree, tap):
        return self.cache[degree, round(tap, 12)]

    @staticmethod
    def freeze(layout, graph, taps):
        """Precompute every block a machine on ``graph`` needs."""
        blocks = {}
        for degree in sorted({len(nbrs) for nbrs in graph}):
            for tap in sorted(set(taps)):
                blocks[degree, round(tap, 12)] = layout.block(
                    degree, tap)
        return FrozenLayout(
            layout.n_drives, blocks, getattr(layout, "reference", None))


class PassiveLayout(Layout):
    """Rows ``drive, ports, memory``; columns ``reflection, ports,
    memory, tap``."""


class BellLayout(Layout):
    """Rows ``d1, d2, ports, memory``; columns ``d1', d2', ports,
    memory, tap``. The reference rails never mix with the network:
    their two-by-two unitary is kept aside in :attr:`reference`."""

    n_drives = 2

    def select(self, array, degree):
        rows = [0, 1] + list(range(4, degree + 5))
        cols = [0, 1] + list(range(4, degree + 6))
        reference = array[np.ix_([2, 3], [2, 3])]
        assert np.abs(array[np.ix_([2, 3], cols)]).max() < 1e-12
        assert np.abs(array[np.ix_(rows, [2, 3])]).max() < 1e-12
        if not hasattr(self, "reference"):
            self.reference = reference
        assert np.abs(self.reference - reference).max() < 1e-12
        return array[np.ix_(rows, cols)]


def assemble(graph, taps, layout):
    """The one-photon step matrix of a graph, cell ``b``'s tap set to
    ``taps[b]``."""
    n, k = len(graph), layout.n_drives
    pairs = [((u, graph[u].index(v)), (v, graph[v].index(u)))
             for u, nbrs in enumerate(graph) for v in nbrs if u < v]
    partner = {}
    for left, right in pairs:
        partner[left], partner[right] = right, left
    slot = {port: i for i, port in enumerate(sorted(partner))}
    n_paired = len(slot)
    n_rows = k * n + n_paired + n
    n_cols = (k + 1) * n + n_paired + n
    matrix = np.zeros((n_rows, n_cols), dtype=complex)

    def row(b, j, degree):
        return (k * b + j if j < k
                else k * n + slot[(b, j - k)] if j < k + degree
                else k * n + n_paired + b)

    def column(b, j, degree):
        return (k * b + j if j < k
                else (k + 1) * n + slot[partner[(b, j - k)]]
                if j < k + degree
                else (k + 1) * n + n_paired + b if j == k + degree
                else k * n + b)

    for b, nbrs in enumerate(graph):
        degree = len(nbrs)
        block = layout.block(degree, taps[b])
        for j in range(k + degree + 1):
            for i in range(k + degree + 2):
                matrix[row(b, j, degree),
                       column(b, i, degree)] += block[j, i]
    return matrix


class Machine:
    """A graph's step matrices, plain and kicked. A detection at cell
    ``c`` re-programs the taps of ``c`` and, when ``broadcast``, of its
    neighbours, from the next tick on; ``kicked_tap == tap`` is the
    passive machine."""

    def __init__(self, graph, layout, tap=0.3, kicked_tap=None,
                 broadcast=True):
        self.graph, self.layout, self.tap = graph, layout, tap
        self.kicked_tap = tap if kicked_tap is None else kicked_tap
        self.broadcast = broadcast
        n = self.n = len(graph)
        k = self.n_drives = layout.n_drives
        self.S = assemble(graph, [self.tap] * n, layout)
        self.n_in = self.S.shape[0]
        self.n_meas = (k + 1) * n
        self.cell_of_wire = [w // k for w in range(k * n)] \
            + list(range(n))
        self.adj = [set(nbrs) for nbrs in graph]
        self.cache = {}

    def step(self, kicked):
        region = set()
        for cell in kicked:
            region |= {cell} | (self.adj[cell] if self.broadcast
                                else set())
        key = frozenset(region)
        if key not in self.cache:
            taps = [self.tap] * self.n
            for cell in key:
                taps[cell] = self.kicked_tap
            self.cache[key] = assemble(self.graph, taps, self.layout)
        return self.cache[key]

    def rel(self, c, u):
        return 0 if c == u else 1 if c in self.adj[u] else 2

    def drive(self, cell, rail=0):
        return self.n_drives * cell + rail


def detections(n_meas, d):
    """Sorted ``d``-tuples of measured wires, with multiplicities."""
    if d == 1:
        return [((w,), 1) for w in range(n_meas)]
    return [((w1, w2), 2 if w1 != w2 else 1)
            for w1 in range(n_meas) for w2 in range(w1, n_meas)]


def invariant_key(machine, u, v, record, left, ctx=()):
    """Detection times, the relations of each detected cell to the
    injection pair, the pairwise relations between detected cells, the
    photons left inside and any model context."""
    cells = [machine.cell_of_wire[w] for _, w in record]
    times = [t for t, _ in record]
    singles = tuple(sorted(
        (t, tuple(sorted((machine.rel(c, u), machine.rel(c, v)))))
        for t, c in zip(times, cells)))
    pairs = tuple(sorted(
        (min(times[i], times[j]), max(times[i], times[j]),
         machine.rel(cells[i], cells[j]))
        for i in range(len(cells)) for j in range(i + 1, len(cells))))
    return (left, singles, pairs) + tuple(ctx)


def run_pair(machine, u, v, n_ticks, injection=None, ctx=(), raw=False):
    """The record distribution for two photons injected at tick zero:
    one in each of the drives of ``u`` and ``v`` by default, or an
    arbitrary symmetric two-photon ``injection`` over the input wires."""
    n_meas, n_in = machine.n_meas, machine.n_in
    K = slice(n_meas, None)
    bins = {}

    def add(record, left, p):
        key = ((record, left, ctx) if raw
               else invariant_key(machine, u, v, record, left, ctx))
        bins[key] = bins.get(key, 0.0) + p

    if injection is None:
        injection = np.zeros((n_in, n_in), dtype=complex)
        du, dv = machine.drive(u), machine.drive(v)
        injection[du, dv] = injection[dv, du] = 1 / 2 ** .5
    branches = {(): (frozenset(), injection)}
    for t in range(n_ticks):
        new = {}
        for record, (kicked, A) in branches.items():
            k = A.ndim
            S = machine.step(kicked)
            B = np.einsum(EINSUM[k], A, *([S] * k), optimize=True)
            keep = B[(K,) * k]
            if (np.abs(keep) ** 2).sum() > 1e-24:
                full = np.zeros((n_in,) * k, dtype=complex)
                offset = n_in - keep.shape[0]
                full[(slice(offset, None),) * k] = keep
                new[record] = (kicked, full)
            for d in range(1, k + 1):
                for wires, mult in detections(n_meas, d):
                    amp = (mult * comb(k, d)) ** .5 \
                        * B[wires + (K,) * (k - d)]
                    weight = (np.abs(amp) ** 2).sum() if k > d \
                        else np.abs(amp) ** 2
                    if weight <= 1e-24:
                        continue
                    rec = record + tuple((t, w) for w in wires)
                    if k == d:
                        add(rec, 0, float(np.abs(amp) ** 2))
                    else:
                        cells = frozenset(
                            machine.cell_of_wire[w] for w in wires)
                        full = np.zeros(n_in, dtype=complex)
                        full[n_in - amp.shape[0]:] = amp
                        new[rec] = (kicked | cells, full)
        branches = new
    for record, (kicked, A) in branches.items():
        p = float((np.abs(A) ** 2).sum())
        if p > 1e-24:
            add(record, A.ndim, p)
    return bins


def spacetime_transfer(machine, n_ticks):
    """The passive spacetime transfer matrix: every input wire against
    the measured wires of every tick, then the kept wires of the final
    tick. Only valid when ``kicked_tap == tap``."""
    assert machine.kicked_tap == machine.tap
    S, n_meas, n_in = machine.S, machine.n_meas, machine.n_in
    n_kept = S.shape[1] - n_meas
    offset = n_in - n_kept
    V = np.zeros((n_in, n_ticks * n_meas + n_kept), dtype=complex)
    P = np.eye(n_in, dtype=complex)
    for t in range(n_ticks):
        out = P @ S
        V[:, t * n_meas:(t + 1) * n_meas] = out[:, :n_meas]
        P = np.zeros((n_in, n_in), dtype=complex)
        P[:, offset:] = out[:, n_meas:]
    V[:, n_ticks * n_meas:] = P[:, offset:]
    return V


def passive_pair_bins(machine, V, n_ticks, u, v, injections=None,
                      ctx=()):
    """The binned record distribution of :func:`run_pair` for the
    passive machine, from the spacetime transfer matrix: amplitudes of
    two independent photons are permanents of two-by-two submatrices."""
    n_meas = machine.n_meas
    n_out = n_ticks * n_meas
    if injections is None:
        injections = [(np.eye(machine.n_in)[machine.drive(u)],
                       np.eye(machine.n_in)[machine.drive(v)], 1., ctx)]
    bins = {}

    def add(key, p):
        bins[key] = bins.get(key, 0.0) + p

    kept = V[:, n_out:]
    gram = kept @ kept.conj().T
    for x, y, weight, tag in injections:
        a, b = (x @ V)[:n_out], (y @ V)[:n_out]
        perm = np.outer(a, b)
        perm = perm + perm.T
        prob = np.abs(perm) ** 2
        prob[np.arange(n_out), np.arange(n_out)] /= 2
        for o1 in range(n_out):
            row = prob[o1, o1:]
            for off in np.nonzero(row > 1e-24)[0]:
                o2 = o1 + int(off)
                t1, w1 = divmod(o1, n_meas)
                t2, w2 = divmod(o2, n_meas)
                add(invariant_key(
                    machine, u, v, ((t1, w1), (t2, w2)), 0, tag),
                    weight * float(row[off]))
        left_x = float((x @ gram @ x.conj()).real)
        left_y = float((y @ gram @ y.conj()).real)
        cross = y @ gram @ x.conj()
        for o in range(n_out):
            t, w = divmod(o, n_meas)
            p = (np.abs(a[o]) ** 2 * left_y
                 + np.abs(b[o]) ** 2 * left_x
                 + 2 * (a[o] * np.conj(b[o]) * cross).real)
            if p > 1e-24:
                add(invariant_key(machine, u, v, ((t, w),), 1, tag),
                    weight * float(p))
        stay = left_x * left_y + abs(cross) ** 2
        add(invariant_key(machine, u, v, (), 2, tag),
            weight * float(stay))
    return bins


def bell_injections(machine, u, v):
    """The mixture over reference outcomes: detecting the local half of
    cell ``c``'s pair at outcome ``r`` collapses the network half onto
    ``sum_k R[k, r] e_{drive(c, k)} / sqrt(2)``."""
    R = machine.layout.reference
    out = []
    for ru in range(2):
        for rv in range(2):
            x = np.zeros(machine.n_in, dtype=complex)
            y = np.zeros(machine.n_in, dtype=complex)
            for k in range(2):
                x[machine.drive(u, k)] = R[k, ru] / 2 ** .5
                y[machine.drive(v, k)] = R[k, rv] / 2 ** .5
            out.append((x, y, 1., ((min(ru, rv), max(ru, rv)),)))
    return out


def aggregate(machine, n_ticks, engine="branch", ensemble=None):
    """The post-selected two-herald ensemble: the uniform average over
    unordered injection pairs — all of them, or an invariant
    ``ensemble`` such as the edges — of the invariantly binned record
    distribution."""
    n = machine.n
    pairs = (list(combinations(range(n), 2)) if ensemble is None
             else sorted(ensemble))
    total = {}
    V = (spacetime_transfer(machine, n_ticks)
         if engine == "spacetime" else None)
    bell = machine.n_drives == 2
    for u, v in pairs:
        injections = bell_injections(machine, u, v) if bell else None
        if engine == "spacetime":
            bins = passive_pair_bins(
                machine, V, n_ticks, u, v, injections)
        else:
            bins = {}
            for x, y, w, tag in (injections or [(None, None, 1., ())]):
                injection = None
                if x is not None:
                    injection = (np.outer(x, y) + np.outer(y, x)) \
                        / 2 ** .5
                for key, p in run_pair(
                        machine, u, v, n_ticks, injection=injection,
                        ctx=tag).items():
                    bins[key] = bins.get(key, 0.0) + w * p
        for key, p in bins.items():
            total[key] = total.get(key, 0.0) + p / len(pairs)
    return total


def separation(bins_left, bins_right):
    """The largest difference between two ensembles, bin by bin."""
    keys = set(bins_left) | set(bins_right)
    return float(max(abs(bins_left.get(key, 0.0)
                         - bins_right.get(key, 0.0)) for key in keys))


def machine_for(model, graph, frozen=False, **settings):
    """The exact machine of a photonic model on a graph. ``frozen``
    precomputes the blocks so the machine pickles without optyx."""
    from models import ActiveModel, BellModel, PassiveModel
    if isinstance(model, BellModel):
        layout, taps = BellLayout(model), [model.tap]
    elif isinstance(model, ActiveModel):
        layout, taps = PassiveLayout(PassiveModel(model.tap)), \
            [model.tap, model.kick]
        settings = {"kicked_tap": model.kick, **settings}
    elif isinstance(model, PassiveModel):
        layout, taps = PassiveLayout(model), [model.tap]
    else:
        raise ValueError(model)
    if frozen:
        layout = FrozenLayout.freeze(layout, graph, taps)
    return Machine(graph, layout, tap=model.tap, **settings)
