"""
Exact few-photon certificates of a map of passive cells, read on the
output modes of the last tick.

Every drive receives a fresh photon at every tick, so attenuating the
sources and post-selecting on two heralds leaves the uniform average,
over unordered pairs of distinct injection slots ``(vertex, tick)``, of
the exact two-photon dynamics — the certificate of PR #69 with the slots
of every tick in the ensemble, since here only the last tick is read.
The step matrix is :func:`assemble`\\ d from the path matrix of every
box, glued by the read and write routing of the
:class:`optyx.interaction.CMap`; it equals ``CMap.step.to_path()``,
which the tests assert. :func:`transfer` unrolls it into one spacetime
matrix, :func:`amplitudes` gives every slot its single-photon amplitude
over the outputs of every tick and the wires left at the end, and the
statistics of two photons are permanents of two-by-two submatrices.

The readout is the multiset over vertices of the distribution of the
photon count on each output mode, sorted outcome by outcome as the
ladder's per-cell trajectory multiset, and the distribution of the
total count: :func:`bins`. Two single-particle certificates share the
same readout: one heralded photon, and a coherent-state drive of
matched mean photon number whose counts are Poisson in the intensity
of the coherent sum of the slot amplitudes.
"""

from itertools import combinations

import numpy as np


def assemble(cmap):
    """The one-photon step matrix of a map of passive cells, rows and
    columns indexed by ``cmap.wires``: the path matrix of each box,
    the port it writes sent to the slot its partner reads."""
    index = {wire: i for i, wire in enumerate(cmap.wires)}
    matrix = np.zeros((len(index), len(index)), dtype=complex)
    for b, box in enumerate(cmap.boxes):
        inputs = [(b, j) for j in range(len(box.ports @ box.memory))]
        rows = [index[wire] for wire in inputs]
        columns = [index[cmap.partner.get(wire, wire)] for wire in inputs]
        matrix[np.ix_(rows, columns)] += np.asarray(
            box.channel.to_path().array, dtype=complex)
    return matrix


def transfer(step, n_out, n_ticks):
    """The spacetime transfer matrix: from every wire read at the first
    tick to the ``n_out`` outputs of every tick, then the wires kept
    after the last one. The outputs are the first columns of the step
    and the drives its first rows."""
    n = step.shape[0]
    matrix = np.zeros((n, n_ticks * n_out + n - n_out), dtype=complex)
    kept = np.eye(n, dtype=complex)
    for tick in range(n_ticks):
        out = kept @ step
        matrix[:, tick * n_out:(tick + 1) * n_out] = out[:, :n_out]
        kept = out.copy()
        kept[:, :n_out] = 0
    matrix[:, n_ticks * n_out:] = kept[:, n_out:]
    return matrix


def amplitudes(step, n_out, n_ticks, drives=None):
    """``A[t, v]``: the amplitude of one photon injected into the
    ``v``-th drive row at tick ``t``, over the spacetime columns of
    :func:`transfer`; the drives are the first ``n_out`` rows unless
    given. Distinct slots give orthonormal rows."""
    n = step.shape[0]
    drives = list(range(n_out)) if drives is None else list(drives)
    result = np.zeros((n_ticks, len(drives), n_ticks * n_out + n - n_out),
                      dtype=complex)
    for tick in range(n_ticks):
        later = transfer(step, n_out, n_ticks - tick)[drives]
        result[tick, :, tick * n_out:n_ticks * n_out] = \
            later[:, :(n_ticks - tick) * n_out]
        result[tick, :, n_ticks * n_out:] = later[:, (n_ticks - tick) * n_out:]
    return result


def bosons(a, b):
    """The joint amplitudes of two indistinguishable photons, squared:
    ``|a_i b_j + a_j b_i|^2``."""
    return np.abs(np.outer(a, b) + np.outer(b, a)) ** 2


def distinguishable(a, b):
    """The joint probabilities of two distinguishable photons, which
    never interfere: ``|a_i|^2 |b_j|^2 + |a_j|^2 |b_i|^2``."""
    a, b = np.abs(a) ** 2, np.abs(b) ** 2
    return np.outer(a, b) + np.outer(b, a)


def count_statistics(pairs, measured, joint_of=bosons):
    """The per-mode and total count distributions of two photons in
    orthogonal single-photon states ``a, b``, averaged over the
    ``(a, b)`` pairs: ``joint_of(a, b)[i, j]`` is the probability of
    one photon in each of two modes, half of it that of two in one."""
    n_out = len(measured)
    nodes, total = np.zeros((n_out, 3)), np.zeros(3)
    rest = np.ones(pairs[0][0].shape[0], dtype=bool)
    rest[measured] = False
    for a, b in pairs:
        joint = joint_of(a, b)
        diagonal = np.diag(joint)
        nodes[:, 2] += diagonal[measured] / 2
        nodes[:, 1] += joint[measured].sum(axis=1) - diagonal[measured]
        nodes[:, 0] += (joint.sum() - 2 * joint[measured].sum(axis=1)
                        + diagonal[measured]) / 2
        total[2] += joint[np.ix_(measured, measured)].sum() / 2
        total[1] += joint[np.ix_(measured, rest)].sum()
        total[0] += joint[np.ix_(rest, rest)].sum() / 2
    return nodes / len(pairs), total / len(pairs)


def two_photon(slots, measured):
    """The two-herald certificate: uniform over pairs of distinct slots."""
    return count_statistics(
        list(combinations(slots.reshape(-1, slots.shape[-1]), 2)), measured)


def two_distinguishable(slots, measured):
    """The two-herald certificate with distinguishable photons."""
    return count_statistics(
        list(combinations(slots.reshape(-1, slots.shape[-1]), 2)), measured,
        distinguishable)


def one_photon(slots, measured):
    """The one-herald certificate: uniform over slots."""
    probabilities = np.abs(slots.reshape(-1, slots.shape[-1])) ** 2
    nodes = np.zeros((len(measured), 3))
    nodes[:, 1] = probabilities[:, measured].mean(axis=0)
    nodes[:, 0] = 1 - nodes[:, 1]
    total = np.array([1 - nodes[:, 1].sum(), nodes[:, 1].sum(), 0.])
    return nodes, total


def coherent(slots, measured, mean_photons=2):
    """A coherent state of the same amplitude in every slot, with
    ``mean_photons`` injected in total: the outputs are coherent states
    of the summed amplitudes and the counts are Poisson."""
    n_slots = slots.shape[0] * slots.shape[1]
    intensity = mean_photons / n_slots * np.abs(
        slots.reshape(n_slots, -1).sum(axis=0)[measured]) ** 2

    def poisson(rate):
        return np.exp(-rate) * np.array([1, rate, rate ** 2 / 2])

    return np.stack([poisson(rate) for rate in intensity]), \
        poisson(intensity.sum())


CERTIFICATES = {"two-photon": two_photon,
                "distinguishable": two_distinguishable,
                "one-photon": one_photon, "coherent": coherent}


def bins(nodes, total):
    """The graph-level statistic: the multiset over vertices of per-node
    count distributions, sorted outcome by outcome, and the total-count
    distribution, keyed for the ladder's :func:`separation`."""
    ranked = np.sort(nodes, axis=0)
    result = {("node", k, c): ranked[k, c]
              for k in range(len(nodes)) for c in (1, 2)}
    result.update({("total", c): total[c] for c in (1, 2)})
    return result


def drives(cmap):
    """The boundary wires that are drives: the first port of a box.
    Any other boundary wire is a dangling dart, never injected and
    discarded at every tick."""
    return [i for i, (_, port) in enumerate(cmap.boundary) if port == 0]


def slots_of(cmap, n_ticks):
    """The slot amplitudes of a map driven for ``n_ticks`` and the
    columns of its last outputs."""
    n_out, injected = len(cmap.dom), drives(cmap)
    return (amplitudes(assemble(cmap), n_out, n_ticks, injected),
            (n_ticks - 1) * n_out + np.array(injected))


def read(slots, measured, certificate="two-photon"):
    """The count distributions of one certificate and its total
    probability."""
    nodes, total = CERTIFICATES[certificate](slots, measured)
    return nodes, total, float(total.sum())


def statistics(cmap, n_ticks, certificate="two-photon"):
    """The count distributions of the map driven for ``n_ticks`` and
    read on its last drives, and the total probability."""
    return read(*slots_of(cmap, n_ticks), certificate)


def joint(cmap, n_ticks, first, second, joint_of=bosons):
    """The joint distribution of the counts on the last outputs for two
    photons injected at the slots ``first`` and ``second``, as an array
    indexed by the count of every output, for grounding the certificate
    in the contraction of the functor image."""
    slots, measured = slots_of(cmap, n_ticks)
    probabilities = joint_of(slots[first], slots[second])
    position = {column: k for k, column in enumerate(measured)}
    result = np.zeros((3,) * len(measured))
    for i in range(probabilities.shape[0]):
        for j in range(i, probabilities.shape[1]):
            counts = [0] * len(measured)
            for wire in (i, j):
                if wire in position:
                    counts[position[wire]] += 1
            result[tuple(counts)] += probabilities[i, j] / (2 if i == j else 1)
    return result
