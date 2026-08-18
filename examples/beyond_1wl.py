"""
Proposal B of the interaction module: photonic interference beyond 1-WL.

Generated pairs of non-isomorphic graphs that classical message passing
provably cannot tell apart (Xu et al., ICLR 2019; Morris et al., AAAI
2019), with the three models of the comparison:

1. a vanilla GNN, the textbook isotropic message-passing network;
2. a MapNN from ``discopy.neural``, the same graphs interpreted as
   port-addressed interaction maps (discopy/discopy#585);
3. an :class:`optyx.interaction.CMap` whose nodes are interferometers
   with coherent memory and coherent messages, read out through the
   photon statistics of its prediction wires.

A graph is a tuple of sorted neighbour tuples, the format of
``discopy.neural.signature.from_relation``. Every readout below is
invariant under graph isomorphism, so a nonzero separation on a pair is
a certificate that the model computes a graph invariant beyond 1-WL.
"""

from itertools import combinations, permutations, product

import numpy as np

from optyx.channel import qmode, Diagram
from optyx.interaction import Box, CMap
from optyx.photonic import Create, Gate, TBS


def cycle(n_vertices: int) -> tuple:
    """The cycle graph on ``n_vertices``."""
    return tuple(
        tuple(sorted(((i - 1) % n_vertices, (i + 1) % n_vertices)))
        for i in range(n_vertices))


def path(n_vertices: int) -> tuple:
    """The path graph on ``n_vertices``."""
    return tuple(
        tuple(j for j in (i - 1, i + 1) if 0 <= j < n_vertices)
        for i in range(n_vertices))


def disjoint_union(left: tuple, right: tuple) -> tuple:
    """The disjoint union of two graphs, shifting the right one."""
    return left + tuple(
        tuple(j + len(left) for j in nbrs) for nbrs in right)


def fused_cycles(girth: int, bridge: bool) -> tuple:
    """
    Two ``girth``-cycles joined by an edge when ``bridge``, sharing an
    edge otherwise: the carbon skeletons of bicyclopentyl (two pentagons
    and a bridge) and decalin (two hexagons and a shared edge), the
    1-WL-equivalent pair of Sato's survey (arXiv:2003.04078). Both have
    ten vertices, eleven edges, two adjacent vertices of degree three.
    """
    inner = girth - 1 if bridge else girth - 2
    edges = [(0, 1)]
    for first in (0, 1):
        ring = [first] + list(range(
            2 + first * inner, 2 + (first + 1) * inner)) + [
            first if bridge else 1 - first]
        edges += list(zip(ring, ring[1:]))
    return tuple(
        tuple(sorted(v for edge in edges for u, v in (edge, edge[::-1])
                     if u == vertex))
        for vertex in range(2 + 2 * inner))


def rook() -> tuple:
    """The 4x4 rook's graph: same row or same column on a 4x4 board."""
    return tuple(
        tuple(sorted(4 * a + b for a, b in product(range(4), range(4))
                     if (a == i) != (b == j)))
        for i, j in product(range(4), range(4)))


def shrikhande() -> tuple:
    """
    The Shrikhande graph, the Cayley graph of ``Z_4 x Z_4`` with
    connection set ``{(1, 0), (0, 1), (1, 1)}`` and inverses: the same
    strongly regular parameters srg(16, 6, 2, 2) as :func:`rook`, so the
    two are not distinguished even by 3-WL, yet they are not isomorphic
    -- the neighbourhood of a vertex is ``2C3`` in the rook's graph and
    ``C6`` in the Shrikhande graph, the smallest pair below.
    """
    steps = ((1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3))
    return tuple(
        tuple(sorted(4 * ((i + a) % 4) + (j + b) % 4 for a, b in steps))
        for i, j in product(range(4), range(4)))


#: The three 1-WL-equivalent pairs of the proposal, plus the smallest
#: generated pair: two triangles against a hexagon.
PAIRS = {
    "2C3 vs C6": (disjoint_union(cycle(3), cycle(3)), cycle(6)),
    "2C6 vs C12": (disjoint_union(cycle(6), cycle(6)), cycle(12)),
    "decalin vs bicyclopentyl": (fused_cycles(6, bridge=False),
                                 fused_cycles(5, bridge=True)),
    "rook vs shrikhande": (rook(), shrikhande()),
}

#: A pair that 1-WL does distinguish, the positive control: a model
#: whose separation is zero on it is broken, not bounded.
CONTROL = (cycle(6), path(6))


def wl_colours(graph: tuple) -> list:
    """
    The stable colouring of 1-WL colour refinement, as a sorted list:
    the invariant that bounds message passing with anonymous inputs and
    permutation-invariant aggregation (Xu et al., ICLR 2019).
    """
    colours = [0] * len(graph)
    for _ in range(len(graph)):
        keys = [(colours[v], tuple(sorted(colours[u] for u in nbrs)))
                for v, nbrs in enumerate(graph)]
        table = {key: colour for colour, key in enumerate(sorted(set(keys)))}
        refined = [table[key] for key in keys]
        if refined == colours:
            break
        colours = refined
    return sorted(colours)


def wl_equivalent(left: tuple, right: tuple) -> bool:
    """Whether 1-WL fails to distinguish the two graphs."""
    return wl_colours(left) == wl_colours(right)


def coupler(degree: int, theta: float = 0.4, phi: float = 0.7,
            chi: float = 1.1, drive: float = 0.8) -> np.ndarray:
    """
    A port-symmetric interferometer on ``1 + degree + 1`` modes -- the
    drive, the message ports, the memory: the drive and the memory couple
    to every port with the same amplitude and to each other, and the
    ports mix symmetrically among themselves, so the unitary is invariant
    under permutations of the ports and the readout below is a graph
    invariant. Like any interferometer it decomposes into beam splitters
    and phase shifters (Reck et al., PRL 73, 58 (1994)).
    """
    modes = degree + 2
    ports, memory = range(1, degree + 1), modes - 1
    coupling = np.zeros((modes, modes), dtype=complex)
    for port in ports:
        coupling[port, memory] = theta * np.exp(1j * phi) / degree ** .5
        coupling[0, port] = 0.25 * np.exp(1j * 0.5) / degree ** .5
        for other in ports:
            if port != other:
                coupling[port, other] = chi / degree
    coupling[0, memory] = drive * np.exp(1j * 0.3)
    coupling[memory, memory] = 0.9
    coupling += np.triu(coupling, 1).conj().T
    energies, modes_ = np.linalg.eigh(coupling)
    return modes_ @ np.diag(np.exp(1j * energies)) @ modes_.conj().T


def vertex_box(degree: int, tap: float = 0.3) -> Box:
    """
    The interferometer cell of a degree-``degree`` vertex: a ``qmode``
    drive port read from the environment at every tick -- where the
    photons come in -- one ``qmode`` message port per incident edge, a
    ``qmode`` coherent memory mixed with all of them by :func:`coupler`,
    and a beam-splitter tap from the memory into a fresh vacuum mode as
    the prediction: the input and output couplers of a little cavity.
    """
    modes = degree + 2
    mix = Gate(coupler(degree), modes, modes, f"mix{degree}")
    channel = mix >> Diagram.id(qmode ** (modes - 1)) @ (
        qmode @ Create(0) >> TBS(tap))
    return Box(f"cell{degree}", qmode, qmode ** degree, channel,
               memory=qmode, prediction=qmode)


def stateful_channel(box: Box, input_state: Diagram = None) -> Diagram:
    """
    The interpretation of one box of the map as a stateful channel: its
    local channel with the internal memory fed back to itself with a
    one-tick delay, a recurrent channel from its ports to its ports and
    prediction. An ``input_state`` prepares the drive inside the loop,
    afresh at every tick, as in :meth:`optyx.interaction.CMap.fix` --
    one photon input at every time step is simply ``Create(1)`` in the
    diagram with the feedback loop.
    """
    channel, dom = box.channel, box.ports
    if input_state is not None:
        channel = input_state @ Diagram.id(
            box.cod @ box.memory) >> channel
        dom = box.cod
    move_memory_last = Diagram.id(box.ports) @ Diagram.swap(
        box.memory, box.prediction)
    return (channel >> move_memory_last).feedback(
        dom=dom, cod=box.ports @ box.prediction, mem=box.memory)


def graph_cmap(graph: tuple, tap: float = 0.3) -> CMap:
    """
    The combinatorial map of a graph: one :func:`vertex_box` per vertex
    and one edge pairing the matching message ports of its endpoints.
    The drive ports stay unpaired, so the boundary of the map is one
    ``qmode`` per vertex, read from the environment at every tick -- the
    mechanism of ``input_state`` in :meth:`optyx.interaction.CMap.fix`.
    """
    return CMap(
        [vertex_box(len(nbrs), tap) for nbrs in graph],
        [((u, nbrs.index(v) + 1), (v, graph[v].index(u) + 1))
         for u, nbrs in enumerate(graph) for v in nbrs if u < v])


def step_amplitudes(cmap: CMap) -> np.ndarray:
    """
    The one-photon amplitudes of one tick of the map: the path matrix of
    :attr:`CMap.step` restricted to its physical inputs -- the drive
    ports then the memory, with rows as inputs and columns as outputs,
    the convention of :class:`optyx.core.path.Matrix`. The columns are
    the reflections at the drive ports, the predictions, the memory.
    """
    amplitudes = np.asarray(cmap.step.to_path().array, dtype=complex)
    return amplitudes[:len(cmap.dom) + len(cmap.memory), :]


def transfer(amplitudes: np.ndarray, n_vertices: int, source: int,
             n_ticks: int, decohered: bool = False) -> np.ndarray:
    """
    The single-particle transfer matrix of the driven protocol: entry
    ``(output, tick)`` is the amplitude on one of the ``2 n_vertices``
    output wires of one tick -- the drive reflections then the
    predictions -- for the photon injected at the source's drive port at
    that tick, starting from vacuum memory. With ``decohered`` the
    photon is measured at every tick, so the entries are probabilities
    propagated by the squared moduli instead.
    """
    amplitudes = np.abs(amplitudes) ** 2 if decohered else amplitudes
    n_out = 2 * n_vertices
    n_memory = amplitudes.shape[0] - n_vertices
    result = np.zeros((n_ticks * n_out, n_ticks),
                      dtype=float if decohered else complex)
    for injected in range(n_ticks):
        state = np.zeros(n_memory, dtype=result.dtype)
        for tick in range(injected, n_ticks):
            occupation = np.zeros(amplitudes.shape[0], dtype=result.dtype)
            occupation[source] = tick == injected
            occupation[n_vertices:] = state
            output = occupation @ amplitudes
            result[tick * n_out:(tick + 1) * n_out, injected] = \
                output[:n_out]
            state = output[n_out:]
    return result


def photon_statistics(amplitudes: np.ndarray, n_ticks: int) -> tuple:
    """
    The photon statistics of a transfer matrix, aggregated over vertices
    so that both are graph invariants: the mean photon number leaving at
    each tick, and the two-photon coincidences between each pair of
    ticks. With independent single photons on the inputs of a passive
    network, the coincidence between two output wires is a sum of
    two-by-two permanents of the transfer matrix -- Hong-Ou-Mandel
    interference -- computed here through its Green's function.
    """
    n_out = amplitudes.shape[0] // n_ticks
    mean = (np.abs(amplitudes) ** 2).sum(axis=1)
    green = amplitudes @ amplitudes.conj().T
    squared = np.abs(amplitudes) ** 2
    diagonal = squared @ squared.T
    coincidence = (np.outer(mean, mean) - diagonal
                   + np.abs(green) ** 2 - diagonal)
    return (mean.reshape(n_ticks, n_out).sum(axis=1),
            coincidence.reshape(
                n_ticks, n_out, n_ticks, n_out).sum(axis=(1, 3)))


def third_order_statistic(matrix: np.ndarray,
                          n_ticks: int) -> np.ndarray:
    """
    An aggregated third-order photon statistic of a transfer matrix:
    three-photon coincidences summed over the vertices of each tick
    triple, from the three-by-three permanents of the transfer matrix
    over every unordered triple of injection ticks -- one order past
    the pairwise coincidences of :func:`photon_statistics`, where the
    statistics of non-interacting photons keep going.
    """
    n_out = matrix.shape[0] // n_ticks
    total = np.zeros((n_ticks, n_ticks, n_ticks))
    for columns in combinations(matrix.T, 3):
        permanents = sum(
            np.einsum("x,y,z->xyz", *ordering)
            for ordering in permutations(columns))
        total += (np.abs(permanents) ** 2).reshape(
            n_ticks, n_out, n_ticks, n_out, n_ticks, n_out
        ).sum(axis=(1, 3, 5))
    return total


def profile(graph: tuple, n_ticks: int, tap: float = 0.3,
            decohered: bool = False) -> np.ndarray:
    """
    The response profile of a graph under the driven protocol: for each
    source vertex, the mean-photon-number curve followed by the upper
    triangle of the tick-by-tick coincidences, as a sorted array over
    sources -- a multiset over vertices, hence a graph invariant. The
    decohered ablation propagates probabilities instead of amplitudes;
    its particles are independent, so only the mean curve carries
    information and the coincidence entries are set to zero.
    """
    step = step_amplitudes(graph_cmap(graph, tap))
    rows = []
    for source in range(len(graph)):
        matrix = transfer(step, len(graph), source, n_ticks, decohered)
        if decohered:
            mean = matrix.sum(axis=1).reshape(n_ticks, -1).sum(axis=1)
            coincidence = np.zeros((n_ticks, n_ticks))
        else:
            mean, coincidence = photon_statistics(matrix, n_ticks)
        rows.append(np.concatenate([
            mean, coincidence[np.triu_indices(n_ticks)]]))
    rows = np.array(rows)
    return rows[np.lexsort(np.round(rows, 9).T[::-1])]


def separation(left: tuple, right: tuple, n_ticks: int,
               **params) -> float:
    """
    The largest difference between the response profiles of two graphs:
    zero iff the readout does not distinguish them.
    """
    return float(np.abs(
        profile(left, n_ticks, **params)
        - profile(right, n_ticks, **params)).max())


def shots_to_separate(margin: float, confidence: float = 5) -> float:
    """
    The number of runs per source vertex after which a separation
    ``margin`` on a probability stands ``confidence`` standard
    deviations above shot noise, from the normal approximation of a
    Bernoulli estimate.
    """
    return (confidence / margin) ** 2


def _torch():
    # pylint: disable=import-outside-toplevel
    import torch
    return torch


def vanilla_gnn_embedding(graph: tuple, width: int = 6, rounds: int = 8,
                          seed: int = 0) -> np.ndarray:
    """
    The graph embedding of a randomly initialised vanilla GNN: one
    hidden vector per vertex from a constant initial feature, one shared
    message MLP, sum aggregation over neighbours, a shared GRU update
    and mean readout over vertices -- the textbook isotropic
    message-passing network, bounded by 1-WL.
    """
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)

    def linear(rows, cols):
        return torch.randn(
            rows, cols, generator=generator,
            dtype=torch.float64) / cols ** .5

    message, update = linear(width, width), linear(width, 2 * width)
    with torch.no_grad():
        states = torch.ones(len(graph), width, dtype=torch.float64)
        for _ in range(rounds):
            messages = torch.tanh(states @ message.T)
            pooled = torch.stack([
                sum(messages[u] for u in nbrs) for nbrs in graph])
            gates = torch.cat([states, pooled], dim=1) @ update.T
            states = torch.tanh(gates) + states * torch.sigmoid(pooled)
        return states.mean(dim=0).numpy()


def mapnn_embedding(graph: tuple, width: int = 6, rounds: int = 8,
                    seed: int = 0) -> np.ndarray:
    """
    The graph embedding of a randomly initialised MapNN from
    ``discopy.neural``: the graph becomes a closed wiring through
    ``from_relation``, one shared ``Site`` cell fills every vertex with
    sum pooling over its peer orbit, the solver iterates the compiled
    interaction and the readout averages the cell states over vertices.
    """
    torch = _torch()
    # pylint: disable=import-outside-toplevel
    from discopy.frobenius import Ty as RoleTy
    from discopy.neural import Dim, Mode, Orbit, Signature, Site, Sym
    from discopy.neural.model import MapNN
    from discopy.neural.signature import from_relation
    from discopy.neural.solver import Iterate

    peer, state = RoleTy("peer"), RoleTy("state")
    node = Signature((Orbit(peer, 2, Sym.PERM), Orbit(state, traced=True)))
    torch.manual_seed(seed)
    model = MapNN(
        ob={peer: Dim(width), state: Dim(width)},
        ar={"cell": Site(node, {peer: width, state: width},
                         {state: Mode.STATE}, hidden=2 * width, pool="sum")},
        solver=Iterate(rounds=rounds))
    diagram = from_relation(graph, node)
    with torch.no_grad():
        final = model(diagram)
        return model.read(
            diagram, final, ("cell", state)).mean(dim=1).numpy().ravel()


def embedding_separation(embed, left: tuple, right: tuple,
                         n_seeds: int = 5, **params) -> float:
    """
    The largest difference between the graph embeddings of two graphs
    over random parameter draws, for either classical model.
    """
    return float(max(
        np.abs(embed(left, seed=seed, **params)
               - embed(right, seed=seed, **params)).max()
        for seed in range(n_seeds)))
