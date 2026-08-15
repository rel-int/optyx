"""Photonic channels for the sudoku map: cells as recurrent linear
optical circuits.

Each cell is a small interferometer: two fresh photons are injected at
every tick into an eight-mode mesh of Givens rotations whose angles are
feedforward-controlled by the measured incoming digits and by the
memory digit; all modes are photon-counted and a fixed lookup -- the
lowest occupied mode -- gives the digit the cell broadcasts to its
three constraints, its prediction and its next memory. Each constraint
routes a single photon through a mesh with one feedforward layer per
claim; the verdict of dimension two is whether the photon reaches the
accept mode, and per-verdict prepare vectors give what it writes back.

The mesh layout is the odd-even transposition network (the Clements
interferometer): any permutation of modes is realised by firing a
subset of its comparators at a right angle, so controlled mode routing
-- and with it the exact sudoku solver -- lives inside the trainable
ansatz in closed form (`solver_photonic`). The cell routes photon one
to the memory digit's mode and photon two to that mode plus four; the
constraint walks its photon through the subset automaton of the
all-different constraint, one layer per claim, reaching the accept mode
exactly when the four claims are pairwise distinct -- colliding claims
are shunted to reject modes, and eight modes hold the automaton
exactly.

Messages, memory and predictions are measured, so the network tensors
have the dimension-four legs of the classical family and contract in
the regime established by the previous experiment; the quantum part is
the two-photon interference inside each tick.
"""

from itertools import combinations

import numpy as np

N_MODES = 8
INJECT = (4, 5)                 # the two fresh photons, every tick
START = 1                       # the constraint photon's input mode
ACCEPT = 0                      # the constraint's accept mode


def mesh_layout(n_modes=N_MODES):
    """The odd-even transposition network: `n_modes` rounds of
    alternating nearest-neighbour comparators, a universal sorting
    network and the Clements interferometer layout."""
    layout = []
    for parity in range(n_modes):
        layout += [(i, i + 1)
                   for i in range(parity % 2, n_modes - 1, 2)]
    return layout


LAYOUT = mesh_layout()
N_ANGLES = len(LAYOUT)


def permutation_to_angles(permutation, layout=None):
    """Realise a mode permutation -- ``permutation[source] = target`` --
    by firing comparators of the network at a right angle: odd-even
    transposition sort of the targets."""
    layout = LAYOUT if layout is None else layout
    targets = list(permutation)
    angles = np.zeros(len(layout))
    for position, (i, j) in enumerate(layout):
        if targets[i] > targets[j]:
            targets[i], targets[j] = targets[j], targets[i]
            angles[position] = np.pi / 2
    assert targets == sorted(targets), "network too short to sort"
    return angles


def pattern_pairs(n_modes=N_MODES):
    """The two-photon counting patterns, as ordered mode pairs."""
    return [(a, b) for a in range(n_modes) for b in range(a, n_modes)]


PAIRS = pattern_pairs()
DIGIT_OF_PATTERN = np.array([a % 4 for a, b in PAIRS])


def cell_routing(digit):
    """Photon one to the digit's mode, photon two to digit plus four."""
    targets = [digit, digit + 4]
    rest = iter(m for m in range(N_MODES) if m not in targets)
    permutation = [None] * N_MODES
    permutation[INJECT[0]], permutation[INJECT[1]] = targets
    for source in range(N_MODES):
        if permutation[source] is None:
            permutation[source] = next(rest)
    return permutation


def alldiff_automaton():
    """Mode maps of the subset automaton: after reading claim ``p``,
    live states are the ``p``-subsets of the digits, and a photon whose
    claim collides is parked outside the next stage's live modes so it
    can never be mistaken for a valid state. Eight modes hold the
    automaton exactly; the accept mode finally holds the photon iff the
    four claims are pairwise distinct."""
    def subsets(size):
        return [frozenset(s) for s in combinations(range(4), size)]

    stages = [
        {frozenset(): START},
        {s: 1 + i for i, s in enumerate(subsets(1))},      # modes 1-4
        {s: 1 + i for i, s in enumerate(subsets(2))},      # modes 1-6
        {s: 1 + i for i, s in enumerate(subsets(3))},      # modes 1-4
        {frozenset(range(4)): ACCEPT},
    ]
    occupied = {START}
    transitions = np.zeros((4, 4, N_MODES), dtype=int)
    for position in range(4):
        current, following = stages[position], stages[position + 1]
        live_next = set(following.values())
        next_occupied = set(live_next)
        for digit in range(4):
            permutation = [None] * N_MODES
            for subset, mode in current.items():
                if digit not in subset:
                    permutation[mode] = following[subset | {digit}]
            parking = [m for m in range(N_MODES)
                       if m not in live_next and m != ACCEPT]
            parking += [ACCEPT] if position < 3 else []
            pool = iter(parking)
            for source in sorted(occupied):
                if permutation[source] is None:
                    target = next(pool)
                    permutation[source] = target
                    next_occupied.add(target)
            rest = iter(m for m in range(N_MODES)
                        if m not in permutation)
            for source in range(N_MODES):
                if permutation[source] is None:
                    permutation[source] = next(rest)
            assert sorted(permutation) == list(range(N_MODES))
            transitions[position, digit] = permutation
        occupied = next_occupied
    return transitions


def check_automaton(transitions):
    for claims in np.ndindex(4, 4, 4, 4):
        mode = START
        for position, digit in enumerate(claims):
            mode = transitions[position, digit][mode]
        assert (mode == ACCEPT) == (len(set(claims)) == 4), claims


def solver_photonic(echo_angles=0):
    """The exact solver as mesh angles inside the trainable ansatz."""
    cell = {
        "base": np.zeros(N_ANGLES),
        "memory_bank": np.stack([
            permutation_to_angles(cell_routing(digit))
            for digit in range(4)]),
        "echo_banks": np.zeros((3, 4, echo_angles)),
    }
    transitions = alldiff_automaton()
    check_automaton(transitions)
    constraint = {
        "base": np.zeros(N_ANGLES),
        "claim_banks": np.stack([
            [permutation_to_angles(transitions[position, digit])
             for digit in range(4)]
            for position in range(4)]),
    }
    writes = np.zeros((4, 2, 4))
    writes[:, 0, :] = 0.5                # verdict ok: uniform echo
    return cell, constraint, writes


def parameter_count_photonic(echo_angles):
    return int(
        N_ANGLES + 4 * N_ANGLES + 3 * 4 * echo_angles      # cell
        + N_ANGLES + 4 * 4 * N_ANGLES                      # constraint
        + 4 * 2 * 4)                                       # writes


def init_photonic(echo_angles, seed, mode="random", noise=0.05,
                  scale=0.2):
    random = np.random.default_rng(seed)

    def random_cell():
        return {
            "base": random.normal(0, scale, N_ANGLES),
            "memory_bank": random.normal(0, scale, (4, N_ANGLES)),
            "echo_banks": random.normal(0, scale, (3, 4, echo_angles)),
        }

    def random_constraint():
        return {
            "base": random.normal(0, scale, N_ANGLES),
            "claim_banks": random.normal(0, scale, (4, 4, N_ANGLES)),
        }

    if mode == "random":
        return (random_cell(), random_constraint(),
                random.normal(1, 0.3, (4, 2, 4)))
    cell, constraint, writes = solver_photonic(echo_angles)
    cell = {key: value + noise * random.normal(size=np.shape(value))
            for key, value in cell.items()}
    if mode == "solver":
        constraint = {
            key: value + noise * random.normal(size=np.shape(value))
            for key, value in constraint.items()}
        writes = writes + noise * np.abs(
            random.normal(size=writes.shape))
        return cell, constraint, writes
    return cell, random_constraint(), random.normal(1, 0.3, (4, 2, 4))


def make_photonic_tensors_fn(config):
    """The network tensors of the measured-memory photonic family, in
    the dense classical-family format of ``contraction.py``."""
    import jax
    import jax.numpy as jnp

    echo_angles = config.get("echo_angles", 0)
    pairs = np.asarray(PAIRS)
    same = jnp.asarray(pairs[:, 0] == pairs[:, 1])
    lower, upper = jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    digit_onehot = jnp.asarray(np.eye(4)[DIGIT_OF_PATTERN])

    def mesh(angles, unitary=None):
        unitary = jnp.eye(N_MODES) if unitary is None else unitary
        for position in range(angles.shape[-1]):
            i, j = LAYOUT[position]
            cos = jnp.cos(angles[position])
            sin = jnp.sin(angles[position])
            row_i, row_j = unitary[i], unitary[j]
            unitary = unitary.at[i].set(cos * row_i - sin * row_j)
            unitary = unitary.at[j].set(sin * row_i + cos * row_j)
        return unitary

    def cell_distribution(cell, memory, echoes):
        unitary = mesh(cell["memory_bank"][memory])
        for port in range(3):
            if echo_angles:
                unitary = mesh(
                    cell["echo_banks"][port][echoes[port]], unitary)
        unitary = mesh(cell["base"], unitary)
        one, two = unitary[:, INJECT[0]], unitary[:, INJECT[1]]
        amp = one[lower] * two[upper] + one[upper] * two[lower]
        amp = jnp.where(same, amp / 2 ** .5, amp)
        return digit_onehot.T @ (amp ** 2)

    def constraint_distribution(constraint, claims):
        unitary = mesh(constraint["base"])
        for position in range(4):
            unitary = mesh(
                constraint["claim_banks"][position][claims[position]],
                unitary)
        ok = unitary[ACCEPT, START] ** 2
        return jnp.stack([ok, 1 - ok])

    combos = jnp.asarray(np.stack(np.meshgrid(
        *[np.arange(4)] * 4, indexing="ij"), axis=-1).reshape(-1, 4))

    def tensors_fn(params):
        cell, constraint, writes = params
        q = jax.vmap(
            lambda combo: cell_distribution(cell, combo[0], combo[1:])
        )(combos).reshape(4, 4, 4, 4, 4)   # (mu, e1, e2, e3, d)
        q = jnp.moveaxis(q, 0, 3)          # (e1, e2, e3, mu, d)
        eye = jnp.eye(4)
        cell_tensor = jnp.einsum(
            "abcmd,dw,dx,dy,dn,dp->abcmwxynp",
            q, eye, eye, eye, eye, eye)
        read = jax.vmap(
            lambda combo: constraint_distribution(constraint, combo)
        )(combos).reshape(4, 4, 4, 4, 2)
        return {
            "cell": cell_tensor,
            "constraint_read": read,
            "constraint_write": jnp.square(writes),
        }
    return tensors_fn
