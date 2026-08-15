"""The purely quantum and mixed photonic families.

``photonic-pure`` -- only a quantum loop: every wire between boxes is a
coherent optical mode. A cell is a seven-mode interferometer with fixed
trainable angles: three message modes pass through to its constraints,
two memory modes loop back to itself, and two injection modes receive
one photon each at every tick and are photon-counted on the way out as
the prediction -- the only measurement in the network. A constraint is
a passive four-mode interferometer on its port modes. No classical
feedforward anywhere: what recurs is light.

``photonic-both`` -- classical and quantum loops together: messages are
measured and feed forward into the mesh angles as in the measured
family, while a two-mode memory register stays coherent between ticks
and interferes with the two photons injected at each tick; the counting
pattern of the six work modes is looked up as the broadcast digit.

Amplitudes are collision-free boson amplitudes, ``<y|U|x> = per(U[y,x])``
for occupancy vectors with at most one photon per mode, computed with
Ryser's formula -- differentiable, exact in the collision-free sector,
bunched patterns truncated (the dual-rail convention of linear optical
computing).
"""

from itertools import combinations

import numpy as np

import photonic as ph


def occupancies(n_modes, count):
    return list(combinations(range(n_modes), count))


def make_permanents(n):
    """Batched differentiable permanents of (batch, n, n) matrices by
    Ryser's formula."""
    import jax.numpy as jnp

    if n == 0:
        return lambda batch: jnp.ones(batch.shape[0])
    masks = np.array(
        [[bool(subset >> j & 1) for j in range(n)]
         for subset in range(1, 2 ** n)], dtype=float)
    signs = np.array([
        (-1.) ** (n - int(masks[index].sum()))
        for index in range(2 ** n - 1)])
    masks, signs = jnp.asarray(masks), jnp.asarray(signs)

    def permanents(batch):
        row_sums = jnp.einsum("kij,sj->kis", batch, masks)
        return jnp.prod(row_sums, axis=1) @ signs
    return permanents


def make_amplitudes(n_modes):
    """All collision-free amplitudes of a mode unitary, as a dense
    table over occupancy bitmasks: ``table[out_bits, in_bits]``."""
    import jax.numpy as jnp

    per_sector = []
    for n in range(1, n_modes + 1):
        sector = occupancies(n_modes, n)
        rows = np.array([[list(y) for y in sector]] * len(sector))
        cols = np.array([[list(x)] * len(sector) for x in sector])
        bits = np.array([sum(1 << m for m in modes)
                         for modes in sector])
        out_bits = np.tile(bits, len(sector))
        in_bits = np.repeat(bits, len(sector))
        per_sector.append((
            jnp.asarray(cols.reshape(-1, n)),
            jnp.asarray(rows.reshape(-1, n)),
            jnp.asarray(in_bits), jnp.asarray(out_bits),
            make_permanents(n)))

    def amplitudes(unitary):
        table = jnp.zeros((2 ** n_modes, 2 ** n_modes))
        table = table.at[0, 0].set(1.)
        for cols, rows, in_bits, out_bits, permanents in per_sector:
            subs = unitary[rows[:, :, None], cols[:, None, :]]
            table = table.at[out_bits, in_bits].set(permanents(subs))
        return table
    return amplitudes


def mesh_unitary(angles, layout, n_modes):
    """A (possibly multi-sweep) Givens mesh, differentiable."""
    import jax.numpy as jnp

    unitary = jnp.eye(n_modes)
    flat = angles.reshape(-1, len(layout))
    for sweep in range(flat.shape[0]):
        for position, (i, j) in enumerate(layout):
            cos = jnp.cos(flat[sweep, position])
            sin = jnp.sin(flat[sweep, position])
            row_i, row_j = unitary[i], unitary[j]
            unitary = unitary.at[i].set(cos * row_i - sin * row_j)
            unitary = unitary.at[j].set(sin * row_i + cos * row_j)
    return unitary


# ---------------------------------------------------------------------------
# photonic-pure: only a quantum loop
# ---------------------------------------------------------------------------

PURE_CELL_MODES = 7          # 3 messages, 2 memory, 2 inject/predict
PURE_CONS_MODES = 5          # 4 ports and one injected, traced ancilla
CELL_LAYOUT = ph.mesh_layout(PURE_CELL_MODES)
CONS_LAYOUT = ph.mesh_layout(PURE_CONS_MODES)


def parameter_count_pure(sweeps=1):
    return sweeps * (len(CELL_LAYOUT) + len(CONS_LAYOUT)) + 16


def init_pure(sweeps, seed, scale=0.4):
    """Mesh angles for cells and constraints, and the local classical
    post-processing of the prediction counters, started near the
    identity lookup."""
    random = np.random.default_rng(seed)
    return (random.normal(0, scale, (sweeps, len(CELL_LAYOUT))),
            random.normal(0, scale, (sweeps, len(CONS_LAYOUT))),
            np.eye(4) + 0.1 * np.abs(random.normal(size=(4, 4))))


def make_pure_tensors_fn(config):
    import jax.numpy as jnp

    n_photons = config.get("n_photons", 2)
    cell_amplitudes = make_amplitudes(PURE_CELL_MODES)
    cons_amplitudes = make_amplitudes(PURE_CONS_MODES)
    inject = (1 << 5) | (1 << 6) if n_photons == 2 else (1 << 5)

    # Cell gather maps: network legs (p1, p2, p3, M | o1, o2, o3, N, d)
    # with p, o doubled ports, M, N doubled two-mode memories, d the
    # measured prediction pattern; each doubled leg value is
    # 2 * copy_one_bit + copy_two_bit per port, 4 * copy_one + copy_two
    # for the memory pairs.
    grid = np.indices((4, 4, 4, 16, 4, 4, 4, 16, 4))
    p_in = grid[0:3]
    mem_in = grid[3]
    p_out = grid[4:7]
    mem_out = grid[7]
    pred = grid[8]

    def msg_bits(ports, copy):
        bit = [(port >> (1 - copy)) & 1 for port in ports]
        return bit[0] | (bit[1] << 1) | (bit[2] << 2)

    def mem_bits(memory, copy):
        return (memory >> (2 * (1 - copy))) & 3

    def cell_x(copy):
        return (msg_bits(p_in, copy) | (mem_bits(mem_in, copy) << 3)
                | inject)

    def cell_y(copy):
        return (msg_bits(p_out, copy) | (mem_bits(mem_out, copy) << 3)
                | (pred << 5))

    cell_gather = [(jnp.asarray(cell_y(copy)), jnp.asarray(cell_x(copy)))
                   for copy in (0, 1)]

    grid4 = np.indices((4, 4, 4, 4, 4, 4, 4, 4))
    c_in, c_out = grid4[0:4], grid4[4:8]

    def cons_bits(ports, copy):
        bits = [(port >> (1 - copy)) & 1 for port in ports]
        return (bits[0] | (bits[1] << 1) | (bits[2] << 2)
                | (bits[3] << 3))

    cons_inject = 1 << 4                 # the constraint's ancilla photon
    cons_gather = [
        [(jnp.asarray(cons_bits(c_out, copy) | (ancilla << 4)),
          jnp.asarray(cons_bits(c_in, copy) | cons_inject))
         for copy in (0, 1)] for ancilla in (0, 1)]

    def tensors_fn(params):
        cell_angles, cons_angles, post = params
        cell_table = cell_amplitudes(
            mesh_unitary(cell_angles, CELL_LAYOUT, PURE_CELL_MODES))
        cons_table = cons_amplitudes(
            mesh_unitary(cons_angles, CONS_LAYOUT, PURE_CONS_MODES))
        cell = (cell_table[cell_gather[0][0], cell_gather[0][1]]
                * cell_table[cell_gather[1][0], cell_gather[1][1]])
        cell = jnp.einsum("...o,od->...d", cell, jnp.square(post))
        constraint = sum(
            cons_table[gather[0][0], gather[0][1]]
            * cons_table[gather[1][0], gather[1][1]]
            for gather in cons_gather)   # traced ancilla
        return {"cell": cell, "constraint": constraint}
    return tensors_fn


# ---------------------------------------------------------------------------
# photonic-both: classical and quantum loops together
# ---------------------------------------------------------------------------

BOTH_MODES = 8               # 6 work modes, 2 coherent memory modes
BOTH_LAYOUT = ph.mesh_layout(BOTH_MODES)
BOTH_INJECT = (1 << 4) | (1 << 5)
MEMORY_SHIFT = 6

DIGIT_OF_WORK = np.array([
    (min(m for m in range(6) if (bits >> m) & 1) % 4) if bits else 0
    for bits in range(64)])


def parameter_count_both(echo_angles):
    cell = len(BOTH_LAYOUT) + 3 * 4 * echo_angles
    constraint = ph.N_ANGLES + 4 * 4 * ph.N_ANGLES
    return int(cell + constraint + 4 * 2 * 4)


def init_both(echo_angles, seed, scale=0.2):
    random = np.random.default_rng(seed)
    cell = {
        "base": random.normal(0, scale, len(BOTH_LAYOUT)),
        "echo_banks": random.normal(0, scale, (3, 4, echo_angles)),
    }
    constraint = {
        "base": random.normal(0, scale, ph.N_ANGLES),
        "claim_banks": random.normal(0, scale, (4, 4, ph.N_ANGLES)),
    }
    return cell, constraint, random.normal(1, 0.3, (4, 2, 4))


def make_both_tensors_fn(config):
    """Cell: coherent two-mode memory interfering with two injected
    photons on six measured work modes, mesh angles feedforward
    controlled by the incoming digits; the work pattern's lowest
    occupied mode gives the broadcast digit. Constraint: the verdict
    mesh of the measured family."""
    import jax
    import jax.numpy as jnp

    echo_angles = config.get("echo_angles", 28)
    amplitudes = make_amplitudes(BOTH_MODES)
    digit_onehot = jnp.asarray(np.eye(4)[DIGIT_OF_WORK])

    def apply_mesh(angles, layout, unitary):
        for position in range(angles.shape[-1]):
            i, j = layout[position]
            cos = jnp.cos(angles[position])
            sin = jnp.sin(angles[position])
            row_i, row_j = unitary[i], unitary[j]
            unitary = unitary.at[i].set(cos * row_i - sin * row_j)
            unitary = unitary.at[j].set(sin * row_i + cos * row_j)
        return unitary

    combos3 = jnp.asarray(np.stack(np.meshgrid(
        *[np.arange(4)] * 3, indexing="ij"), axis=-1).reshape(-1, 3))
    combos4 = jnp.asarray(np.stack(np.meshgrid(
        *[np.arange(4)] * 4, indexing="ij"), axis=-1).reshape(-1, 4))

    work = np.arange(64)
    mem = np.arange(4)
    y_bits = jnp.asarray(
        work[:, None] | (mem[None, :] << MEMORY_SHIFT))   # (w, nu)
    x_bits = jnp.asarray(BOTH_INJECT | (mem << MEMORY_SHIFT))

    def cell_amplitude(cell, echoes):
        unitary = jnp.eye(BOTH_MODES)
        for port in range(3):
            if echo_angles:
                unitary = apply_mesh(
                    cell["echo_banks"][port][echoes[port]],
                    BOTH_LAYOUT, unitary)
        unitary = apply_mesh(cell["base"], BOTH_LAYOUT, unitary)
        table = amplitudes(unitary)
        return table[y_bits[:, :, None], x_bits[None, None, :]]

    def constraint_distribution(constraint, claims):
        unitary = apply_mesh(constraint["base"], ph.LAYOUT,
                             jnp.eye(ph.N_MODES))
        for position in range(4):
            unitary = apply_mesh(
                constraint["claim_banks"][position][claims[position]],
                ph.LAYOUT, unitary)
        ok = unitary[ph.ACCEPT, ph.START] ** 2
        return jnp.stack([ok, 1 - ok])

    def tensors_fn(params):
        cell, constraint, writes = params
        amp = jax.vmap(lambda combo: cell_amplitude(cell, combo))(
            combos3)                                # (64e, 64w, nu, mu)
        doubled = jnp.einsum("ewnm,ewNM->ewnNmM", amp, amp)
        q = jnp.einsum("ewnNmM,wd->ednNmM", doubled, digit_onehot)
        q = q.reshape(64, 4, 16, 16).reshape(4, 4, 4, 4, 16, 16)
        eye = jnp.eye(4)
        cell_tensor = jnp.einsum(
            "abcdNM,dw,dx,dy->abcMwxyNd", q, eye, eye, eye)
        read = jax.vmap(
            lambda claims: constraint_distribution(constraint, claims)
        )(combos4).reshape(4, 4, 4, 4, 2)
        return {
            "cell": cell_tensor,
            "constraint_read": read,
            "constraint_write": jnp.square(writes),
        }
    return tensors_fn
