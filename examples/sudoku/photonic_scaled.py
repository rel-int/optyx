"""The scaled purely quantum photonic family.

Scales the pure family of `photonic_quantum` in directions that do not
touch the contraction width:

* **Internal vacuum ancillas**: extra modes inside each box, starting
  empty and traced out at the end -- photons can scatter into them and
  be lost, giving structured non-unitary channels and quadratically
  more mesh parameters at zero contraction cost.
* **Complex meshes**: each comparator of the Clements layout carries a
  rotation and a phase, doubling the parameters; the doubled tensors
  conjugate the second copy, so the network tensors stay real.
* **Fixed photon loss on links**: a pure-loss channel of fixed
  transmittivity ``eta >= 0.9`` folded into every coherent write leg;
  no new parameters, no new network terms.

Cell modes: three messages, two memory, two inject/predict, then
vacuum ancillas. Constraint modes: four ports, one injected ancilla,
then vacuum ancillas. Amplitudes are collision-free (cutoff one) and
computed per input column, so eleven-mode cells stay cheap; with zero
extra ancillas, real angles and no loss this reduces exactly to the
family of `photonic_quantum.make_pure_tensors_fn`.
"""

import numpy as np

import photonic as ph
from photonic_quantum import make_permanents, occupancies


def parameter_count_scaled(config):
    cell_layout = ph.mesh_layout(config.get("cell_modes", 11))
    cons_layout = ph.mesh_layout(config.get("cons_modes", 8))
    per_angle = 2 if config.get("complex", True) else 1
    return int(config.get("sweeps", 4) * per_angle
               * (len(cell_layout) + len(cons_layout)) + 16)


def init_scaled(config, seed, scale=0.3):
    random = np.random.default_rng(seed)
    cell_layout = ph.mesh_layout(config.get("cell_modes", 11))
    cons_layout = ph.mesh_layout(config.get("cons_modes", 8))
    per_angle = 2 if config.get("complex", True) else 1
    sweeps = config.get("sweeps", 4)
    return (
        random.normal(0, scale, (sweeps, len(cell_layout), per_angle)),
        random.normal(0, scale, (sweeps, len(cons_layout), per_angle)),
        np.eye(4) + 0.1 * np.abs(random.normal(size=(4, 4))))


def loss_matrix(eta):
    """The doubled pure-loss channel on one cutoff-one mode, acting on
    a doubled leg of dimension four (basis ``2 * copy1 + copy2``)."""
    kraus = [np.array([[1., 0.], [0., eta ** 0.5]]),
             np.array([[0., (1 - eta) ** 0.5], [0., 0.]])]
    return sum(np.kron(k, k) for k in kraus)


def memory_loss_matrix(eta):
    """Independent pure loss on both memory modes, in the fused memory
    basis ``4 * m1 + m2`` with ``m = mode3 + 2 * mode4``."""
    loss = loss_matrix(eta).reshape(2, 2, 2, 2)
    return np.einsum("acik,bdjl->badcjilk", loss, loss).reshape(16, 16)


def column_scatter(n_modes, input_occupancy, out_shape, out_index):
    """Static data for one input column: mode lists for the permanent
    submatrices and flat scatter indices into the dense amplitude
    array, for every same-photon-number collision-free output."""
    n_photons = sum(input_occupancy)
    cols = np.array([m for m, bit in enumerate(input_occupancy)
                     if bit], dtype=int)
    outputs = occupancies(n_modes, n_photons)
    rows = np.array([list(y) for y in outputs], dtype=int)
    flat = np.array([
        np.ravel_multi_index(out_index(frozenset(y)), out_shape)
        for y in outputs], dtype=int)
    return cols, rows, flat, n_photons


def make_box_amplitudes(n_modes, input_specs, out_shape, out_index):
    """A jax function mapping a mode unitary to the dense single-copy
    amplitude array ``A[input combo, flattened output legs]``."""
    import jax.numpy as jnp

    columns = []
    for key, occupancy in input_specs:
        cols, rows, flat, n_photons = column_scatter(
            n_modes, occupancy, out_shape, out_index)
        columns.append((key, jnp.asarray(cols), jnp.asarray(rows),
                        jnp.asarray(flat), make_permanents(n_photons),
                        n_photons))
    flat_size = int(np.prod(out_shape))

    def amplitudes(unitary):
        result = jnp.zeros((len(columns), flat_size),
                           dtype=unitary.dtype)
        for slot, (key, cols, rows, flat, permanents, n) in enumerate(
                columns):
            if n == 0:
                continue
            subs = unitary[rows[:, :, None], cols[None, None, :]]
            result = result.at[slot, flat].set(permanents(subs))
        return result
    return amplitudes


def complex_mesh(angles, layout, n_modes):
    import jax.numpy as jnp

    unitary = jnp.eye(n_modes, dtype=complex)
    flat = angles.reshape(-1, len(layout), angles.shape[-1])
    for sweep in range(flat.shape[0]):
        for position, (i, j) in enumerate(layout):
            theta = flat[sweep, position, 0]
            cos, sin = jnp.cos(theta), jnp.sin(theta)
            phase = (jnp.exp(1j * flat[sweep, position, 1])
                     if angles.shape[-1] == 2 else 1.)
            row_i, row_j = unitary[i], unitary[j]
            unitary = unitary.at[i].set(cos * row_i
                                        - sin * phase * row_j)
            unitary = unitary.at[j].set(
                sin * jnp.conj(phase) * row_i + cos * row_j)
    return unitary


def make_scaled_tensors_fn(config):
    import jax.numpy as jnp

    cell_modes = config.get("cell_modes", 11)
    cons_modes = config.get("cons_modes", 8)
    eta = config.get("eta", 1.0)
    assert eta >= 0.9
    n_cell_anc = cell_modes - 7
    n_cons_anc = cons_modes - 5

    # --- cell: single-copy legs (w1, w2, w3, mem, pred, anc) ---------
    cell_shape = (2, 2, 2, 4, 4, 2 ** n_cell_anc)

    def cell_out_index(modes):
        w = [int(m in modes) for m in range(3)]
        mem = (3 in modes) + 2 * (4 in modes)
        pred = (5 in modes) + 2 * (6 in modes)
        anc = sum((1 << k) for k in range(n_cell_anc)
                  if 7 + k in modes)
        return (w[0], w[1], w[2], mem, pred, anc)

    cell_inputs = []
    for m1 in range(2):
        for m2 in range(2):
            for m3 in range(2):
                for mem in range(4):
                    occupancy = ((m1, m2, m3, mem & 1, mem >> 1, 1, 1)
                                 + (0,) * n_cell_anc)
                    cell_inputs.append((None, occupancy))
    cell_amplitudes = make_box_amplitudes(
        cell_modes, cell_inputs, cell_shape, cell_out_index)

    # --- constraint: single-copy legs (w1..w4, anc) ------------------
    cons_shape = (2, 2, 2, 2, 2 ** (n_cons_anc + 1))

    def cons_out_index(modes):
        w = [int(m in modes) for m in range(4)]
        anc = sum((1 << k) for k in range(n_cons_anc + 1)
                  if 4 + k in modes)
        return (w[0], w[1], w[2], w[3], anc)

    cons_inputs = []
    for bits in range(16):
        occupancy = (tuple((bits >> (3 - p)) & 1 for p in range(4))
                     + (1,) + (0,) * n_cons_anc)
        cons_inputs.append((None, occupancy))
    cons_amplitudes = make_box_amplitudes(
        cons_modes, cons_inputs, cons_shape, cons_out_index)

    cell_layout = ph.mesh_layout(cell_modes)
    cons_layout = ph.mesh_layout(cons_modes)
    loss = jnp.asarray(loss_matrix(eta))
    memory_loss = jnp.asarray(memory_loss_matrix(eta))

    def tensors_fn(params):
        cell_angles, cons_angles, post = params
        cell_unitary = complex_mesh(cell_angles, cell_layout,
                                    cell_modes)
        cons_unitary = complex_mesh(cons_angles, cons_layout,
                                    cons_modes)
        amp = cell_amplitudes(cell_unitary).reshape(
            (2, 2, 2, 4) + cell_shape)
        # doubled: trace ancillas (shared k), keep prediction (shared
        # p), fuse per-wire copies.
        cell = jnp.einsum(
            "abcudefvpk,ABCUDEFVpk->aAbBcCuUdDeEfFvVp",
            amp, jnp.conj(amp)).real
        cell = cell.reshape(4, 4, 4, 16, 4, 4, 4, 16, 4)
        cell = jnp.einsum("...o,od->...d", cell, jnp.square(post))
        camp = cons_amplitudes(cons_unitary).reshape(
            (2, 2, 2, 2) + cons_shape)
        constraint = jnp.einsum(
            "abcdwxyzk,ABCDWXYZk->aAbBcCdDwWxXyYzZ",
            camp, jnp.conj(camp)).real
        constraint = constraint.reshape(4, 4, 4, 4, 4, 4, 4, 4)
        if eta < 1.0:
            for write_leg in (4, 5, 6):
                cell = jnp.moveaxis(jnp.tensordot(
                    cell, loss, axes=[[write_leg], [1]]),
                    -1, write_leg)
            cell = jnp.moveaxis(jnp.tensordot(
                cell, memory_loss, axes=[[7], [1]]), -1, 7)
            for write_leg in (4, 5, 6, 7):
                constraint = jnp.moveaxis(jnp.tensordot(
                    constraint, loss, axes=[[write_leg], [1]]),
                    -1, write_leg)
        return {"cell": cell, "constraint": constraint}
    return tensors_fn
