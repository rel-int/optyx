"""Verification of the scaled purely quantum photonic family."""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

import photonic_quantum as pq
import photonic_scaled as ps

# 1. Reduction: with no extra ancillas, real angles and no loss, the
# scaled family must reproduce the pure family exactly.
config = {"cell_modes": 7, "cons_modes": 5, "sweeps": 2,
          "complex": False, "eta": 1.0}
cell_angles, cons_angles, post = pq.init_pure(2, seed=7)
pure = pq.make_pure_tensors_fn({})((cell_angles, cons_angles, post))
scaled = ps.make_scaled_tensors_fn(config)(
    (cell_angles[..., None], cons_angles[..., None], post))
for name in ("cell", "constraint"):
    diff = float(jnp.max(jnp.abs(pure[name] - scaled[name])))
    print(f"{name} equivalent: {diff < 1e-10} max diff {diff}")
    assert diff < 1e-10, name

# 2. Loss channels are trace-preserving on the doubled legs.
trace4 = np.array([1., 0., 0., 1.])
trace16 = np.eye(4).ravel()
for eta in (0.9, 0.95):
    gap4 = np.max(np.abs(trace4 @ ps.loss_matrix(eta) - trace4))
    gap16 = np.max(np.abs(trace16 @ ps.memory_loss_matrix(eta) - trace16))
    print(f"eta {eta}: trace gaps {gap4:.2e} {gap16:.2e}")
    assert gap4 < 1e-12 and gap16 < 1e-12

# 3. The big config builds finite tensors of the network shapes.
big = {"cell_modes": 11, "cons_modes": 8, "sweeps": 4,
       "complex": True, "eta": 0.95}
params = ps.init_scaled(big, seed=0)
sizes = [p.size for p in jax.tree_util.tree_leaves(params)]
print("params", sum(sizes), "count", ps.parameter_count_scaled(big))
assert sum(sizes) == ps.parameter_count_scaled(big)
tensors = ps.make_scaled_tensors_fn(big)(tuple(map(jnp.asarray, params)))
assert tensors["cell"].shape == (4, 4, 4, 16, 4, 4, 4, 16, 4)
assert tensors["constraint"].shape == (4,) * 8
for name, tensor in tensors.items():
    print(name, tensor.shape, "finite", bool(jnp.all(jnp.isfinite(tensor))),
          "max", float(jnp.max(jnp.abs(tensor))))
    assert jnp.all(jnp.isfinite(tensor))
print("PASSED")
