"""The stochastic family contains an exact solver: hand-coded cores must
decode every held-out puzzle at any tick count above one."""

import sys
import time

import numpy as np

import experiment as ex
import contraction as co


def solver_tensors(family):
    import jax.numpy as jnp

    if family == "square":
        cell, constraint_read, constraint_write = ex.solver_cores()
        return {
            "cell": co.assemble_cell([jnp.asarray(c) for c in cell]),
            "constraint_read": co.assemble_constraint_read(
                [jnp.asarray(c) for c in constraint_read]),
            "constraint_write": jnp.stack(
                [jnp.asarray(c) for c in constraint_write]),
        }, 1
    if family == "angles":
        params = tuple(map(jnp.asarray, ex.solver_angles(10, 1)))
        tensors_fn = co.make_tensors_fn(
            "quantum", {"cell_depth": 10, "cons_depth": 1, "feedback": 2})
        return tensors_fn(params), 2
    assert family == "quantum"
    cell_unitary, constraint_unitary, writes = ex.solver_quantum()
    kraus = jnp.asarray(
        cell_unitary.reshape(1024, 4, 4, 4, 4, 4)[..., 0]
    ).reshape((4,) * 9)
    cell = jnp.einsum(
        "MABCPuabc,NABCPvabc->abcuvABCMNP", kraus, kraus
    ).reshape(4, 4, 4, 16, 4, 4, 4, 16, 4)
    vkraus = jnp.asarray(
        constraint_unitary.reshape(256, 2, 256, 2)[..., 0])
    read = jnp.einsum("jka,jka->ak", vkraus, vkraus).reshape(
        4, 4, 4, 4, 2)
    return {
        "cell": cell,
        "constraint_read": read,
        "constraint_write": jnp.asarray(writes),
    }, 2


def main(ticks=2, n_puzzles=16, family="square"):
    import jax

    jax.config.update("jax_enable_x64", True)
    structure = ex.sudoku_structure()
    assert len(structure.edges) == 48
    assert len(structure.boxes) == 28
    train_cases, test_cases = ex.make_dataset()
    tensors, feedback = solver_tensors(family)
    scores = co.make_scores(
        structure, ticks,
        "quantum" if family == "angles" else family, feedback=feedback)

    def probabilities(tensors, effects):
        raw = scores(tensors, effects)
        return raw / raw.sum(axis=-1, keepdims=True)

    batched = jax.jit(jax.vmap(probabilities, in_axes=(None, 0)))
    cases = test_cases[:n_puzzles]
    effects, solutions, hidden = co.encode_cases(cases, ticks)
    started = time.perf_counter()
    p = np.asarray(batched(tensors, effects))
    first = time.perf_counter() - started
    started = time.perf_counter()
    p = np.asarray(batched(tensors, effects))
    second = time.perf_counter() - started
    guess = p.argmax(axis=-1)
    hits = (guess == solutions) & hidden
    accuracy = hits.sum() / hidden.sum()
    solved = int((hits.sum(1) == hidden.sum(1)).sum())
    print(f"family={family} ticks={ticks} solver cell accuracy: "
          f"{accuracy:.3f} solved {solved}/{len(cases)} "
          f"(compile+run {first:.1f}s, run {second:.2f}s)")
    assert accuracy == 1.0
    print("exact solver check passed")


if __name__ == "__main__":
    main(ticks=int(sys.argv[1]) if len(sys.argv) > 1 else 2,
         family=sys.argv[2] if len(sys.argv) > 2 else "square")
