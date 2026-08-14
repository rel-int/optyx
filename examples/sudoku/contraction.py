"""Exact batched contraction and training of the unrolled sudoku map.

The unrolled protocol of the sudoku ``interaction.CMap`` is contracted
exactly -- no compressed bond -- with a Cotengra path, on JAX arrays so
the whole computation is differentiated and batched over puzzles on the
GPU. One expression per readout cell leaves that cell's final prediction
leg open, giving its four digit scores in a single contraction; clue
predictions meet their digit effect at every tick and free predictions
are marginalised.
"""

import cotengra as ctg
import numpy as np

import experiment as ex


def family_dims(family):
    """Leg dimensions per ansatz family. Born messages are single qubits
    as in the reference notebook and its memory a single coherent qubit;
    the quantum family reads and writes measured two-qubit digit
    messages and keeps a two-qubit memory coherent, so its memory wire
    is the doubled space of dimension sixteen; the classical families
    carry dimension-four digit wires throughout. Predictions fold to one
    dimension-four leg everywhere."""
    if family == "born":
        return {"port": 2, "memory": 2, "prediction": 4}
    if family == "quantum":
        return {"port": 4, "memory": 16, "prediction": 4}
    return {"port": 4, "memory": 4, "prediction": 4}


def build_expression(structure, ticks, target, family, optimize,
                     feedback=1, strip_exponent=True):
    """One readout expression: index lists, argument spec and path."""
    dims = family_dims(family)
    box_in, box_out, initial, final, prediction = ex.unrolled_indices(
        structure, ticks)
    inputs, arg_spec, size_dict = [], [], {}

    def leg_dim(box, key):
        if key < box.n_ports:
            return dims["port"]
        if key < box.n_ports + box.n_memory:
            return dims["memory"]
        return dims["prediction"]

    def add(indices, spec, shape):
        inputs.append(tuple(indices))
        arg_spec.append(spec)
        for index, dim in zip(indices, shape):
            size_dict[index] = dim

    fresh_bond = iter(range(10 ** 9, 2 * 10 ** 9))
    split = family != "born"
    for tick in range(ticks):
        for b, box in enumerate(structure.boxes):
            reads = [box_in[(b, k, tick)]
                     for k in range(box.n_ports + box.n_memory)]
            writes = [box_out[(b, k, tick)] for k in range(
                box.n_ports + box.n_memory + box.n_prediction)]
            read_dims = [leg_dim(box, k)
                         for k in range(box.n_ports + box.n_memory)]
            write_dims = [leg_dim(box, k) for k in range(
                box.n_ports + box.n_memory + box.n_prediction)]
            if split and box.kind == "constraint":
                middle = next(fresh_bond)
                add(reads + [middle],
                    ("dense", "constraint_read"), read_dims + [feedback])
                for port, (write, dim) in enumerate(
                        zip(writes, write_dims)):
                    add([middle, write],
                        ("dense", "constraint_write", port),
                        [feedback, dim])
            else:
                add(reads + writes, ("dense", box.kind),
                    read_dims + write_dims)
    for (b, key), index in initial:
        box = structure.boxes[b]
        kind = "memory" if key >= box.n_ports else "port"
        add([index], ("initial", kind), [leg_dim(box, key)])
    for (b, key), index in final:
        box = structure.boxes[b]
        kind = "memory" if key >= box.n_ports else "port"
        add([index], ("final", kind), [leg_dim(box, key)])
    output = ()
    for (b, tick), index in prediction.items():
        if (b, tick) == (target, ticks - 1):
            output = (index,)
            size_dict[index] = dims["prediction"]
        else:
            add([index], ("effect", b, tick), [dims["prediction"]])
    expression = ctg.array_contract_expression(
        inputs, output, size_dict=size_dict, optimize=optimize,
        strip_exponent=strip_exponent)
    return expression, arg_spec


def default_optimizer(max_repeats=16):
    """A reusable random-greedy hyper search, shared by the sixteen
    readout expressions of one configuration."""
    return ctg.ReusableHyperOptimizer(
        methods=["greedy"], max_repeats=max_repeats, optlib="random",
        parallel=False, progbar=False, minimize="combo")


def make_scores(structure, ticks, family, feedback=1,
                optimize=None, strip_exponent=True, targets=None):
    """The (n_cells, 4) digit scores of every cell, one open-leg
    contraction per cell, as a function of the shared box tensors and the
    per-puzzle prediction effects."""
    import jax.numpy as jnp

    if optimize is None:
        optimize = default_optimizer()
    if family == "born":
        constants = {
            ("initial", "port"): np.ones(2) / 2 ** .5,
            ("initial", "memory"): np.array([1., 0.]),
            ("final", "port"): np.ones(2) / 2 ** .5,
            ("final", "memory"): np.ones(2) / 2 ** .5,
        }
    else:
        constants = {
            ("initial", "port"): np.ones(4) / 4,
            ("initial", "memory"): np.ones(4) / 4,
            ("final", "port"): np.ones(4),
            ("final", "memory"): np.ones(4),
        }
    if family == "quantum":
        constants.update({
            ("initial", "port"): np.array([1., 0., 0., 0.]),
            ("initial", "memory"): np.eye(4).ravel() / 4,
            ("final", "port"): np.ones(4),
            ("final", "memory"): np.eye(4).ravel(),
        })
    constants = {key: jnp.asarray(value) for key, value in constants.items()}
    if targets is None:
        targets = [b for b, box in enumerate(structure.boxes)
                   if box.n_prediction]
    cells = [build_expression(
        structure, ticks, target, family, optimize,
        feedback=feedback, strip_exponent=strip_exponent)
        for target in targets]

    def resolve(spec, tensors, effects):
        kind = spec[0]
        if kind == "dense":
            tensor = tensors[spec[1]]
            return tensor[spec[2]] if len(spec) == 3 else tensor
        if kind == "effect":
            return effects[spec[1], spec[2]]
        return constants[spec]

    def scores(tensors, effects):
        rows = []
        for expression, arg_spec in cells:
            args = [resolve(spec, tensors, effects) for spec in arg_spec]
            result = expression(*args)
            if strip_exponent:
                result = result[0]       # the exponent cancels per cell
            rows.append(result)
        raw = jnp.stack(rows)
        if family == "born":
            raw = raw ** 2
        return raw

    return scores


# ---------------------------------------------------------------------------
# Box tensors from parameters, per ansatz family
# ---------------------------------------------------------------------------

def assemble_cell(cores):
    """The dense cell tensor from its chain of cores: read legs are the
    three ports then the memory, write legs the ports, memory and
    prediction, matching the leg order of ``build_expression``."""
    import jax.numpy as jnp
    return jnp.einsum("iAmn,ABab,BCcd,CDef,DjP->acembdfnP", *cores)


def assemble_constraint_read(cores):
    import jax.numpy as jnp
    return jnp.einsum("iAa,ABb,BCc,Ckd->abcdk", *cores)


def make_tensors_fn(family, config):
    import jax
    import jax.numpy as jnp

    def normalise(core):
        scale = jax.lax.stop_gradient(jnp.max(jnp.abs(core))) + 1e-300
        return core / scale

    if family in ("exp", "square"):
        link = {"exp": jnp.exp, "square": jnp.square}[family]

        def tensors_fn(params):
            cell, constraint_read, constraint_write = params
            return {
                "cell": normalise(assemble_cell(
                    [link(core) for core in cell])),
                "constraint_read": normalise(assemble_constraint_read(
                    [link(core) for core in constraint_read])),
                "constraint_write": jnp.stack(
                    [normalise(link(vector))
                     for vector in constraint_write]),
            }
        return tensors_fn

    assert family in ("born", "quantum")

    def rotation_layer(thetas):
        result = jnp.ones((1, 1))
        for theta in thetas:
            block = jnp.array(
                [[jnp.cos(theta), -jnp.sin(theta)],
                 [jnp.sin(theta), jnp.cos(theta)]])
            result = jnp.kron(result, block)
        return result

    def conditional_rotation(result, thetas, target, width):
        basis = np.arange(2 ** width)
        lower = basis[((basis >> target) & 1) == 0]
        upper = lower | (1 << target)
        cosine, sine = jnp.cos(thetas)[:, None], jnp.sin(thetas)[:, None]
        new_lower = cosine * result[lower] - sine * result[upper]
        new_upper = sine * result[lower] + cosine * result[upper]
        return result.at[lower].set(new_lower).at[upper].set(new_upper)

    def channel_unitary(thetas, width, depth):
        conditional = 2 ** (width - 1)
        result = rotation_layer(thetas[:width])
        for layer in range(depth):
            offset = (depth + 1) * width + layer * conditional
            result = conditional_rotation(
                result, thetas[offset:offset + conditional],
                layer % width, width)
            offset = (layer + 1) * width
            result = rotation_layer(thetas[offset:offset + width]) @ result
        return result

    if family == "born":
        depth = config["depth"]

        def tensors_fn(params):
            cell_thetas, constraint_thetas = params
            cell = channel_unitary(cell_thetas, 6, depth)[:2 ** 4]
            constraint = channel_unitary(constraint_thetas, 4, depth)
            return {
                "cell": cell.reshape((2,) * 4 + (2,) * 4 + (4,)),
                "constraint": constraint.reshape((2,) * 8),
            }
        return tensors_fn

    cell_depth = config["cell_depth"]
    cons_depth = config["cons_depth"]
    feedback = config.get("feedback", 2)

    def doubled_cell(unitary):
        """The doubled channel tensor of the cell circuit: classical
        digit messages in and out, coherent two-qubit memory, measured
        two-qubit prediction from fresh ancillas."""
        kraus = unitary.reshape(1024, 4, 4, 4, 4, 4)[..., 0]
        kraus = kraus.reshape(4, 4, 4, 4, 4, 4, 4, 4, 4)
        tensor = jnp.einsum(
            "MABCPuabc,NABCPvabc->abcuvABCMNP", kraus, kraus)
        return tensor.reshape(4, 4, 4, 16, 4, 4, 4, 16, 4)

    def doubled_constraint(unitary):
        """The verdict distribution of the constraint circuit: nine
        qubits in (four claims and a fresh ancilla), the verdict register
        measured and the other qubits traced out."""
        kraus = unitary.reshape(feedback, 512 // feedback, 256, 2)[..., 0]
        return jnp.einsum("kja,kja->ak", kraus, kraus).reshape(
            4, 4, 4, 4, feedback)

    def tensors_fn(params):
        cell_thetas, constraint_thetas, writes = params
        return {
            "cell": normalise(doubled_cell(
                channel_unitary(cell_thetas, 10, cell_depth))),
            "constraint_read": normalise(doubled_constraint(
                channel_unitary(constraint_thetas, 9, cons_depth))),
            "constraint_write": normalise(jnp.square(writes)),
        }
    return tensors_fn


# ---------------------------------------------------------------------------
# Puzzle encoding, loss, training
# ---------------------------------------------------------------------------

def puzzle_effects(clues, ticks):
    """Per-cell per-tick dim-4 prediction effects: clue digits are
    postselected at every tick, free cells are left unread."""
    effects = np.ones((ex.N_CELLS, ticks, 4))
    for cell, value in enumerate(clues):
        if value:
            onehot = np.zeros(4)
            onehot[value - 1] = 1
            effects[cell, :, :] = onehot
    return effects


def encode_cases(cases, ticks, scope="full"):
    if scope == "full":
        effects = np.stack(
            [puzzle_effects(clues, ticks) for clues, _ in cases])
    else:
        local_cells = [
            ex.local_structure(target)[1]
            for target in range(ex.N_CELLS)]
        effects = np.stack([
            np.stack([puzzle_effects(clues[cells], ticks)
                      for cells in local_cells])
            for clues, _ in cases])
    solutions = np.stack([solution for _, solution in cases]) - 1
    hidden = np.stack([clues == 0 for clues, _ in cases])
    return effects, solutions, hidden


def full_probabilities(scores, tensors_fn):
    """Per-cell digit probabilities from one full-map contraction per
    cell; ``effects`` is batched as ``(batch, n_cells, ticks, 4)``."""
    import jax

    def one_puzzle(tensors, effects):
        raw = scores(tensors, effects)
        return raw / (raw.sum(axis=-1, keepdims=True) + 1e-300)

    batched = jax.vmap(one_puzzle, in_axes=(None, 0))

    def probabilities(params, effects):
        return batched(tensors_fn(params), effects)

    return probabilities


def local_probabilities(ticks, family, feedback, tensors_fn,
                        optimize=None, strip_exponent=False):
    """Per-cell digit probabilities from the light-cone map of each
    cell, one exact contraction per target; ``effects`` is batched as
    ``(batch, n_cells, n_local_boxes, ticks, 4)``, sliced per target
    from the clues of its local cells."""
    import jax
    import jax.numpy as jnp

    scorers = []
    for target in range(ex.N_CELLS):
        structure, _ = ex.local_structure(target)
        scorers.append(make_scores(
            structure, ticks, family, feedback=feedback,
            optimize=optimize or default_optimizer(), targets=[0],
            strip_exponent=strip_exponent))

    def one_puzzle(tensors, effects):
        raw = jnp.stack([
            scorers[target](tensors, effects[target])[0]
            for target in range(ex.N_CELLS)])
        return raw / (raw.sum(axis=-1, keepdims=True) + 1e-300)

    batched = jax.vmap(one_puzzle, in_axes=(None, 0))

    def probabilities(params, effects):
        return batched(tensors_fn(params), effects)

    return probabilities


def make_loss(probabilities):
    import jax.numpy as jnp

    def loss(params, effects, solutions, hidden):
        p = probabilities(params, effects)
        picked = jnp.take_along_axis(
            p, solutions[..., None], axis=-1)[..., 0]
        cross_entropy = -jnp.log(picked + 1e-300)
        return (cross_entropy * hidden).sum() / hidden.sum()

    return loss


def evaluate(probabilities, params, cases, ticks, batch=32, scope="full"):
    """Held-out per-cell accuracy and full-grid solve rate."""
    effects, solutions, hidden = encode_cases(cases, ticks, scope)
    correct, total, solved = 0, 0, 0
    for start in range(0, len(cases), batch):
        p = np.asarray(probabilities(
            params, effects[start:start + batch]))
        guess = p.argmax(axis=-1)
        sol = solutions[start:start + batch]
        hid = hidden[start:start + batch]
        hits = (guess == sol) & hid
        correct += int(hits.sum())
        total += int(hid.sum())
        solved += int((hits.sum(axis=1) == hid.sum(axis=1)).sum())
    return {
        "cell_accuracy": correct / total,
        "solve_rate": solved / len(cases),
    }


def train(config, train_cases, test_cases, log=print):
    """Train one configuration and return its metric history."""
    import time

    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)
    family, ticks = config["family"], config["ticks"]
    seed = config.get("seed", 7)
    structure = ex.sudoku_structure()
    scale = config.get("init_scale", 0.3)
    if family == "born":
        cell, constraint = ex.init_born(config["depth"], seed)
        params = (jnp.asarray(cell), jnp.asarray(constraint))
        n_parameters = cell.size + constraint.size
        feedback = 1
    elif family == "quantum":
        feedback = config.get("feedback", 2)
        params = tuple(map(jnp.asarray, ex.init_quantum(
            config["cell_depth"], config["cons_depth"], feedback,
            seed, scale)))
        n_parameters = ex.quantum_parameter_count(
            config["cell_depth"], config["cons_depth"], feedback)
    else:
        bond, feedback = config["bond"], config.get("feedback", 1)
        groups = ex.init_stochastic(bond, feedback, seed, scale=scale)
        params = tuple(
            [jnp.asarray(core) for core in group] for group in groups)
        n_parameters = ex.parameter_count_stochastic(bond, feedback)
    tensors_fn = make_tensors_fn(family, config)
    scope = config.get("scope", "full")
    strip = config.get("strip_exponent", False)
    if scope == "full":
        scores = make_scores(
            structure, ticks, family, feedback=feedback,
            optimize=default_optimizer(config.get("path_repeats", 32)),
            strip_exponent=strip)
        probabilities = full_probabilities(scores, tensors_fn)
    else:
        probabilities = local_probabilities(
            ticks, family, feedback, tensors_fn,
            optimize=default_optimizer(config.get("path_repeats", 32)),
            strip_exponent=strip)
    loss = make_loss(probabilities)
    probabilities = jax.jit(probabilities)

    optimizer = optax.adam(config.get("learning_rate", 3e-3))
    trainable = params

    @jax.jit
    def update(trainable, opt_state, effects, solutions, hidden):
        value, grads = jax.value_and_grad(loss)(
            trainable, effects, solutions, hidden)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(trainable, updates), opt_state, value

    def full_params(trainable):
        return tuple(trainable)

    effects, solutions, hidden = encode_cases(train_cases, ticks, scope)
    opt_state = optimizer.init(trainable)
    random = np.random.default_rng(seed)
    batch_size = config.get("batch_size", 32)
    n_steps = config.get("n_steps", 1000)
    eval_every = config.get("eval_every", 200)
    history, started = [], time.perf_counter()
    for step in range(n_steps):
        pick = random.choice(len(train_cases), batch_size, replace=False)
        trainable, opt_state, value = update(
            trainable, opt_state, effects[pick], solutions[pick],
            hidden[pick])
        if step % eval_every == eval_every - 1 or step == n_steps - 1:
            metrics = evaluate(
                probabilities, full_params(trainable), test_cases, ticks,
                scope=scope)
            metrics.update(
                step=step + 1, loss=float(value),
                seconds=time.perf_counter() - started)
            history.append(metrics)
            log(f"{config} step {step + 1}: loss {value:.4f} "
                f"test acc {metrics['cell_accuracy']:.3f} "
                f"solved {metrics['solve_rate']:.3f}")
    result = {
        "config": {key: value for key, value in config.items()},
        "n_parameters": int(n_parameters),
        "history": history,
        "final": history[-1] if history else None,
    }
    return result, full_params(trainable)
