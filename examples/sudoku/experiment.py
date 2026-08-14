"""Learning to solve 4x4 sudoku with recurrent channels.

The architecture is the ``interaction.CMap`` of the reference notebook of
rel-int/optyx#16: one box per cell and one box per row, column and square,
one bidirectional digit wire pairing each cell with each of its three
constraints. Each cell carries a private memory and writes a prediction
at every tick. The unrolled protocol is contracted exactly -- no
compressed bond -- with Cotengra paths on JAX arrays, batched over
puzzles.

Three channel ansatze share this architecture:

* ``quantum``: every box is a channel built from an orthogonal
  conditional-rotation circuit -- a ten-qubit circuit for the cell,
  whose two-qubit memory stays coherent between ticks while its digit
  messages and prediction are measured, and a nine-qubit
  measure-and-prepare circuit for the constraint, whose measured verdict
  register of dimension ``feedback`` indexes one prepared state per
  port. By Stinespring this family contains every classical stochastic
  channel, ``solver_quantum`` being the solver as permutation circuits.
* ``exp`` and ``square``: the decohered ablation -- classical stochastic
  channels on dimension-four digit wires, transition tensors factored
  into chains of non-negative cores with bond dimension ``bond``, the
  constraint split into a read chain and per-port write vectors of rank
  ``feedback``. ``solver_cores`` is the exact solver at cell bond four
  and constraint bond six, inside the trainable range.
* ``born``: the fully coherent continuation of the reference notebook,
  single-qubit messages through orthogonal circuits, no decoherence,
  Born-rule readout of the open prediction legs.
"""

from dataclasses import dataclass

import numpy as np

SIZE = 4
N_CELLS = SIZE ** 2
N_CONSTRAINTS = 3 * SIZE
N_CLUES = N_CELLS // 2


def sudoku_groups():
    rows = [tuple(row * SIZE + column for column in range(SIZE))
            for row in range(SIZE)]
    columns = [tuple(row * SIZE + column for row in range(SIZE))
               for column in range(SIZE)]
    squares = [
        tuple((2 * block_row + row) * SIZE + 2 * block_column + column
              for row in range(2) for column in range(2))
        for block_row in range(2) for block_column in range(2)]
    return rows + columns + squares


GROUPS = sudoku_groups()
PEERS = [set() for _ in range(N_CELLS)]
for group in GROUPS:
    for cell in group:
        PEERS[cell].update(set(group) - {cell})


def enumerate_solutions():
    """All 288 completed 4x4 sudoku grids, digits 1 to 4."""
    result, grid = [], [0] * N_CELLS

    def visit(cell):
        if cell == N_CELLS:
            result.append(np.array(grid))
            return
        for value in range(1, SIZE + 1):
            if all(grid[peer] != value for peer in PEERS[cell]):
                grid[cell] = value
                visit(cell + 1)
        grid[cell] = 0

    visit(0)
    return np.stack(result)


def make_dataset(seed=19, n_cases=256, n_train=192):
    """The leakage-free dataset of the reference notebook.

    256 distinct completed grids split 192/64 before masking; every
    puzzle keeps eight clues (cells 0 and 1 always hidden) that identify
    a unique completion in the 288-grid corpus.
    """
    solutions = enumerate_solutions()
    random = np.random.default_rng(seed)
    selection = random.choice(len(solutions), n_cases, replace=False)
    cases = []
    for index in selection:
        solution = solutions[index]
        available = np.arange(2, N_CELLS)
        while True:
            clue_cells = random.choice(available, N_CLUES, replace=False)
            matches = np.all(
                solutions[:, clue_cells] == solution[clue_cells], axis=1)
            if matches.sum() == 1:
                clues = np.zeros(N_CELLS, dtype=int)
                clues[clue_cells] = solution[clue_cells]
                cases.append((clues, solution))
                break
    assert len({tuple(solution) for _, solution in cases}) == n_cases
    return cases[:n_train], cases[n_train:]


def memberships(row, column):
    square = 2 * (row // 2) + column // 2
    position = 2 * (row % 2) + column % 2
    return (
        (N_CELLS + row, column),
        (N_CELLS + SIZE + column, row),
        (N_CELLS + 2 * SIZE + square, position),
    )


@dataclass(frozen=True)
class BoxSpec:
    """One box of the map: message ports, memory legs, prediction legs."""
    kind: str
    n_ports: int
    n_memory: int
    n_prediction: int


@dataclass(frozen=True)
class Structure:
    """The combinatorial structure of the sudoku ``interaction.CMap``."""
    boxes: tuple
    edges: tuple

    @property
    def partner(self):
        result = {}
        for source, target in self.edges:
            result[source] = target
            result[target] = source
        return result


def sudoku_structure():
    """One box per cell, row, column and square: cell port ``s`` is paired
    with port ``p`` of its ``s``-th constraint, for the cell at position
    ``p`` of that constraint. Each incidence carries one bidirectional
    digit wire; every port is read and written at every tick as in
    ``optyx.interaction``."""
    boxes = tuple(
        [BoxSpec("cell", 3, 1, 1) for _ in range(N_CELLS)]
        + [BoxSpec("constraint", 4, 0, 0) for _ in range(N_CONSTRAINTS)])
    edges = []
    for cell in range(N_CELLS):
        row, column = divmod(cell, SIZE)
        for slot, (constraint, position) in enumerate(
                memberships(row, column)):
            edges.append(((cell, slot), (constraint, position)))
    return Structure(boxes, tuple(edges))


def local_structure(target):
    """The light cone of one cell: the target, its seven peers and its
    three constraints, ports to outside constraints left unpaired --
    they read a fresh environment state and write to a discarded one at
    every tick. Returns the structure and the global cell index of each
    local cell box."""
    row, column = divmod(target, SIZE)
    constraints = [c for c, _ in memberships(row, column)]
    cells = [target] + sorted(PEERS[target])
    cell_slot = {cell: i for i, cell in enumerate(cells)}
    cons_slot = {c: len(cells) + j for j, c in enumerate(constraints)}
    boxes = tuple(
        [BoxSpec("cell", 3, 1, 1) for _ in cells]
        + [BoxSpec("constraint", 4, 0, 0) for _ in constraints])
    edges = []
    for cell in cells:
        r, c = divmod(cell, SIZE)
        for slot, (constraint, position) in enumerate(memberships(r, c)):
            if constraint in cons_slot:
                edges.append((
                    (cell_slot[cell], slot),
                    (cons_slot[constraint], position)))
    return Structure(boxes, tuple(edges)), np.asarray(cells)


def check_structure_against_interaction(structure, cell_box, constraint_box,
                                        cmap_factory):
    """Assert the trainer's structure matches an ``interaction.CMap``."""
    boxes = [cell_box] * N_CELLS + [constraint_box] * N_CONSTRAINTS
    cmap = cmap_factory(boxes, list(structure.edges))
    assert len(cmap.boundary) == 0
    assert len(cmap.paired) == 2 * len(structure.edges)
    assert len(cmap.memories) == N_CELLS
    assert len(cmap.predictions) == N_CELLS
    return cmap


# ---------------------------------------------------------------------------
# Tensor-network construction: indices of the unrolled protocol
# ---------------------------------------------------------------------------

def unrolled_indices(structure, ticks):
    """Index names of the unrolled tensor network.

    Each box at each tick reads every port and its memory, and writes
    every port, its memory and its prediction. A paired port's write is
    the partner's read one tick later; memory feeds back to the same box;
    tick-0 reads meet initial states, final writes meet closing effects,
    predictions meet one effect per tick.
    """
    partner = structure.partner
    fresh = iter(range(10 ** 9))

    def wire():
        return next(fresh)

    box_in = {}
    box_out = {}
    initial, final, prediction = [], [], []
    for tick in range(ticks):
        for b, box in enumerate(structure.boxes):
            for p in range(box.n_ports):
                if tick == 0 or (b, p) not in partner:
                    index = wire()
                    initial.append(((b, p), index))
                else:
                    other = partner[(b, p)]
                    index = box_out[(other[0], other[1], tick - 1)]
                box_in[(b, p, tick)] = index
                out = wire()
                box_out[(b, p, tick)] = out
                if tick == ticks - 1 or (b, p) not in partner:
                    final.append(((b, p), out))
            for m in range(box.n_memory):
                key = box.n_ports + m
                if tick == 0:
                    index = wire()
                    initial.append(((b, key), index))
                else:
                    index = box_out[(b, key, tick - 1)]
                box_in[(b, key, tick)] = index
                out = wire()
                box_out[(b, key, tick)] = out
                if tick == ticks - 1:
                    final.append(((b, key), out))
            for o in range(box.n_prediction):
                key = box.n_ports + box.n_memory + o
                out = wire()
                box_out[(b, key, tick)] = out
                prediction.append(((b, tick), out))
    return box_in, box_out, initial, final, dict(prediction)


# ---------------------------------------------------------------------------
# Ansatz families
# ---------------------------------------------------------------------------

def stochastic_shapes(bond, feedback=1):
    """Core shapes of the factored digit-wire ansatz.

    A cell channel is a chain over sites ``[memory, three message ports,
    prediction]``, each carrying one dimension-four read leg and one
    write leg except the prediction which only writes. A constraint
    channel is a chain of its four reads ending in a middle bond of
    dimension ``feedback``, followed by one write vector per port
    indexed by that bond: what a constraint writes back is a product
    over its ports given one of ``feedback`` many read verdicts. The
    hand-coded solver needs ``feedback=1`` -- constraints postselect and
    write nothing back -- and larger ranks make the echo learnable.
    Port-factored writes are what keep the exact contraction of the
    unrolled network tractable.
    """
    cell = [(1, bond, 4, 4)] + [(bond, bond, 4, 4)] * 3 + [(bond, 1, 4)]
    constraint_read = (
        [(1, bond, 4)] + [(bond, bond, 4)] * 2 + [(bond, feedback, 4)])
    constraint_write = [(feedback, 4)] * 4
    return cell, constraint_read, constraint_write


def parameter_count_stochastic(bond, feedback=1):
    return int(sum(
        np.prod(shape) for group in stochastic_shapes(bond, feedback)
        for shape in group))


def init_stochastic(bond, feedback, seed, scale=0.3):
    random = np.random.default_rng(seed)
    return tuple(
        [random.normal(0, scale, shape) for shape in group]
        for group in stochastic_shapes(bond, feedback))


def init_stochastic_structured(bond, feedback, seed, family, noise=0.05):
    """Solver plumbing plus noise for the classical families: cell cores
    start at the digit-memory broadcast of ``solver_cores`` (padded to
    the requested bond with noise), the constraint logic stays random."""
    random = np.random.default_rng(seed)
    solver_cell, _, _ = solver_cores()
    cell_shapes, read_shapes, write_shapes = stochastic_shapes(
        bond, feedback)
    cell = []
    for core, shape in zip(solver_cell, cell_shapes):
        padded = np.zeros(shape)
        padded[tuple(slice(0, s) for s in core.shape)] = core
        if family == "square":
            theta = np.sqrt(padded) + noise * random.normal(size=shape)
        else:
            theta = np.log(padded + 0.02) + noise * random.normal(
                size=shape)
        cell.append(theta)
    read = [random.normal(0, 0.3, shape) for shape in read_shapes]
    write = [random.normal(0, 0.3, shape) for shape in write_shapes]
    return cell, read, write


def solver_cores():
    """Hand-written exact-solver cores in the stochastic family.

    Cell (bond four): the memory carries a digit, every message write
    claims it, the prediction reads it, reads are ignored. Constraint
    (bond six in the middle, feedback rank one): the four reads must be
    pairwise-distinct digits, tracked as a subset along the read chain;
    the writes are uniform.
    """
    cell = [np.zeros((1, 4, 4, 4))]
    for m in range(4):
        cell[0][0, m, m, m] = 1          # memory persists along the bond
    passthrough = np.zeros((4, 4, 4, 4))
    for m in range(4):
        passthrough[m, m, :, m] = 1 / 4      # read ignored, write claims m
    cell += [passthrough] * 3
    predict = np.zeros((4, 1, 4))
    for m in range(4):
        predict[m, 0, m] = 1
    cell.append(predict)

    subsets = [frozenset()] + [frozenset({d}) for d in range(4)] + [
        frozenset(pair) for pair in
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]] + [
        frozenset({0, 1, 2, 3}) - {d} for d in range(4)] + [
        frozenset({0, 1, 2, 3})]
    index = {s: i for i, s in enumerate(subsets)}
    dim = len(subsets)

    def claim_core(site):
        core = np.zeros((dim, dim, 4))
        for s in subsets:
            if len(s) != site:
                continue
            for d in range(4):
                if d not in s:
                    core[index[s], index[s | {d}], d] = 1
        return core

    constraint_read = [claim_core(site) for site in range(4)]
    constraint_read[0] = constraint_read[0][:1]     # boundary: start empty
    constraint_read[3] = constraint_read[3][
        :, index[frozenset({0, 1, 2, 3})]][:, None]  # feedback rank one
    constraint_write = [np.ones((1, 4)) / 4] * 4     # echo nothing back
    return cell, constraint_read, constraint_write


def born_parameter_count(width, depth):
    return (depth + 1) * width + depth * 2 ** (width - 1)


def quantum_parameter_count(cell_depth, cons_depth, feedback):
    """The genuinely quantum family: a ten-qubit orthogonal circuit for
    the cell channel (two memory qubits kept coherent, three two-qubit
    messages read and written, two prediction qubits from fresh
    ancillas), a nine-qubit circuit for the constraint -- its four claim
    registers and one fresh ancilla -- whose measured verdict register
    of dimension ``feedback`` indexes one prepared digit state per
    port, the other eight qubits traced out. Any classical stochastic channel is realised
    by such circuits (Stinespring), so the hand-coded solver of
    ``solver_cores`` stays inside this family as permutation circuits
    with postselected verdicts."""
    return (born_parameter_count(10, cell_depth)
            + born_parameter_count(9, cons_depth) + 16 * feedback)


def init_quantum(cell_depth, cons_depth, feedback, seed, scale=0.3):
    random = np.random.default_rng(seed)
    return (
        random.normal(0, scale, born_parameter_count(10, cell_depth)),
        random.normal(0, scale, born_parameter_count(9, cons_depth)),
        random.normal(1, scale, (4, feedback, 4)))


def solver_quantum():
    """The exact solver as quantum circuits: the cell permutation sends
    ``(memory, messages, ancilla)`` to ``(memory, memory xor messages,
    memory xor ancilla)``; the constraint permutation computes the
    all-different flag of its four claims reversibly into the verdict
    qubit; verdict zero prepares the digit-zero echo, verdict one
    prepares nothing."""
    cell = np.zeros((2 ** 10, 2 ** 10))
    for basis in range(2 ** 10):
        mem = basis >> 8
        messages = [(basis >> (6 - 2 * s)) & 3 for s in range(3)]
        ancilla = basis & 3
        target = mem
        for message in messages:
            target = (target << 2) | (mem ^ message)
        target = (target << 2) | (mem ^ ancilla)
        cell[target, basis] = 1
    constraint = np.zeros((2 ** 9, 2 ** 9))
    for basis in range(2 ** 9):
        claims = [(basis >> (7 - 2 * p)) & 3 for p in range(4)]
        passing = len(set(claims)) == 4 or claims == [0, 0, 0, 0]
        flag = 0 if passing else 1               # zero: the initial claim
        constraint[basis ^ flag, basis] = 1      # verdict is the low bit
    writes = np.zeros((4, 2, 4))
    writes[:, 0, 0] = 1                          # verdict ok: echo zero
    return cell, constraint, writes


def solver_flag(claims):
    passing = len(set(claims)) == 4 or claims == [0, 0, 0, 0]
    return 0 if passing else 1


def solver_angles(cell_depth, cons_depth, feedback=2):
    """The exact solver as conditional-rotation angles, inside the
    trainable quantum ansatz.

    The cell circuit is an XOR ladder: the first ten conditional layers
    flip each message and prediction bit conditioned on the matching
    memory bit (angle ``pi/2`` where the control is one), so the cell
    broadcasts its memory digit and predicts it; deeper layers are
    idle. The constraint circuit computes the all-different flag of its
    four claims into the fresh ancilla with a single conditional layer
    -- one independent angle per claim configuration is exactly a
    generalised multi-controlled flip -- so the measured verdict is the
    flag. Signs from the rotations square away in the doubled channel.
    """
    assert cell_depth >= 10 and cons_depth >= 1 and feedback == 2
    cell = np.zeros(born_parameter_count(10, cell_depth))
    rotations = (cell_depth + 1) * 10
    for layer in range(10):
        target = layer % 10
        if target >= 8:
            continue                             # memory bits persist
        control = 9 if target % 2 else 8
        basis = np.arange(2 ** 10)
        lower = basis[((basis >> target) & 1) == 0]
        angles = np.where((lower >> control) & 1, np.pi / 2, 0.)
        offset = rotations + layer * 2 ** 9
        cell[offset:offset + 2 ** 9] = angles
    constraint = np.zeros(born_parameter_count(9, cons_depth))
    rotations = (cons_depth + 1) * 9
    lower = np.arange(2 ** 8)                    # ancilla-zero states
    flags = np.array([solver_flag([
        (claims >> 6) & 3, (claims >> 4) & 3,
        (claims >> 2) & 3, claims & 3]) for claims in lower])
    constraint[rotations:rotations + 2 ** 8] = np.where(
        flags, np.pi / 2, 0.)
    writes = np.zeros((4, 2, 4))
    writes[:, 0, 0] = 1                          # ok: echo digit zero
    return cell, constraint, writes


def init_quantum_structured(cell_depth, cons_depth, feedback, seed,
                            noise=0.05, logic="random"):
    """Solver plumbing plus noise: the cell keeps its broadcast ladder,
    the constraint logic is ``"solver"`` plus noise or ``"random"``."""
    random = np.random.default_rng(seed)
    cell, constraint, writes = solver_angles(
        cell_depth, cons_depth, feedback)
    cell = cell + noise * random.normal(size=cell.shape)
    if logic == "solver":
        constraint = constraint + noise * random.normal(
            size=constraint.shape)
        writes = writes + noise * np.abs(
            random.normal(size=writes.shape))
    else:
        constraint = 0.1 * random.normal(size=constraint.shape)
        writes = random.normal(1, 0.3, size=writes.shape)
    return cell, constraint, writes


def init_born(depth, seed, scale=0.1):
    """The conditional-rotation ansatz of the reference notebook on the
    single-wire map: a six-qubit unitary gives the cell isometry from its
    four reads (three ports and memory) to its four writes plus two fresh
    prediction qubits, a four-qubit unitary gives the constraint."""
    random = np.random.default_rng(seed)
    return (
        random.normal(0, scale, born_parameter_count(6, depth)),
        random.normal(0, scale, born_parameter_count(4, depth)))
