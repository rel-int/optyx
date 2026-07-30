# TODO

> I've opened issue https://github.com/rel-int/optyx/issues/14, make a plan for implementation.

Plan for `channel.Diagram.fix(n_steps, chi)`: approximate the stationary state of a
stateful diagram by tensor network contraction with bounded bond dimension.
Depends on #12 (`feedback` / `unroll`): implementation starts once #12 merges,
or stacks on its branch if it is still open.

## Mathematical description

A diagram `D` with feedback loops of total memory `mem` denotes a family of channels
indexed by time. For `dom == Ty()`, the state at time `n` is the unrolling over `n`
steps with each loop's `initial_state` plugged in, every output before the last
discarded and the final memory discarded. The stationary state is the limit
`n → ∞`, i.e. the fixed point of the one-step doubled transfer channel on `mem`.
The doubled unrolled network is a quasi-1D ladder, so compressed contraction with
bond dimension `chi` — an MPO power method — approximates the fixed point.

## Unroll and contract

- [x] helper building the time-`n` state diagram: `unroll(n_steps)` with
      intermediate outputs and final memory discarded, from `Ty()` to `cod`
- [x] `channel.Diagram.fix(n_steps, chi, backend=None)` contracting the doubled
      network through `QuimbBackend` with a compressed hyperoptimiser
      (`max_bond=chi`), returning the density matrix over `cod` as an `EvalResult`
- [x] `ValueError` on `dom != Ty()` and on diagrams without feedback loops
- [x] tests against exact power iteration on small instances: the delay line and
      a dephasing rotation loop; a doctest with a known fixed point

## Methods of the ref, as tensor contractions

The ref proposes three methods; each is (or embeds in) a contraction of the same
doubled network, so `fix` can expose them as strategies and the notebook can race
them. *Caveat: arxiv.org is unreachable from the agent environment, so this section
is reconstructed from the abstract and standard formulations — verify against the
paper before implementing, and correct the checkboxes if it disagrees.*

1. **Spatiotemporal mode-unfolding** — time-bins become extra modes: exactly our
   `unroll`, evaluated exactly (permanents through `to_path` /
   `PermanentBackend`). Exponential in total photon number, no truncation error;
   the baseline the others are tested against.
2. **Kraus-operator formalism** — the one-step process on the memory is a channel;
   its doubled transfer map is one tensor. Stationary state = eigenvector of the
   transfer matrix at eigenvalue 1: for small `mem` and Fock cutoff, one
   eigensolve, no `n_steps` at all. Iterating it instead is the `chi = d_mem²`
   case of our MPO power method — `fix` at `chi = None, n_steps` finite.
3. **Correlation-tensor approach** — propagate low-order correlation tensors of
   the mode operators through one step; the stationary values solve a small linear
   fixed-point system. Polynomial in the number of modes but returns marginals,
   not the density matrix `fix` promises — a validation and benchmark tool, not an
   implementation of `fix`.

Efficiency: (2) as a direct eigensolve wins whenever the memory fits in
`cutoff^|mem|`; the `chi`-bounded power method is the only one that scales past
that; (1) is the exact baseline for tests; (3) is the cheapest check on marginals.

- [ ] exact baseline: unrolled evaluation through permanents, reusing
      `PermanentBackend` (tests + notebook, little new code)
- [x] `method="eigen"`: build the doubled one-step transfer tensor, eigensolve
      for the fixed point when the memory dimension is tractable
- [x] `method="power"` (default): the `chi`-compressed contraction above
- [ ] correlation-tensor marginals in the notebook to cross-check both

## Heuristics for the defaults

- [x] `n_steps=None`: double the depth until the distance between successive
      states drops below a new `tol` keyword — Frobenius, not trace distance:
      the trace distance needs a row/column pairing of the doubled wires which
      the single wire doubling of classical types makes ambiguous
- [x] `chi=None`: double the bond dimension until the result is stable within
      `tol`, capped by `max_chi`; both double together in one loop
- [ ] leave a hook for formal bounds (photonic bounds in progress by @armandld,
      heuristics from https://arxiv.org/abs/2602.05566)

## Notebook

- [ ] `examples/` notebook benchmarking quimb compressed contraction (CPU and
      PyTorch backend) against the methods of https://arxiv.org/abs/2602.05566
      on boson sampling with feedback; convergence plots in `n_steps` and `chi`

## Still open after the first implementation round

- the reading of the ref is still unverified: arxiv.org is blocked by the network
  policy of the agent environment, so the three methods above were reconstructed
  from the abstract. The mapping onto `method="eigen"` and `method="power"` needs
  checking against the paper, and `n_steps` / `chi` heuristics taken from it
- `fix` doubles `n_steps` and `chi` together rather than converging them
  separately, which is cheaper to implement but conflates two questions

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.
