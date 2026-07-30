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

- [ ] helper building the time-`n` state diagram: `unroll(n_steps)` with
      intermediate outputs and final memory discarded, from `Ty()` to `cod`
- [ ] `channel.Diagram.fix(n_steps, chi, backend=None)` contracting the doubled
      network through `QuimbBackend` with a compressed hyperoptimiser
      (`max_bond=chi`), returning the density matrix over `cod` as an `EvalResult`
- [ ] `ValueError` on `dom != Ty()` and on diagrams without feedback loops
- [ ] tests against exact power iteration on small instances: the delay line and
      the CNOT ladder of #12; a doctest with a known fixed point

## Heuristics for the defaults

- [ ] `n_steps=None`: double the depth until the trace distance between successive
      states drops below a new `tol` keyword
- [ ] `chi=None`: double the bond dimension until the result is stable within `tol`
- [ ] leave a hook for formal bounds (photonic bounds in progress by @armandld,
      heuristics from https://arxiv.org/abs/2602.05566)

## Notebook

- [ ] `examples/` notebook benchmarking quimb compressed contraction (CPU and
      PyTorch backend) against the methods of https://arxiv.org/abs/2602.05566
      on boson sampling with feedback; convergence plots in `n_steps` and `chi`

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.
