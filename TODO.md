# TODO

> I've opened issue https://github.com/rel-int/optyx/issues/14, make a plan for implementation.
> Implement the plan in the PR.
> Rebase https://github.com/rel-int/optyx/pull/15/changes on the current version of the
> feedback/unroll PR. Why is it showing diff in core.diagram.Feedback? The to_stream addition
> should also be justified. Review the existing code and push a TODO.md for making the code
> cleaner.

> Check out the optyx issue on fixpoints and PR [https://github.com/rel-int/optyx/pull/15/changes](https://github.com/rel-int/optyx/pull/15/changes). Why is it showing diff in core.diagram.Feedback? The to_stream addition should also be justified. Review the existing code and push a TODO.md for making the code cleaner and well documented.

Plan for `channel.Diagram.fix(n_steps, chi)`: approximate the stationary state of a
stateful diagram by tensor network contraction with bounded bond dimension.
Stacked on #12 (`feedback` / `unroll`), rebased on its head at every round.

## Mathematical description

A diagram `D` with feedback loops of total memory `mem` denotes a family of channels
indexed by time. For `dom == Ty()`, the state at time `n` is the unrolling over `n`
steps with each loop's `initial_state` plugged in, every output before the last
discarded and the final memory discarded. A stationary state is a fixed point of
the one-step doubled transfer channel on `mem`; its readout is the `n → ∞` limit
only when the finite-time states converge. The doubled unrolled network is a
quasi-1D ladder, so compressed contraction with bond dimension `chi` — an MPO
power method — approximates the fixed point reached from `initial_state`.

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

The paper separates four pieces which should not be presented as names for the two
`fix` strategies:

1. **Interferometer unfolding** turns time bins into spatial modes. This is the
   exact, quickly growing baseline represented by `unroll`.
2. **Partial-density-matrix evolution** applies one step and traces the detected
   modes at every iteration, keeping only the state in the looped modes.
3. **Kraus operators** represent that same loop channel, including loss. The
   paper obtains the stationary loop state as the eigenvector at eigenvalue one
   of its vectorised superoperator.
4. **Correlation tensors** solve stationary moment equations recursively and can
   reconstruct either the density matrix or only its diagonal, depending on how
   many tensor orders are retained.

Optyx's `method="power"` and `method="eigen"` are therefore contraction strategies
for the common channel, not the paper's method names. The eigen strategy matches
the superoperator calculation when the truncated memory fits in memory; compressed
unrolling is the scalable approximation; exact unfolding is the test baseline.

- [ ] exact baseline: unrolled evaluation through permanents, reusing
      `PermanentBackend` (tests + notebook, little new code)
- [ ] compare one-step partial-density-matrix iteration with the Kraus/superoperator
      construction on a small photonic example
- [x] `method="eigen"`: build the doubled one-step transfer tensor, eigensolve
      for the fixed point when the memory dimension is tractable
- [x] `method="power"` (default): the `chi`-compressed contraction above
- [ ] correlation-tensor reconstruction in the notebook to cross-check both the
      stationary loop state and the returned output distribution

## Heuristics for the defaults

- [x] `n_steps=None`: double the depth until the distance between successive
      states drops below a new `tol` keyword — Frobenius, not trace distance:
      the trace distance needs a row/column pairing of the doubled wires which
      the single wire doubling of classical types makes ambiguous
- [x] `chi=None`: double the bond dimension until the result is stable within
      `tol`, capped by `max_chi`; refine the depth independently at each bond
- [ ] leave a hook for formal bounds (photonic bounds in progress by @armandld,
      heuristics from https://arxiv.org/abs/2602.05566)

## Notebook

- [ ] `examples/` notebook benchmarking quimb compressed contraction (CPU and
      PyTorch backend) against the methods of https://arxiv.org/abs/2602.05566
      on boson sampling with feedback; convergence plots in `n_steps` and `chi`

## Still open after the first implementation round

- the paper truncates the joint multimode Fock space by total photon number
  `N_max`, whereas `eigen_fix` currently gives each doubled memory wire the same
  dimension `cutoff`; relate these conventions and document the error introduced
- [x] converge `n_steps` and `chi` separately so the depth and compression
      errors are checked independently

## Why `to_stream` belongs in this PR

Keep `to_stream` as an intentional conversion, not as the cached `.stream`
property rejected during #12. Both `unroll` and `one_step` need exactly the same
functorial interpretation of nested feedback loops; sharing it guarantees the same
memory and boundary-plug order without duplicating the semantics. Its public
contract is made explicit below:

- [x] document this rationale on `to_stream`, `unroll` and `one_step`, including
      that `stream.now` omits `initial_state` / `final_effect` and that the returned
      plugs follow feedback traversal order
- [x] replace the anonymous tuple annotation and `self.to_stream()[0]` with a named
      result or clear unpacking; test sequential, tensor and nested feedback loops
- [x] explain why this reusable conversion follows `STYLE.md` despite #12's reduced
      interface; if exposing DisCoPy's `Stream` cannot be given a stable contract,
      expose a narrower shared one-step primitive instead

## Cleanup round

Completed implementation review:

- [x] the convergence caps gate the wrong parameter: `fix(n_steps=100)` has
      `steps >= max_steps` on entry, so it warns and never doubles `chi` —
      each cap should only gate the parameter being doubled
- [x] a user-supplied `backend` ignores `chi`: the loop doubles `bond` but the
      contraction never sees it — rebuild the backend per bond, or reject
      `backend` together with `chi=None`
- [x] validate positive `n_steps`, `chi`, `cutoff`, `max_steps`, `max_chi` and
      `tol`; avoid `n_steps or 2` / `chi or 4`, which silently treats zero as a
      request for a default
- [x] converge `n_steps` and `chi` independently and report which cap was reached;
      set the warning stack level to point at the caller of `fix`
- [x] `fixed_point` picks the eigenvalue closest to one silently — check the
      residual and degeneracy, then preserve the initial state or raise/warn when
      the stationary state is not unique; check normalisation, Hermiticity and
      positivity before returning it
- [x] make `fix` a small validated dispatcher with named, separately testable power
      and eigen procedures; name the transfer, readout, memory dimension and
      contraction steps instead of a lambda and the `2 * (dimension,)` reshape
- [x] `at_time`: `Discard(self.cod ** (n_steps - 1))` instead of tensoring a
      list of `Discard`s, and drop the `rest` conditional if `Discard(Ty())`
      is the empty channel
- [x] move the `cotengra` import inside `fix` next to the `QuimbBackend` one,
      so importing `optyx.channel` stays light
- [x] move and rename `fixed_point` and `distance` rather than adding new debt to
      `utils.misc`, whose dissolution is planned in issue #5 / PR #8; use names
      which expose the superoperator convention and Frobenius metric
- [x] add identity, periodic, nearly-degenerate and custom-backend tests, plus
      independent adaptation tests for `n_steps` and `chi`

## Documentation round

- [x] add a short Sphinx guide linked from `docs/index.rst`: draw the one-step
      memory channel and readout; define `feedback`, `unroll`, `to_stream`,
      `one_step`, `at_time` and `fix`; distinguish the stationary loop state from
      the density matrix returned over `cod`
- [x] document the power/eigen trade-off and every parameter, the Fock truncation
      convention, custom-backend behaviour, convergence and uniqueness assumptions,
      warnings and the `dom == Ty()` limit; cite arXiv:2602.05566 precisely
- [x] keep doctests minimal and runnable, then run `pflake8 optyx`, the full
      coverage suite and the Sphinx build before claiming the cleanup round

## Review round: simple backend-agnostic contraction

- [x] rename the named stream result from
      `StreamSemantics` to `Stream` and update its public documentation
- [x] make fixed-point iteration compile through
      `to_tensor` to a `tensor.Diagram`, then execute it through a minimal
      backend interface; keep Quimb/Cotengra, exact tensor, JAX and PyTorch
      execution choices out of the fixed-point algorithm
- [x] simplify and expand the tensor-network
      explanation in `power_fix` and the Sphinx guide, including where
      compression approximates the exact contraction
- [x] add a notebook comparing the fixed-point
      methods and backends, estimate contraction cost before execution, and
      identify when bounded-bond approximation can help
- [x] add compatibility tests for exact and
      compressed tensor backends, then rerun lint, coverage and documentation

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.
