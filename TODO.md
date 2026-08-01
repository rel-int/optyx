# TODO

> Two problems: 1) I see you added a lot of pylint disables, inline comments are not allowed in optyx as in discopy! remove them and check the pylint, 2) The tensor backend is a large addition, write an issue and open a separate PR directly on main that proposes a new routine for evaluating optyx tensor contractions. The logic needs to be simple and flexible accross quimb, cotengra, jax and pytorch

## Review cleanup and tensor-contraction split

- [x] Remove the local validation scratch file and every
  suppression or inline comment introduced by this PR, then run pylint.
- [x] File a focused tensor-contraction issue, extract the backend-neutral
  evaluation routine onto a branch based directly on ``main``, validate it,
  and open a separate draft PR.
- [x] Remove the extracted tensor-backend implementation
  from this fixed-point
  PR, update its tests and documentation to the smaller public interface, and
  restore green checks.
- [x] Remove the platform-dependent convergence-warning
  assertion; the deterministic period-two test already covers the cap.

> The documentation notebook for fixpoint should construct the setup of boson sampling with feedback with state psi and unitatry U, construct it in optyx for some random unitary and different the bosonic product state, then call the different implementations of fixpoints and map when they agree

## Boson-sampling fixed-point example

- [x] Replace the toy reset channel in
  ``docs/examples/fixpoints.rst`` with the optical-feedback setup: inject a
  bosonic product state ``psi`` beside the loop memory, apply a seeded random
  unitary ``U``, feed the last modes back, and compare the fixed-point methods
  over several occupations with an agreement map.

> the docs/fixpoints notebook should be merged with the fixpoint_benchmarks notebook and should be under examples as rst in the docs

## Unified fixed-point example

- [x] Merge the fixed-point semantics and benchmark
  material into ``docs/examples/fixpoints.rst``. Present a feedback diagram
  both as a stream and as an approximate stationary readout, then compare
  every backend through the common ``tensor.Diagram`` without restoring an
  executable notebook.

> Let's edit the PR description, listing every contribution of this PR, described briefly. Remember the discopy STYLE.md guidelines
>
> Currently the repo also seems to have too many additions, the notebook should be integrated as a small documentation snippet in fix
>
> It should not be on the top of docs/ as it currently is

## Concise documentation round

- [x] Remove the standalone fixpoint guide and
  benchmark notebook, keep a small runnable example in ``Diagram.fix``, trim
  their navigation/configuration, and update the PR description with every
  contribution stated briefly.

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
      `PermanentBackend` (tests + a small API example, little new code)
- [ ] compare one-step partial-density-matrix iteration with the Kraus/superoperator
      construction on a small photonic example
- [x] `method="eigen"`: build the doubled one-step transfer tensor, eigensolve
      for the fixed point when the memory dimension is tractable
- [x] `method="power"` (default): the `chi`-compressed contraction above
- [ ] correlation-tensor reconstruction in a focused follow-up to cross-check
      both the stationary loop state and the returned output distribution

## Heuristics for the defaults

- [x] `n_steps=None`: double the depth until the distance between successive
      states drops below a new `tol` keyword — Frobenius, not trace distance:
      the trace distance needs a row/column pairing of the doubled wires which
      the single wire doubling of classical types makes ambiguous
- [x] `chi=None`: double the bond dimension until the result is stable within
      `tol`, capped by `max_chi`; refine the depth independently at each bond
- [ ] leave a hook for formal bounds (photonic bounds in progress by @armandld,
      heuristics from https://arxiv.org/abs/2602.05566)

## Example

- [x] Keep one RST example in `docs/examples/fixpoints.rst`: show the semantic
      map from feedback to a memory-free stationary readout and the common
      `tensor.Diagram` consumed by exact or compressed backends.

## Still open after the first implementation round

- [ ] the paper truncates the joint multimode Fock space by total photon number
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

- [x] document the two semantics locally: `to_stream` / `unroll` for streams,
      and `fix` for an approximate memory-free fixed point; draw the stationary
      memory readout in the fixed-point example rather than a top-level page
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
      explanation in `power_fix` and the fixed-point example, including where
      compression approximates the exact contraction
- [x] keep the backend comparison concise in the RST example: compile through
      `to_tensor` to a `tensor.Diagram`, then explain exact array execution and
      bounded-bond Quimb contraction without a notebook
- [x] add compatibility tests for exact and
      compressed tensor backends, then rerun lint, coverage and documentation

> The notebook should be structured in two parts:
> 1) Boson sampling with feedback
> Explain the setup of Biriukov and Dyakonov and the two simulation methods implemented.
> 2) Learning fixpoints
> Start from a beam splitter 2 -> 2, we feed a photon at every time step to the first
> input, we feed the second output into the second input. We compute the fixpoint of an
> example (does it converge? how big is the density matrix?). Now, the beam splitter has
> two phase parameters (transmittivity, reflexivity) that we can adjust. We tune the
> parameters to find fixpoints that converge faster, to lower dimensional density matrices.
>
> Replace this section by a comparison of the complexity of the power method vs the eigen
> method in terms of the approximation of the stationary distribution.
>
> These hanging methods should be avoided, aither use inline, or define methods of the
> existing classes.
>
> This method could be defined over any optyx diagram without feedback loops, finding the
> eigenvectors of an operator.
>
> Isn't this in optyx Ty already?
>
> The trace can be obtained diagrammatically as the Discard, why do we evaluate it
> separately from the rest of the tensor contraction machinery? We should hand off the
> contraction to the backend and formulate everything diagrammatically
>
> This seems like part of a method about the stationary state and needs to be simplified

## Review round: diagrammatic formulation and a two-part example

- [ ] `docs/examples/fixpoints.rst` part 1, boson sampling with feedback: state the
      Biriukov–Dyakonov setup and say which of its pieces `power` and `eigen` implement
- [ ] part 2, learning fixpoints: a beam splitter `2 -> 2` with one photon into the
      first input at every step and the second output fed back into the second input;
      report whether the fixed point converges and how big its density matrix is
- [ ] part 2, tuning: adjust the two beam-splitter phases towards fixed points that
      converge faster and have lower-dimensional density matrices
- [ ] replace the backend-comparison section with a complexity comparison of `power`
      against `eigen` as approximations of the stationary distribution
- [ ] drop `doubled_dimensions`: `Ty` already carries its doubled dimensions, so read
      the cutoff convention off `Ty.double()` rather than reimplementing it
- [ ] `density_trace`: build the trace as a `Discard` diagram and hand it to the
      backend, instead of `to_tensor` followed by a hand-written `np.tensordot`
- [ ] `stationary_vector`: define it over any optyx diagram without feedback loops, as
      the eigenvectors of the operator the diagram denotes, not of a superoperator array
- [ ] no hanging module-level functions in `channel.py`: fold `frobenius_distance`,
      `normalise_density_matrix` and the rest into methods of the existing classes,
      or inline them at their single call site
- [ ] simplify the normalisation, which reads as part of the stationary-state method

Cross-PR note: the tuning checkbox wants gradients through a contraction, which is
exactly the differentiable half of #21. Landing #21 first makes it a few lines; without
it, part 2 falls back to a parameter sweep.

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.
