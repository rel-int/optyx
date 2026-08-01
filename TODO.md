# TODO

> The notebook should start from the single photon in a beam splitter experiment with feedback and draw the plots of the stationary distribution and its tail. Then it should present the complecity of the two estimation modes, *but* in terms of the number of looped modes L and the number of input/output modes M, let's also assume that n = M, there is one photon input per mode, the unitary is arbitrary. We want to write the complexity of estimating power and eigen in terms of L, M and U only.

## Multimode complexity and notebook order

- [x] derive the power and eigen costs for an
      arbitrary `M`-mode unitary with `L` looped modes, one photon per input
      mode and `n = M`, using only `L`, `M` and the cost of applying `U`.
- [x] make the single-photon beam-splitter feedback
      experiment the notebook entry point, followed immediately by its
      stationary distribution and tail plots.
- [x] move the general boson-sampling construction
      after the motivating experiment and replace the single-mode complexity
      discussion with the multimode result.
- [x] execute the notebook, run focused validation,
      update the PR description and push the completed round.

> Yes, and I think the notebook can be pushed a bit further. The comparison of the complexity between power and eigen currently makes it seem that eigen is much more efficient, is there a regime when power gets more efficient? The max number of photons D is related to the the number of time steps n, so the complexities should be compared as functions of of the same variables. For the photon-adder protocol, it would be interesting to know also the higher moments of the distribution classical (distinguishable) vs quantum (indistinguishable).

## Common scaling and higher moments

- [x] derive the power/eigen costs from the
      same photon cutoff, time depth and target error, then identify and measure
      the regime where power is cheaper.
- [x] derive and validate higher factorial
      moments, skewness and kurtosis for distinguishable and indistinguishable
      photon-adder fixed points.
- [x] update and execute the notebook with the
      common-scaling comparison, higher-moment tables and concise figures.
- [x] remove stale duplicate checkboxes and map
      the superseded review threads to the executed notebook.

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

- [x] `docs/examples/fixpoints.rst` part 1, boson sampling with feedback: state the
      Biriukov–Dyakonov setup and say which of its pieces `power` and `eigen` implement
- [x] part 2, learning fixpoints: a beam splitter `2 -> 2` with one photon into the
      first input at every step and the second output fed back into the second input;
      report whether the fixed point converges and how big its density matrix is
- [x] part 2, tuning: adjust the two beam-splitter phases towards fixed points that
      converge faster and have lower-dimensional density matrices
- [x] replace the backend-comparison section with a complexity comparison of `power`
      against `eigen` as approximations of the stationary distribution
- [x] drop `doubled_dimensions`: `Ty` already carries its doubled dimensions, so read
      the cutoff convention off `Ty.double()` rather than reimplementing it
- [x] `density_trace`: build the trace as a `Discard` diagram and hand it to the
      backend, instead of `to_tensor` followed by a hand-written `np.tensordot`
- [x] `stationary_vector`: define it over any optyx diagram without feedback loops, as
      the eigenvectors of the operator the diagram denotes, not of a superoperator array
- [x] no hanging module-level functions in `channel.py`: fold `frobenius_distance`,
      `normalise_density_matrix` and the rest into methods of the existing classes,
      or inline them at their single call site
- [x] simplify the normalisation, which reads as part of the stationary-state method

Cross-PR note: the tuning checkbox wants gradients through a contraction, which is
exactly the differentiable half of #21. Landing #21 first makes it a few lines; without
it, part 2 falls back to a parameter sweep.

> It is not clear what we learn from this experiment. My impression is that we want to
> understand the density matrix fixpoint and how it is parametrized by theta and phi. Not
> necessarily a question of efficiency. How do we characterize the mixed Fock states that
> we get by feeding back with delay?
>
> The point of this paragraph should be to compute the unroll length at which the power
> method approximates the eigen method, and to compare the time it takes for the two
> methods to keep the same accuracy as the dimensions grow.

## Review round: characterise the fixed point, measure the crossover

Supersedes the "converge faster, to lower dimensional density matrices" framing of the
previous round: part 2 is about the physics of the fixed point, not efficiency.

- [x] part 2: characterise the
      fixed-point density matrix as a function of `theta` and `phi` — photon-number
      distribution, mean photon number, purity and effective Fock rank — instead of
      convergence speed
- [x] part 2: say what family of
      mixed Fock states delayed feedback produces, with the analytic anchors at
      full transmission and full reflection
- [x] complexity paragraph:
      compute the unroll length at which `power` reaches `eigen` within a stated
      tolerance, rather than tabulating errors at fixed depths
- [x] complexity paragraph: time both methods while growing the physically linked
      depth and cutoff, report both errors, and identify the asymptotic crossover

> CI flake: test_adaptive_defaults raises "Compressed contraction returned zero or
> non-finite trace" — 513487b passed and d2a6075 (TODO.md text only) failed.

## Fix the flaky trace guard

- [x] take the trace from the density matrix already contracted, or contract the
      `Discard`-composed diagram exactly: never bound its bonds, since a `max_bond`
      approximation of a trace is both wrong and non-deterministic (#19)
- [x] give the trace-validity floor its own constant: `abs(trace) <= tol` reuses the
      convergence tolerance, so `fix(tol=1e-4)` rejects any trace below `1e-4`
- [x] re-run the focused suite several times to confirm the flake is gone

> I added some comments on the repo, the documentation on fixpoints should be clearer,
> also I see you made an examples repo in the docs, what I want is for the fixpoint.rst to
> appear in the docs' examples along with the others linked there
>
> Again, I don't know what you're learning, clearly theta has an important influence on the
> state we get at the end. Are there values of theta for which the fixpoint has amplitudes
> over all photon numbers?
>
> Is the state in the limit a coherent state? does it have amplitudes for all photon numbers?
>
> The aim at the end is to characterise the amplitudes of the fixpoint of the beam splitter +
> photon setup analytically, so that we can check that the values we compute agree with the maths
>
> Spend less time showing the distances, add a paragraph comparing the complexity of the power
> and eigen methods.

## Review round: analytic fixed point, examples location

- [x] move `fixpoints.rst` next to the other examples in `docs/notebooks/`, drop the
      one-file `docs/examples/` directory and update the toctree and the `:doc:` cross
      reference in `channel.py`
- [x] state the analytic characterisation and check the computed values against it:
      `<n> = 1` exactly, and `Var(n) = 4T/(1+T)`, which is exactly **twice** the
      binomial-thinning variance `2T/(1+T)` of distinguishable particles — HOM bunching
      doubles the variance and leaves the mean alone
- [x] answer both questions in the text: the fixed point is Fock-diagonal, so it is **not**
      a coherent state (nor a phase-averaged one — at `T = 1/3` the variance is Poissonian
      but the distribution is not Poisson, L1 gap 0.15), and it has weight on **every**
      photon number for every `T > 0`, only collapsing to `|1>` at the full swap `T = 0`
- [x] say what `theta` does: `theta = 0` is the full swap, `0.25` is 50:50, `0.5` is the
      identity; the Mandel parameter is `Q = (3T-1)/(1+T)`, crossing zero at `T = 1/3`
- [x] pin down the `theta -> T` calibration: direct amplitude evaluation gives
      `T = |U00|^2 = sin^2(pi theta)`; distinguishing the loop from the emitted
      state resolves the earlier apparent factor of two
- [x] replace the distance tables with the complexity paragraph (duplicate of the earlier
      line-120 note)

## Docs are not verified

- [x] execute `docs/notebooks/fixpoints.ipynb` from a fresh kernel in the docs
      job before building Sphinx, so every numerical claim and assertion is checked

> Go in implementation mode and write this notebook in the same way as the others, make
> sure it gives pictures of the ideas.

## Notebook rewrite

- [x] rewrite `fixpoints.rst` as `docs/notebooks/fixpoints.ipynb`, executed, in the style
      of the other notebooks: 8 drawn figures (the loop, its unrolling, the semantic
      equation, the transmissivity curve, the stationary distributions, the Mandel curve)
- [x] part 1 boson sampling with feedback, part 2 the beam splitter fed one photon a step
- [x] state and check the analytic anchors: `<n> = 1` exactly, `Var = 4T/(1+T)`, exactly
      twice the distinguishable-particle variance — ratio measured as 2.000000
- [x] validate `fix` against an independent Fock-space transfer matrix: distributions
      agree to 8e-17
- [x] answer the questions in the text: Fock-diagonal so never a coherent state, weight on
      every photon number except at the full swap, and the sub/super-Poissonian crossover
- [x] distinguish the loop state from the emitted state that `fix` returns; they differ
      except at 50:50, which is why the closed form is stated for the loop

> Check out the PR https://github.com/rel-int/optyx/pull/16, they may have developed a more
> efficient way to unroll the stateful diagram into a tensor network for their simulations

## Adopt the combinatorial-map contraction for `power`

PR #16 does have one, in `optyx/core/contract.py::_cmap_to_quimb`. Instead of a sequential
`tensor.Diagram`, it builds a `discopy.tensor.CMap` — boxes plus a pairing of ports — and
translates it straight to a Quimb network by naming **one index per edge**. Wire routing
becomes index naming, so permutations never become tensors: open ports get an identity,
loops get an identity on a repeated index. Measured on the beam-splitter loop, our
`at_time(n).double().to_tensor()` route emits 33 explicit `Swap` tensors at `n = 6` out of
212 boxes, and the swap count grows roughly quadratically in `n` because the memory wire is
routed past every accumulated output wire. All of them disappear under the CMap route.

- [ ] build the unrolling of a feedback diagram as a `tensor.CMap` — boxes plus the port
      pairing given by the loop — instead of composing `unroll(n)` and permuting
- [ ] route `power` through `_cmap_to_quimb`, so cotengra optimises a hypergraph with no
      routing tensors in it; measure the contraction-time difference against the table in
      the notebook before claiming a win
- [ ] check how compressed contraction behaves without the swap tensors: bond truncation
      currently sees them as ordinary tensors
- [x] ordering: defer the CMap optimisation until #21 lands the contraction routine;
      `discopy.tensor.CMap` is available in the pinned revision, but changing the
      contraction representation is outside this correctness and documentation round

> Check that the new formulas are correct [...] Definitely the notebook should be improved
> to check the complexity claims in the implementation. The last part on characterising the
> distribution should be made much shorter and focus on stating why it is not poissonian,
> etc, I wonder how it is different from a Gaussian state.

## Review round: shorter characterisation, Gaussian comparison, measured complexity

Formulas verified first: `F_r^Q = r! F_r^C` with `F_r^C = r! T^{r(r-1)/2} / prod_j [j]_T`
reproduces the measured factorial moments to 1.00000 for `r = 1..4` at `T = 0.2, 0.4, 0.6`.
They are correct as written.

- [x] cut the characterisation section down: one statement of why the state is
      not Poissonian, not a tour
- [x] add the Gaussian comparison:
      the only Fock-diagonal single-mode Gaussian states are thermal, so `F_r = r!` at
      mean one; ours is `r! F_r^C`, which reaches `r!` only as `T -> 1`
- [x] measure the complexity claims rather than asserting them: fitted exponents are
      2.14 in `D` for eigen against a predicted 6, and 1.84 in `n` for power against a
      predicted 1. Both are explained in the notebook: the eigen solve is not yet the
      bottleneck at reachable cutoffs, and power is superlinear because unrolling widens
      the diagram as it deepens — the routing overhead the CMap contraction would remove

> If the complexity claims are contradicted by the measurements then we should reconsider
> these claims, whether they are true. Let's spend less time on the gaussian comparison. We
> need the following structure in the notebook: 1) one beam splitter, one photon per step,
> as is, 2) Complexity of boson sampling with feedback: presents the stationary boson
> sampling problem in terms of state psi, unitary, L and M, gives the complexity for
> evaluating it with eigen and with power methods. Checks that this is reflected in
> simulation time (up to constants). 3) Says in one brief paragraph what the distribution is
> not, and ends by asking whether stationary boson sampling can be reduced to one-shot boson
> sampling over polynomially many more modes/photons.

## Restructure into three parts

- [x] part 1 unchanged: one beam
      splitter, one photon per step, stationary distribution and tail
- [x] part 2: state the stationary
      boson-sampling problem in `psi`, `U`, `L`, `M`; give eigen and power costs and check
      them against measured time up to constants
- [x] correct the claims themselves: the `h^6` eigensolve never dominates at reachable
      sizes, so the honest cost is the `h^2` build — measured exponent 1.97 against a
      predicted 2. Power is not linear in `n` as implemented: measured 1.81. The network
      is *not* superlinear in size (boxes exponent 1.06); the swap count is (1.60), and
      exact contraction grows the intermediates with depth
- [x] part 3: one paragraph on what
      the distribution is not, ending with the reduction question to one-shot boson
      sampling over polynomially many more modes and photons
- [x] cut the Gaussian section to a
      clause and drop what the three-part plan does not need

> I have two important suggestions for improvement. First, let's keep only parts 1 and 2 and
> split the conclusion in the relevant parts. Second, the comparison of costs should include
> the space complexity and should make the first time steps to understanding the
> relationship between the expected time of convergence and U, M and L.

## Two parts, space complexity, convergence time

- [x] drop part 3: the "what it is not" paragraph moves into part 1, the reduction question
      becomes the closing section of part 2
- [x] give the cost comparison a space column: `power` holds one `h^2` density matrix,
      `eigen` materialises an `h^2 x h^2` transfer operator, so `h^4` — the square. Space,
      not the `h^6` solve, is what stops `eigen` first
- [x] first steps on convergence time: the error decays like `|lambda_2|^n`, so
      `n* = log(eps)/log|lambda_2|`. Measured `|lambda_2| = T` exactly for the one-mode
      loop, giving `n* = log(eps)/log(T)`
- [x] relate it to `U`, `M`, `L`: `|lambda_2| ~ s_1^2` with `s_1` the largest singular value
      of the `L x L` loop block of `U` while one photon dominates; `M` enters through how
      fast the loop fills. The many-photon regime is left open and named as the next
      experiment
- [ ] measure `|lambda_2|` directly for random `U` at small `L` and `M` — the open half of
      the convergence question, and the input the reduction question needs

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.
