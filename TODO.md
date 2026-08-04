# TODO

> In the fixpoint PR notebook, we need more test of L, M and the eta, gamma bounds to understand their dependence, if we can have an analytic formula that would be great, let's at least plot the dependence. Also I can see an optyx-pr15-body.md file here, and a fixpoint-learning.png online, these should not be part of the PR but please report the current state of the PR in the md file via message

## Dependence of the convergence certificates

- [x] derive every analytic dependence available
      for the Dobrushin coefficient `eta`, the loss parameter `gamma`, and the
      exact asymptotic rate over `L` loop modes and `M` external modes.
- [x] extend the executed notebook with a
      systematic seeded sweep and concise plots over `L`, `M` and `gamma`,
      comparing the `eta` and loss certificates with the exact rate.
- [x] remove `fixpoint-learning.png` from the PR,
      confirm the temporary PR-body Markdown is untracked, and report the live
      PR state in a concise GitHub message instead of committing that report.
- [x] execute the notebook, run the repository
      checks, update the PR description, push the round and verify green CI.

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
- [x] the wider question — should `CMap` be the underlying data structure of optyx,
      rather than a route bolted onto `power`? — is #23, since swaps are an artefact of
      the sequential representation and every contraction consumer pays for them

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
- [x] measure `|lambda_2|` directly for Haar `U` at small `L` and `M`, and check the
      unrolling: `|l1| = 1` exactly, `n*` from the population gap hits the 1e-4 target and
      doubling the depth squares the error. Two findings — the relevant rate is the
      **population-sector** eigenvalue (a Fock-diagonal start never excites the coherence
      sector), matching the measured decay to four digits, while the full `|lambda_2|` is a
      conservative bound; and the `s1^2` guess is **refuted**, disagreeing in every Haar
      case, so the gap must be measured rather than read off `U`

> It's also good if superoperator norm is an upper bound, can it be computed efficiently?
> The goal is to predict a big enough unroll time so that the truncation is a good
> approximation of the limiting process. Even better if we can show it is polynomial in
> L and/or M.

## A certified unroll depth

- [x] use the trace-norm contraction coefficient instead of `|lambda_2|`: because the
      dynamics stays Fock-diagonal this is the classical Dobrushin coefficient of the
      population chain, submultiplicative, so it bounds every power at once
- [x] verify `eta >= |lambda_2|` and that `n*(eta)` clears the target: measured
      2.4e-16, 1.3e-7, 2.6e-16 against a 1e-4 goal, so the depth is a certificate
- [x] cost it: `O(h)` channel applications plus `O(h^2)` arithmetic, against `eigen`'s
      `O(h^2)` applications and `O(h^6)` solve — the bound is a factor `h` cheaper than
      the answer, so a depth can be certified where `eigen` is out of reach
- [x] scaling: polynomial in `M` at fixed `L`, exponential in `L` since `h ~ (nM)^L/L!`
- [x] state the analysis for the general channel `Phi_{U,psi,L,M}`, not the beam splitter:
      Fock-diagonality holds for any `L`, `M` by number conservation, so the Dobrushin
      reduction is general; the beam splitter is the `L = M = 1` special case
- [x] reference the mathematics: collision models (Ciccarello et al. 2022), trace-norm
      contractivity (Perez-Garcia, Wolf, Petz, Ruskai 2006), the non-normality caveat on
      `|lambda_2|` (Szehr, Reeb, Wolf 2015), the ergodicity coefficient (Dobrushin 1956),
      boson sampling (Aaronson, Arkhipov 2011)
- [x] define every quantity from first principles in the notebook: the blocks of `U`, the
      loop block `D` and its singular values, the step channel, the loop chain `P`, `h`,
      `|lambda_2|` as the second-largest eigenvalue modulus of `P`, `eta`, `n*`
- [x] classical Dobrushin condition on the loop chain: holds (`eta < 1`) in 100% of Haar
      draws at `L = 1`, fails at one step in 100% of draws at `L = 2` where `eta`
      saturates at 1 — so a `k`-step coefficient is needed there
- [x] Haar ensemble answering the conjecture: growing `M` at `L = 1` drives the mean gap
      down (0.50, 0.39, 0.28) and the worst case with it (0.94, 0.78, 0.57), so slow
      mixing gets *less* likely as `M` grows; growing `L` reverses it (mean 0.75 at
      `L = 2`). The conjecture turns on how `L` scales with `M`
- [ ] open: the quantum Dobrushin condition of Bakshi, Liu, Moitra and Tang
      (arXiv:2510.08542) is **not** checked — arxiv is unreachable from this environment,
      so its precise statement is unknown. Guess to verify: it becomes a bound on the
      off-diagonal strength of `D`, polynomial in `L` and `M`
- [ ] open: `k`-step Dobrushin coefficient `eta(P^k)` for `L >= 2`, giving a certificate
      at effective rate `eta(P^k)^(1/k)`

> Instead of asking if the gap grows as we grow U, let's see if at given U if we add some
> noise to the system in the form of a photon loss channel for each feedback loop. Can we
> then ensure that lambda_2 is far from 1 and we can estimate the lossy circuit efficiently

## Loss gives a U-independent gap

- [x] add a pure-loss channel of transmissivity `gamma` on each feedback mode; on the
      population chain this is exactly binomial thinning, `P_gamma = P B`
- [x] measure: `|lambda_2| <= gamma` in every configuration sampled (`M = 1,2,3`,
      `L = 1,2`, `gamma = 0.95, 0.8, 0.5`), and not tight — worst draw about `0.8 gamma`
- [x] consequence: `n* <= log(eps)/log(gamma)`, depending only on the loss rate and the
      target accuracy, not on `U`, `M` or `L`. 270 steps at `gamma = 0.95` for `1e-6`,
      62 at `0.8`, 20 at `0.5` — worst case over all `U` simultaneously
- [x] consequence for the reduction: with `gamma < 1` fixed, `n*` is constant in `M` and
      `L`, so the unrolling has `O(M log(1/eps))` modes and photons and lossy stationary
      boson sampling reduces to one-shot boson sampling with polynomial overhead. The
      reduction question only bites in the lossless limit
- [x] prove `|lambda_2| <= gamma` rather than sampling it — done in the section "Why
      |lambda_2| <= gamma", see the round below

> We should calculate n star in terms of gamma in the lossy case, so for each noise level we
> determine the unroll length needed to reach a given accuracy epsilon
>
> Squash down the explanation that L impacts the complexity more than M, the notebook
> repeats that statement too many times

## n*(gamma, eps) and one statement of the L/M asymmetry

- [x] closed form `n*(gamma, eps) = ceil(log eps / log gamma)`, with the small-loss form
      `n* ~ ln(1/eps)/delta` for `delta = 1 - gamma`: depth inverse in the loss, only
      logarithmic in the accuracy
- [x] table of certified against observed depth over `gamma` and `eps`, plus a figure of
      `n*` against round-trip loss. At `gamma = 0.99` the certificate asks 1375 steps for
      `1e-6` while Haar draws converge by 221 — the slack is in the premise, since
      `|lambda_2| <= gamma` allows a `U` that contributes no mixing itself
- [x] say the `L` versus `M` asymmetry once, in the cost/certificate section, and let the
      later sections inherit it rather than restate it

> You should also check which bound is better in practice, the loss bound or the one
> based on eta?

## Loss bound against the Dobrushin bound

- [x] measure both certificates on the same lossy chains at `eps = 1e-6`, alongside the
      depth actually needed, over `gamma` and over `(M, L)`
- [x] verdict splits on `L`. At `L = 1` the thinned chain has `eta ~ 0.5` even at
      `gamma = 0.99`, so `n*(eta) = 21` against `n*(loss) = 1375`, within a small factor
      of the observed 7 — `eta` sees the mixing `U` contributes, the loss bound throws it
      away. At `L = 2` `eta` saturates at 1 and its certificate blows up (8e8 steps at
      `gamma = 0.99`) while the loss bound is unchanged
- [x] state the prescription: `n* = min(n*(eta), n*(gamma))`, both being valid at once,
      with the loss term keeping the minimum finite when `eta` saturates

> Prove the triangularity claim for |lambda_2| <= gamma

## Proof of the loss bound

- [x] Step 1, triangularity: `phi_r(n) = prod_i (n_i)_{r_i}` is the normally ordered
      `prod_i a_i^{dag r_i} a_i^{r_i}`, a passive `U` neither raises nor lowers
      normally ordered degree, and the Fock input kills every unbalanced term, so
      `P phi_r` is a polynomial of degree at most `|r|`. The leading part is the
      substitution through the loop block `D`
- [x] Step 2: thinning has the `phi_r` as eigenvectors with eigenvalue `gamma^{|r|}`,
      so `P_gamma = P B` preserves the same filtration with graded blocks `gamma^d A_d`
- [x] Step 3, `rho(A_d) <= 1`: the photon budget `N_{t+1} <= N_t + M` bounds
      `P^n phi_r` by `(|m| + nM)_d` pointwise, Newton's forward differences turn that
      into `||A_d^n|| = O(n^d)`, hence spectral radius at most one. Also gives the
      finite-`n` statement `||P_gamma^n|| = O(n^d gamma^{dn})` on `V_d`, so the
      non-normal transient is only a polynomial prefactor
- [x] verify each step numerically: `A_0 = 1` exactly, `rho(A_d)` below one and
      decreasing in `d`, and `spec(P_gamma)` matching `max_d gamma^d rho(A_d)`
- [x] sharpen to the exact rate: `A_1 = |D|^2` entrywise, confirmed to 1e-15, so
      `|lambda_2| = gamma rho(|D|^2)` — an `O(L^3)` computation, polynomial in `L` and
      `M`, and the correct replacement for the refuted `s_1^2` guess (`s_1^2 = 1`
      whenever `M < L`, so that guess was vacuous where it mattered)
- [x] record the two caveats: the proof is for the exact chain on `N^L` while the
      matrices are Fock-truncated, and the same grading on the full operator algebra
      only gives `sqrt(gamma)` once coherences are included

> This todo can be removed
>
> This requires some serious rethinking of how we build unrolling, should we switch to
> CMap entirely for the underlying datastructure of optyx? If so write an issue about this
>
> Why are these methods of Ty? Let's integrate them where they are actually used
>
> Why does the method need to check hermiticity and positivity? Draw the composition in
> the example
>
> Can't we use an existing contract method?
>
> These hanging functions need to be justified

## Review round: contraction reuse and where the helpers live

- [x] drop the "Methods of the ref, as tensor contractions" section: the four pieces it
      separates are described in the notebook, and the open items it carried are either
      done or tracked elsewhere
- [x] open #23 on whether `CMap` should be optyx's underlying data structure
- [x] `Ty.density_trace` and `Ty.normalise_density_matrix` were about stationary states,
      not about types. Moved to `channel.Diagram` as `_discard_trace` and
      `_validated_state`, next to `stationary_state` and `power_fix`, their only callers.
      Fully inlining them instead trips pylint's branch and statement limits and inline
      disables are banned here, so they stay as private methods on the calling class
- [x] justify the Hermiticity and positivity checks: `stationary_state` accepts any
      loop-free endomorphism, not only a channel, and trace preservation — already
      checked diagrammatically — does not imply positivity.
      `test_stationary_state_validates_density_matrix` exhibits a trace-preserving map
      whose fixed point is neither Hermitian nor positive
- [x] draw the fixed-point equation and the `Discard` trace in the `stationary_state`
      example, saved to `docs/_static/stationary_state.svg`
- [x] use the existing contraction path: `QuimbBackend.eval` accepted `**extra` and
      dropped it, so `fix` had to copy the backend and mutate `contraction_params`.
      `eval` now merges `**extra` over `contraction_params` for the call, `contract`
      becomes `self.at_time(steps).eval(backend, max_bond=bond)`, and the `copy` import
      goes. `RecordingBackend` in the tests records the merged parameters
- [x] justify the three closures in `power_fix`: `contract` memoises into a cache that
      must not outlive one call, and all three read `backend`, `compressed`, `tol`,
      `n_steps` and `max_steps`; as methods they would take those as arguments and the
      cache would have to live on an immutable diagram

> Why are these hidden methods? Can we make them more general methods about any diagram so
> that "state" is not an input. For example, checking if the state is normalised corresponds
> to checking if the process is causal, i.e. f >> Discard = Discard. A process can also be
> rescaled by a constant simply by tensoring by the scalar. Algorithms should be expressed
> diagrammatically
>
> this looks like it should be in optyx Tys
>
> Why do we need this check? Seems expensive
>
> Overall this method seems expensive to run so it shouldn't be used during evaluation
>
> Why is this a hidden method? Remember DisCoPy's guidelines in AGENTS.md
>
> Shouldn't this be handled in the backends? Consider improving #21
>
> That's a cool concept, is the boundary only the first input and last output memory or does
> it also include the inputs/outputs occuring at every time step? Can you point to some
> references where this concept can be grounded?
>
> We should have meaningful default values for these parameters, based also on the progress
> shown in the notebook. n_steps and chi can be determined to match a given tol with a given
> loss rate. We need a better organisation of this method, one mode should be doing the naive
> approach and checking if successive contractions converge, another mode should be guessing
> n_steps and chi with some heuristics on the size of the memory and convergence behaviour
> (with cost prediction before contraction), when the parameters are set they are used to skip
> some parts of the above pipelines.
>
> Why do we need to set these parameters? Aren't they the same as chi and n_steps?

## Review round: no secrets, causality, and defaults from the notebook

STYLE.md says optyx has no secrets and exposes every subprocedure as a method that can be
tested and reused, so the `_`-prefixed helpers of the previous round were the wrong shape.

- [x] `Ty.double_axes` and `Ty.dagger_axes`: the classical/ket-bra bookkeeping is type data,
      so it lives on `Ty` with its own doctests
- [x] `Diagram.is_causal(dimensions, tol)`: a process is causal when discarding its outputs
      discards its inputs, `f >> Discard(cod) == Discard(dom)`. No `state` argument, general
      over any diagram, and for a state it is exactly normalisation. Rescaling is tensoring
      with a scalar, which is what the doctest breaks causality with
- [x] `Diagram.discard_trace`: the array of `self >> Discard(cod)` — the trace of a state and
      the causality witness of a process are the same composition
- [x] `Diagram.is_hermitian` and `Diagram.is_positive` are public and are **no longer run
      during evaluation**. They cost a diagonalisation, as much as the eigensolve which
      produced the state, so `stationary_state` normalises and returns, and a caller checks
      the result. `is_positive` diagonalises the Hermitian part and says so
- [x] `backends.QuimbBackend.process_term` is no longer semiprivate
- [x] trade-off recorded: a causal endomorphism which is not a channel now returns a fixed
      point that is no density matrix, silently.
      `test_stationary_state_is_not_a_density_matrix` pins that behaviour and checks both
      predicates instead of expecting a raise
- [x] `FeedbackBoundary`: the boundary is the memory wire only — `initial_state` before the
      first step, `final_effect` after the last — while the per-step inputs and outputs stay
      open as the stream's domain and codomain at each tick. Grounded in Di Lavore, de Felice
      and Roman, *Monoidal Streams for Dataflow Programming* (LICS 2022), which is what
      `discopy.monoidal.Stream` implements, and contrasted with the trace of Katis, Sabadini
      and Walters (2002): `mem` is a delay, so the loop unrolls rather than closing on a
      fixed point

- [x] superseded by the round below, which USER specified in full
- [ ] normalisation in the backends rather than in `fix` (#21): the trace is a `Discard`
      composition, so a backend which evaluates the traced diagram alongside the state
      returns a normalised result with no second pass

> This method is likely used to compute a normalisation scalar for a channel given a state.
> So let's have a method `normalisation(self, state=None)` which computes the scalar
> `state >> self >> Discard(self.cod)`, and returns an error in the case where `state is None`
> and `self.dom is not Ty()`.
>
> Yes I agree to focus on n_steps, we need a separate method for the estimation of n_steps
> given a diagram and a loss threshold, or other parameter. Similarly, for approximate
> contraction we should have a separate method to handle chi. I would say that the inputs of
> the method are n_steps=None, chi=None, method="power", tol=1e-6, loss=0. If method is eigen
> then chi can be interpreted as the max dimension of the truncation and n_steps is not
> needed. If no chi is given, then it should be inferred by a separate method. If the method
> is power, the tensor network for n_steps with max bond dim chi is implemented and executed,
> the exact tensor if chi is not given. If n_steps is not given, then the method should be our
> most efficient and guaranteed way of estimating the fixpoint given tol and loss.
>
> Since one is dead when the other isn't, n_steps and chi can be interpreted as max_steps and
> max_chi when needed
>
> I'm pretty sure you can collapse these two input params into one.
>
> Don't make the test suite too expensive of course

## Review round: one depth, one truncation dimension

- [x] `Diagram.normalisation(state=None)`: the scalar `state >> self >> Discard(cod)`,
      raising when `state` is absent and `dom` is not `Ty()`. Replaces `discard_trace`
- [x] `fix(n_steps=None, chi=None, *, method="power", tol=1e-6, loss=0, backend=None)`.
      `cutoff`, `max_steps` and `max_chi` are gone: there is one truncation dimension and
      one depth, used where they apply and serving as the caps where they are searched for
- [x] `chi` is the truncation dimension throughout — the bond bound for `power`, the memory
      dimension for `eigen`. `power` contracts *exactly* when it is absent
- [x] `Diagram.unroll_depth(tol, loss)`: `n* = ceil(log tol / log(1 - loss))` from the
      notebook's proof, a depth rather than a guess and independent of the memory size.
      Raises for `loss = 0`, where no finite depth is guaranteed
- [x] `Diagram.truncation_dimension(tol)`: double from two until the truncated transfer map
      is causal, which is the condition `stationary_state` already refuses to violate.
      Supplies `eigen`'s `chi` when it is not given
- [x] `power_fix`: use the depth when `n_steps` or a positive `loss` is given, otherwise
      double until successive contractions agree, capped by the public `MAX_UNROLL`. The
      `chi` doubling is gone with `max_chi`
- [x] keep the suite cheap: `test_power_does_not_alias_period_two` never converges by
      construction, so at the new cap it took 153s on its own. It monkeypatches
      `MAX_UNROLL` to 8, which is what `max_steps=8` used to do. `test_fix.py` runs in 10s,
      down from 32s before this round
- [ ] open: `backend` is still a parameter of `fix`, though USER's list did not name it. It
      is orthogonal to the search knobs and three tests use it, so it stayed — confirm

## Blocked on design

- `dom != Ty()`: the process "prepare an input state at every time step, return the
  final output" is nonlinear in the input, so its one-shot approximation as a CPMap
  from `dom` to `cod` needs a design choice — waiting for USER / @armandld before
  filing checkboxes.


> I need a proper review on https://github.com/rel-int/optyx/pull/15. My idea would be to split this PR in two. Basically, one part needs to go into the previous PR on feedback and stream semantics. Particularly the Diagram.stream and Diagram.now methods. And the treatment of feedback boundaries. The latter requires some more thought. Do you have any ideas for incorporating this concept, e.g. we could have (top, bot) instead of initial state and final effect, and have them as boundary.top, boundary.bot. We also need a different way of standardising the boundary which should make some of the code simpler. Write a plan of action in the TODO.md

> Actually the proper way to do this is to have 3 prs in total: 1) Feedback and unroll, 2) Streams and boundaries, 3) Fixpoint semantics

> Let' think properly about this Boundary class. Do we actually need it? Aren't the top and bottom boundaries properties of Feedback bubbles? When do we actually need to call the class?

> I'm asking to rethink the plan to see if we can remove step 2 in the story, since stream is only needed to unroll and boundaries come with feedback

> We should have handles in unroll to control the boundary. We should call the parameters state and effect, instead of top and bot. The boundary is always defined when creating feedback bubbles, but the default values should be: for photonic: empty initial state, discard at the end, for qubits: there is no natural choice of state so give the identity (open wire) by default, final effect is the discard by default. Setting any of these parameters to None is interpreted as keepin the wire open, setting state or effect =None in unroll should be interpreted as overriding the bubble boundary information and leaving it open. This way unroll remains flexible as a method, you can always decide on the initial state/final effect on the spot but you need to know yourself its dimension, or give None to keep it open. In this setting we can define now as unroll(0, state=None, effect=None). Do we then actually need stream() anywhere?

> good proposal for unroll, should be documented that None opens the input/output memory.  Given your comments, I've decided that only the effect should be fixed for channels, in all other cases the parameters should be given explicitly or will be interpreted as open wires. This resolves the two middle points. For the last point, I think we should follow the discopy convention on indexing, so that should also be added to this PR.

> So state and effect are interpreted as open wires as default except the effect for channel.Diagrams

> Explain better why you think the stream method should remain? I don't see how we would use it outside unrolling. For now I see the point of getting eacy access to the "feedback normal form". But now I'm thinking that one_step() was a better name if there's no stream anymore.

## Split into two PRs

| PR | Scope | Base |
| --- | --- | --- |
| 1 — The feedback category (#12) | `feedback(dom, cod, mem, state, effect)`, `unroll(n, state, effect)`, `one_step()` | `main` |
| 2 — Fixpoint semantics (#15) | `fix`, `power`, `eigen`, `stationary_state`, backends, notebook | PR 1 |

- [ ] order the rebases 1 → 2, rebasing PR 2 on PR 1's head at every round

## `stream()` is needed nowhere

With boundary handles on `unroll`, every consumer of the stream goes through `unroll`:

| was | becomes |
| --- | --- |
| `now()` | `one_step()`, a one-line alias for `unroll(0, state=None, effect=None)` |
| `truncation_dimension`, `eigen_fix` | the same call; they then split `step.cod[len(self.cod):]` as today |
| `at_time(n)` | deleted; `power_fix` composes on `unroll(n, effect=None)` |
| `unroll(n)` | keeps the functor inline, exactly as #12 shipped |

So `stream()`, `core.diagram.Stream` and `core.diagram.FeedbackBoundary` are all deleted, and
PR 1's public surface is `feedback` and `unroll` — #12's shipped surface — plus `one_step`
as sugar. The per-loop boundary list never leaves `unroll`, so the aggregate `(state, effect)`
is the only boundary anyone sees.

Why the open call gives what `eigen` needs: with both handles open the plugs are identities,
so `unroll(1, state=None, effect=None)` is `stream.now` on the nose — the one-step map from
`dom @ mem` to `cod @ mem` with the memory trailing, which is all
`truncation_dimension` and `eigen_fix` ever read.

- [ ] delete `stream()`, `Stream` and `FeedbackBoundary`
- [ ] delete `at_time`


## The boundary handles

`initial_state` and `final_effect` become `state` and `effect`, ordinary attributes of the
`Feedback` bubble and keyword parameters of both `feedback` and `unroll`.

**`None` is an input spelling, not a stored value.** It resolves at construction to
`Diagram.id(mem)` — an open memory wire *is* the identity on `mem` — so `Feedback.state` and
`Feedback.effect` are always diagrams. That is what removes the `None` branching at ten
sites, each currently spelling `None if x is None else f(x)` twice:

```python
def double(self):
    """The feedback loop of the doubled diagram."""
    return Diagram.double(self.arg).feedback(
        mem=self.mem.double(),
        state=Diagram.double(self.state), effect=Diagram.double(self.effect))
```

**Defaults: open wires everywhere, except the effect of a `channel.Diagram`.**

| | `state` | `effect` |
| --- | --- | --- |
| `core.diagram.Diagram` | `Id(mem)` | `Id(mem)` |
| `channel.Diagram` | `Id(mem)` | `Discard(mem)` |

There is no vacuum default, so nothing reaches from `channel` up into `photonic` and
classical `mode` needs no vacuum of its own. A forgotten initial state stays an open wire and
fails loudly downstream rather than silently answering from vacuum. `Discard` is the one
default that is unambiguous and it exists only in the CPM layer, which is exactly where it is
applied.

- [ ] `feedback(dom=None, cod=None, mem=None, state=None, effect=None)`, each `None`
      resolving to `Id(mem)` except `channel`'s `effect`, which resolves to `Discard(mem)`
- [ ] validate `state.cod == mem` and `effect.dom == mem` in `Feedback.__init__` — the check
      that does not exist today, so the two plug-size cases of `test_feedback_axioms` move
      from "raises at `unroll`" to "raises at `feedback`"
- [ ] `unroll(n_steps=1, state=..., effect=...)`, where `...` uses the bubble's boundary,
      `None` overrides it to an open memory wire, and a diagram overrides it to that diagram.
      Document that `None` opens the input or output memory, and that an override applies to
      the aggregate memory, so the caller supplies its dimension
- [ ] `one_step()` as `unroll(0, state=None, effect=None)`, returning the feedback normal
      form: every `Feedback` bubble eliminated and the memory moved to the boundary, from
      `dom @ mem` to `cod @ mem`. Named `one_step` and not `now` because `now` is DisCoPy's
      name for a *field of a* `Stream`, and with no stream left in optyx it would borrow from
      a structure the reader cannot see
- [ ] `__str__` and `__repr__` print `state` and `effect` only when they differ from the
      default for their `mem`, keeping `eval(repr(x)) == x`
- [ ] the structural transports — `conjugate`, `inflate` twice, `double`, `get_kraus`, both
      `Functor.__call__` — become single expressions, and `is_pure` becomes
      `all(part.is_pure for part in (self.arg, self.state, self.effect))`
- [ ] `unroll(n).dom == dom ** (n + 1) @ state.dom`, empty for a genuine state and `mem` for
      an open wire, so "open wires gathered at the end of the domain in loop order" stops
      being prose
- [ ] the "Stateful channels" module docstring without its "Fixpoints." paragraph, with
      `feedback.png` and `unroll.png`, carrying the Di Lavore, de Felice and Roman (LICS
      2022) and Katis, Sabadini and Walters (2002) grounding out of the deleted
      `FeedbackBoundary` docstring

## Follow DisCoPy's indexing

`Diagram.unroll(n_steps)` currently means `n_steps` **time steps** and computes
`stream.unroll(n_steps - 1).now`, refusing `n_steps < 1`. DisCoPy's `Stream.unroll` is
`@inductive`: `n_steps` is the number of **unrollings**, and `unroll(0)` is the stream
itself. Adopting it makes `unroll(0)` one time step, which is what lets `one_step` be a call
rather than a method with its own machinery.

| call | time steps |
| --- | --- |
| `unroll(0)` | 1 |
| `unroll(1)` — the default | 2 |
| `unroll(n)` | n + 1 |

- [ ] `unroll(n_steps=1)` becomes `stream.unroll(n_steps).now`, refusing `n_steps < 0`;
      `test_unroll_open_wires`'s `unroll(0)` raises becomes `unroll(-1)` raises
- [ ] shift every call site by one: the `unroll` and `Feedback` doctests, the CNOT ladder
      that regenerates `docs/_static/cnot_ladder.svg`, and `test_feedback.py` throughout
- [ ] state in the docstring that `n_steps` counts unrollings and not time steps, since the
      name invites the other reading, and that `unroll()` with no argument now covers two
      time steps rather than one

## PR 2: fixpoint semantics

Everything else in the current diff, unchanged in substance: `normalisation`, `is_causal`,
`is_hermitian`, `is_positive`, `stationary_state`, `unroll_depth`, `truncation_dimension`,
`fix`, `power_fix`, `eigen_fix`, `Ty.double_axes`, `Ty.dagger_axes`, `MAX_UNROLL`,
`MAX_TRUNCATION`, `backends.py`, the `misc.py` deletion, `test_fix.py`, the notebook,
`docs/conf.py`, `docs/notebooks.rst`, the workflow and the "Fixpoints." paragraph.

`at_time` is `unroll` plus one composition. Both call the identical stream unrolling, and
with `dom == Ty()` their initial plug is the same expression, so for `n` time steps and
`effect == Discard(mem)`:

```
unroll(n - 1) >> Discard(cod ** (n - 1)) @ Id(cod)
  = initial >> unrolled >> (Id(cod ** n) @ Discard(mem))
             >> (Discard(cod ** (n - 1)) @ Id(cod))
  = initial >> unrolled >> (Discard(cod ** (n - 1)) @ Id(cod) @ Discard(mem))
  = at_time(n)
```

by interchange, the two composites acting on disjoint wires. `power_fix` should not rely on
the loop's `effect` being a discard, since a caller may have set another one — conditioning
is a different computation — so it opens the memory and discards it itself.

**The off-by-one seam.** `fix(n_steps)`, `unroll_depth(tol, loss)` and `MAX_UNROLL` are all
depths in **time steps**: `unroll_depth` returns a mixing time, `ceil(log tol / log(1 -
loss))`. They stay in time steps, and the single conversion to unrollings lives in
`power_fix`.

- [ ] `power_fix`'s `contract` closure becomes `self.unroll(steps - 1, effect=None)`, then
      `Discard(self.cod ** (steps - 1)) @ self.id(self.cod) @ Discard(memory)` with
      `memory = step.cod[len(self.cod) * steps:]`
- [ ] keep every public depth in time steps and convert exactly once, in `contract`; state
      the seam in `power_fix`'s docstring so the next reader does not have to derive it
- [ ] `truncation_dimension` and `eigen_fix` replace `self.now()` with `self.one_step()`;
      neither reads anything but the open one-step map
- [ ] carry `at_time`'s three guards into `power_fix`: empty domain, positive integer depth,
      and every loop's `state` a state — now one check, `state.dom == Ty()`, on the aggregate
      rather than a scan over loops. With no vacuum default this guard genuinely bites, which
      is the intent
- [ ] `steps == 1` still needs its branch: `Discard(Ty())` is a box and not an identity, and
      `power_fix` reaches `contract(depth - 1)` with `depth == 2` on the hot path
- [ ] `test_fix.py` asserts diagram equality in `conditioned.at_time(2) ==
      plain.at_time(2)`; the composed form puts the memory discard in its own layer, so
      restate it as an evaluation comparison or accept the new layer order
- [ ] note where the rewrite happens that `Discard(m0 @ m1) != Discard(m0) @ Discard(m1)` as
      terms; irrelevant to every diagram `fix` is called on, all of which are single-loop
- [ ] rename `initial_state=` to `state=` in the `test_fix.py` fixtures and the notebook, and
      drop the now-redundant `final_effect=Discard(...)` where it matches the default
- [ ] reword `stationary_state`'s guard, whose message names `now`, to name `one_step`, and
      its test matching `"without feedback"`

> What is the difference between eigen and stationary_state? Could they also be merged into one?

> This edit is becoming more and more important. If we can restrict to feedback, unroll, one_step (first PR) and at_time, eigen_fix, power_fix, fix (second PR) it would be great. At_time seemed useful to me as a shortcut, although it can be defined with discard and unroll. I want to avoid to_stream so we don't need to define our own stream class and the semantics of channel remains external to the package. I think stationary_state can be included in the eigen_fix method if we don't use it elsewhere. The only thing that would be worth keeping separate is the spectral decomposition but it would have to be generalised properly to arbitrary channels if so. I don't like these methods that work only for certain diagrams. Edit the PR description making a list of the proposed methods that remain and those that are removed by the restructuring of fix PR.

## The surface after the restructuring

Seven methods survive, from sixteen public additions. PR 1 keeps `feedback`, `unroll` and
`one_step`; PR 2 keeps `at_time`, `fix`, `power_fix` and `eigen_fix`. `at_time` stays as the
shortcut even though `unroll` and `Discard` define it. Listed in the description of #26.

Call graph checked before deciding: `is_hermitian` and `is_positive` have **zero** callers in
the package — `stationary_state` deliberately stopped running them and nothing picked them up
— and `Ty.double_axes` / `Ty.dagger_axes` exist only to serve those two. `is_causal` has one
caller, `truncation_dimension`, which is itself absorbed. `unroll_depth` and
`truncation_dimension` have one caller each. `stationary_state` has one caller in code,
`eigen_fix`, though five tests, a doctest and a notebook section call it directly.

- [ ] delete `Stream`, `FeedbackBoundary` and `stream()`
- [ ] absorb `stationary_state` and `truncation_dimension` into `eigen_fix`, and
      `MAX_TRUNCATION` into a local cap there; fix `truncation_dimension` while absorbing it,
      since it always returns two today — it measures the unprojected map while
      `stationary_state` refuses the projected one
- [ ] absorb `unroll_depth` into `power_fix`; it is arithmetic in `tol` and `loss` that never
      touches the diagram
- [ ] delete `is_causal`, `is_hermitian`, `is_positive`, `Ty.double_axes` and
      `Ty.dagger_axes`, with the tests that only covered them
- [ ] keep `at_time`, defined as `unroll(n - 1, effect=None)` composed with the discards
      rather than re-implementing the plumbing

### Two open points

- [ ] `normalisation(state=None, dimensions=None)` is not among the seven and I would keep
      it: general over any diagram, specified in review thread `r3698759394`, and with two
      surviving callers — the trace in `power_fix` and in the absorbed `stationary_state`.
      Inlining duplicates a diagrammatic contraction that is not a plain trace on mixed
      classical and quantum types. Its `state=` array parameter is the wart, and #28 removes
      it. Confirm or strike
- [ ] if the spectral decomposition is kept, it should be `spectrum(dimensions)` — the
      eigenvalues and eigenvectors of the doubled operator and nothing else. It passes the
      generality test: `dom == cod` is the definition of an eigenvalue problem, not a
      restriction on which diagrams work. It also gives `|lambda_2|` directly, which the
      notebook currently only derives analytically. The causality residual, the
      no-fixed-point and non-uniqueness guards and the normalisation are fixed-point
      concerns and move to `eigen_fix`
