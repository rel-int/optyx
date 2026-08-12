06-algorithm.md est déjà retiré. Peux tu m'aider à faire tout ça? Faisons un pull origin d'optyx et ouvrons une pull request pour faire des tests du SBS sur optyx directement.

une fois tout ça fait, je pusherai optyx et mes fichiers markdowns.

## Mathematical and implementation plan

The verification compares the finite-depth SBS certificate, the ordinary-BS reduction,
unitary-family scaling, and repeated Fock truncation with independent finite-dimensional
calculations. Passive unitaries and loop blocks are NumPy arrays; exact low-photon states
use an independent Fock-basis simulator; each mathematical claim becomes a deterministic
pytest assertion. Wiki-specific execution and local-workspace discovery do not belong here.

- [x] Port the independent Fock simulator and reduction/certificate checks into Optyx tests.
- [x] Port the Haar/structured-family and repeated-truncation checks into Optyx tests.
- [x] Remove wiki-only runners and expose one documented pytest entry point.
- [WIP] @codex-root-2026-08-11 20:38 Run `pflake8 optyx` and `coverage run -m pytest`.
- [ ] Prepare the Optyx PR and update wiki-content links to stable Optyx code.

Have you finished?

Have you checked that each file of the photonic folder would run on the wiki-content respecting its constrains?

Also I want you to do the tests and implement the algorithm (that estimate the feasibility of converging to a fixpoint) on optyx opening a PR on top of the PRs (12 and 15) of the fixpoints of giodefelice.

## Preflight implementation

The preflight works only with the one-step passive optical isometry. It computes the finite-depth
certificate from powers of its loop block, the stationary mean occupation from a discrete Lyapunov
equation, and sufficient/necessary cutoff bounds. A frozen result records one of three rigorous
verdicts without constructing a Fock state or starting a contraction.

- [WIP] @codex-root-2026-08-12 09:00 Add the three-way fixpoint preflight API and focused tests.
- [ ] Validate every photonic wiki-content page and the rendered experiment notebook.
- [ ] Run the complete Optyx checks and open a draft PR stacked on PR #15.
