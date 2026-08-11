06-algorithm.md est déjà retiré. Peux tu m'aider à faire tout ça? Faisons un pull origin d'optyx et ouvrons une pull request pour faire des tests du SBS sur optyx directement.

une fois tout ça fait, je pusherai optyx et mes fichiers markdowns.

## Mathematical and implementation plan

The verification compares the finite-depth SBS certificate, the ordinary-BS reduction,
unitary-family scaling, and repeated Fock truncation with independent finite-dimensional
calculations. Passive unitaries and loop blocks are NumPy arrays; exact low-photon states
use an independent Fock-basis simulator; each mathematical claim becomes a deterministic
pytest assertion. Wiki-specific execution and local-workspace discovery do not belong here.

- [WIP] @codex-root-2026-08-11 20:01 Port the independent Fock simulator and reduction/certificate checks into Optyx tests.
- [ ] Port the Haar/structured-family and repeated-truncation checks into Optyx tests.
- [ ] Remove wiki-only runners and expose one documented pytest entry point.
- [ ] Run `pflake8 optyx` and `coverage run -m pytest`.
- [ ] Prepare the Optyx PR and update wiki-content links to stable Optyx code.
