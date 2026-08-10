# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
🌙 Evening session, per the discopy#553/optyx#46 precedent (see also
optyx#34, discopy#537).

Fixes #17: `test/test_backends.py::TestQuimbBackend::test_pure_circuit[circuit3]`
flaked sporadically. Two things combined: `chip_mzi`'s ansatz phases were
drawn from `np.random.uniform` with no seed at module import time (so every
CI run tested a different 4x4 interferometer), and `TestQuimbBackend`
compared that exact circuit against a deliberately lossy compressed
contraction (`ReusableHyperCompressedOptimizer`) using the module's default
`rel_tol=1e-05`/`abs_tol=1e-10` — tight enough for exact-vs-exact
comparisons, too tight for exact-vs-compressed. Most draws passed; some did
not, by roughly two orders of magnitude on the failing run's own numbers.

- [x] Seed `chip_mzi`'s random draw with `np.random.default_rng(seed)` and
      sort `ansatz.free_symbols` (a `set`, whose iteration order depends on
      sympy `Symbol` hashing and hence `PYTHONHASHSEED`) before zipping it
      with the draws, so the parametrised circuit is fully reproducible
      rather than merely less random.
- [x] Give the exact-vs-compressed comparisons in
      `TestQuimbBackend.test_pure_circuit` their own `COMPRESSION_TOL`
      (`rel_tol=1e-3`, `abs_tol=1e-6`), leaving every other comparison in
      the module — all exact vs exact — at the tight default. Scoped to
      that one test: `test_mixed_circuit` doesn't use `chip_mzi` at all.
- [x] `pytest test/test_backends.py`: 99 passed, `TestQuimbBackend` reran 3x
      clean to confirm the flake is gone, not just less likely (fresh
      `python3.12` venv, `pip install -e .[test]`, same `uv sync`
      workaround as #46 — see optyx#47).
- [x] `pytest test/`: 1020 passed, no regressions elsewhere.
- [x] `pflake8 optyx`: clean (the change is confined to `test/`, which the
      repo's CI lint job doesn't cover — `SRC_DIR: optyx` in
      `.github/workflows/main.yml`).
