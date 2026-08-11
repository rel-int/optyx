# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
🌙 Evening session, per the discopy#553/optyx#46 precedent (see also
optyx#48, optyx#34).

Fixes #19: `utils.misc.preprocess_quimb_tensors_safe`, applied by
`QuimbBackend` on every approximate (compressed) contraction, perturbs
rank-deficient 2D tensors with `np.random.normal` drawn from the global
unseeded RNG before the SVD-based compressed contraction. Two identical
calls to `eval` with a compressed hyperoptimiser can therefore return
different numbers, which STYLE.md's determinism guideline forbids and
which is a likely contributor to tight-tolerance test flakes (e.g.
optyx#15's `test_adaptive_defaults`, though this fix does not assume that
flake is fully explained by it). Separately, the function's silent
zero-filling (`data[data == 0] = epsilon`) was undocumented.

- [x] Replace the module-level `np.random.normal` call with a local
      `numpy.random.Generator` (`np.random.default_rng(seed)`) constructed
      fresh inside `preprocess_quimb_tensors_safe`, with an optional
      `seed` keyword argument defaulting to `0` so repeated calls on the
      same input are bit-identical by default, while still letting a
      caller override the seed if ever needed.
- [x] Document the zero-filling behaviour in the function's docstring
      (perturbs rank-deficient 2D tensors and replaces exact zeros with
      `epsilon` before quimb's SVD-based contraction, and why) per
      STYLE.md's "speaks for itself" guideline — no behaviour change.
- [x] Add a regression test proving determinism: call
      `preprocess_quimb_tensors_safe` twice on the same rank-deficient
      quimb tensor network and assert the results are bit-identical.
- [x] `pytest test/test_backends.py`: 100 passed (fresh `python3.12` venv,
      `pip install -e .[test] quimb cotengra` — same `uv sync` workaround
      as #46/#48, tracked as optyx#47).
- [x] `pytest test/`: 1021 passed, no regressions elsewhere.
- [x] `pflake8 optyx`: clean. Also ran `pylint optyx`: 9.54/10 (CI gate is
      `--fail-under=9`), and `pylint optyx/utils/misc.py` on its own:
      9.03/10, no new findings on the touched lines.
- [x] Open a draft PR against `main`, referencing #19, per the
      TODO.md-deletion precedent tracked as giodefelice/desire#6.
