# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
🌙 Evening session, per the optyx#46/#48/#34 precedent (see also discopy#553).

Fixes the first of the two bugs filed in #51 (found while building the
sudoku experiment on `optyx.interaction`, #16): `Box.conjugate()`'s generic
array branch swapped `dom`/`cod` and renamed the result `.dagger()`,
computing the conjugate-with-flipped-orientation instead of the entrywise,
shape-preserving conjugate every caller of `conjugate()` actually expects
(e.g. `Channel.double()`'s `self.kraus @ self.kraus.conjugate()`, which
requires the two operands to keep their own `dom`/`cod`). This made
`Channel.double()` raise `AxiomError` for any channel whose kraus map was a
plain array box with `dom != cod`.

`#51`'s other two reports — classical doubling mis-assembling when
`len(dom) != len(cod)` even with a shape-preserving `conjugate` patched in,
and the same arity mis-assembly in `to_tensor` outside `double()` entirely —
are a separate bug in the permutation/width bookkeeping, not this one, and
are **not** fixed here; left open on #51 for a follow-up session, since they
need a closer look at `calculate_right_offset`/`get_max_dim_for_box`
territory rather than a one-line fix.

- [x] Reproduce bug #1 from #51: a non-square array `Box` (`dom != cod`)
      passed through `Channel.double()` raises `AxiomError`.
- [x] Fix `Box.conjugate()` in `optyx/core/diagram.py` to preserve `dom`/
      `cod` and drop the misleading `.dagger()` name suffix, conjugating
      only the array entries.
- [x] Add regression tests in `test/test_channel.py`: `Box.conjugate()`
      keeps `dom`/`cod` and conjugates the array entrywise, and
      `Channel.double()` no longer raises on a non-square kraus map.
      Confirmed both fail on the pre-fix code (`AxiomError` / shape
      assertion) and pass after.
- [x] `pflake8 optyx`: clean.
- [x] `coverage run -m pytest` (via the optyx#47 workaround venv, Python
      3.12 + `pip install -e '.[test]'`, since `uv sync` still can't resolve
      on this repo): 1022 passed, no regressions.
- [x] Comment on #51 noting bug #1 is fixed here and bugs #2/#3 remain open.
