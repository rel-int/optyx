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

## 🌙 Evening 2026-08-16: bugs #2/#3

Stacked on this same branch's commit above, since both regression tests
need the shape-preserving `conjugate()` that fix provides. No verbatim
human prompt — same self-initiated-bugfix shape as above.

The other two reports on #51 turned out to share one root cause, not two:
`Box.determine_output_dimensions()`'s generic array branch (around line
604 pre-fix) returned `input_dims` unchanged — a list of length
`len(self.dom)` — instead of a list of length `len(self.cod)`. For any
array-backed box with `dom != cod`, `to_tensor()`'s `layer_dims`
bookkeeping (a plain Python list, updated per-box) silently went out of
sync with the wire *count* tracked separately alongside it, corrupting
every downstream box's offset slicing. Bit types are always dimension 2,
so the fix mirrors `Box.truncation`'s existing pattern for the same
class: `[2] * len(self.cod)`.

This explains both remaining reports in #51 as the same bug at two call
sites: `Channel.double()` (bug #2, composes `kraus @ kraus.conjugate()`
then a permutation ladder sized from the corrupted `layer_dims`) and a
plain sequential diagram with an idle parallel wire downstream of a
non-square array box (bug #3, `to_tensor()` outside `double()` entirely).
Neither `calculate_right_offset` nor `get_max_dim_for_box` — the two
functions #51 and this PR's first commit both guessed at — turned out to
be at fault; both are self-consistent given the (correct) inputs they're
handed.

- [x] Reproduce bug #2: `Channel(...).double().to_tensor().eval()` on a
      non-square array kraus map raises `AxiomError`, exactly as in #51.
- [x] Reproduce bug #3: a plain diagram — a non-square array box in
      parallel with an idle wire, then a second box consuming that idle
      wire alongside the first box's output — raises `AxiomError` on
      `to_tensor().eval()`, confirming the bug isn't specific to
      `Channel`/`double()`.
- [x] Root-cause both to `Box.determine_output_dimensions()` returning the
      wrong-length list for non-square array boxes (see above), not the
      `calculate_right_offset`/`get_max_dim_for_box` territory #51 guessed
      at.
- [x] Fix: `[2] * len(self.cod)`, matching `Box.truncation`'s existing
      convention for array-backed boxes.
- [x] Add regression tests in `test/test_channel.py` for both repros,
      confirmed to fail with `AxiomError` on the pre-fix code and pass
      after; also checks the doubled non-square kraus evaluates to the
      expected elementwise square, and the plain-diagram case to the
      expected matrix product where a numeric check is tractable.
- [x] `pflake8 optyx`: clean.
- [x] `coverage run -m pytest`: 1151 passed, no regressions.
- [x] Comment on #51 noting both remaining bugs are fixed here.
