# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
🌙 Evening session, per the discopy#513/#514 precedent (see also optyx#34,
discopy#537).

Fixes #18: `qubits.Ket`/`qubits.Bra` compare their `value` parameter against
the integers `0`/`1` (`value in (0, 1)`), but a caller naturally writes
`Ket("0")`/`Ket("1")` (strings) by analogy with `"+"`/`"-"`. Since
`"0" != 0` and `"1" != 1` in Python, both membership tests silently fail and
fall through to the `zx.Z`/phase-0.5 branch — exactly the branch used for
`Ket("-")` — building the wrong state with no error.

- [x] Add `_normalise_basis_value` in `optyx/qubits.py`: reads `"0"`/`"1"`
      as the integers `0`/`1`, and raises `ValueError` on anything else
      (previously any invalid value silently fell through to the `"-"`
      branch too). Used by both `Ket.__init__` and `Bra.__init__`.
- [x] Regression tests in `test/test_qubit.py`:
      `test_ket_bra_accept_string_digits` (`Ket("0") == Ket(0)`, etc.) and
      `test_ket_bra_reject_invalid_value`.
- [x] `pflake8 optyx/qubits.py`: clean.
- [x] `pytest test/test_qubit.py`: 11 passed (fresh `python3.12` venv,
      `pip install -e` against this session's local discopy clone since
      `uv sync` can't resolve optyx's `requires-python = ">=3.9"` against
      its pinned discopy commit needing `>=3.12` in this sandbox — worth
      a separate issue, not touched here since it's unrelated to this fix).
