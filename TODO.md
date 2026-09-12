# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
🌙 Evening session, per the discopy#553/optyx#46/#48/#49 precedent.

Fixes #72: `test/test_qubit.py` fails to *collect* (not just fail) on every
fresh `optyx` CI run, `main` included, now that `pytket-qiskit` resolved to
`0.78.0` and dropped `AerBackend`. The three imports it broke (`Circuit`,
`AerBackend`, `probs_from_counts`) are only used inside two tests already
commented out (`test_tket_discopy`, `test_to_tket`) — dead weight since
before this break, now fatal to the whole module's collection.

- [x] Remove the three unused `pytket` imports from `test/test_qubit.py`
      (`from pytket import Circuit`, `from pytket.extensions.qiskit import
      AerBackend`, `from pytket.utils import probs_from_counts`). Checked
      every active (non-commented) use of `Circuit` in the file first —
      all of them are `pyzx.Circuit`, `qubits.Circuit` or `graphix.Circuit`,
      never the bare `pytket.Circuit` this import shadowed.
- [x] Reproduced red-before: `pytest test/test_qubit.py --collect-only`
      raises `ImportError: cannot import name 'AerBackend' from
      'pytket.extensions.qiskit'`, 2 collection errors, matching #72
      exactly.
- [x] Confirmed green-after: `pytest test/test_qubit.py` — 9 passed.
- [x] `pytest test/`: full suite green, no regressions elsewhere (see PR
      description for the count).
- [x] `pflake8 optyx`: clean (the change is confined to `test/`, which the
      repo's CI lint job doesn't cover — `SRC_DIR: optyx` in
      `.github/workflows/main.yml`, same as optyx#48's note).

Not touched: the `qubits.py` doctest comment mentioning `AerBackend` (#72
flags it only as background, not part of the fix) and reviving the two
commented-out tket tests themselves, which #72 calls out as a separate
decision (needs a replacement for `AerBackend`, e.g. `qiskit_aer` directly
or pinning `pytket-qiskit < 0.78`).
