# TODO

> Ok let's move to the second PR suggested in the roadmap issue. We will open the PR as draft based on the current align with DisCoPy PR. This PR should focus on removing/deduplicating code while keeping all the tests and coverage in check.

- [x] Delete commented-out dead code (qubits.py, core/channel.py, utils/perceval_conversion.py) and fix the phantom `lo` docstring references
- [x] Default `Box.conjugate` to the box itself and delete the identical per-box bodies (`determine_output_dimensions` overrides in zx/zw turned out non-redundant: the base raises without an array, and the dagger orientation differs — kept)
- [x] Deduplicate `Scalar`, the `Id` helpers, the empty-sum grad fallbacks and the `from_bosonic_operator` operator loop
- [x] Dissolve `utils/misc.py` into core modules and break the layering inversion in `utils/perceval_conversion.py`
- [ ] Rename `test/qpath.py` and `test/grad.py` to the `test_` convention and replace star imports with explicit imports
