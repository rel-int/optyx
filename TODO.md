# TODO

> Ok let's move to the second PR suggested in the roadmap issue. We will open the PR as draft based on the current align with DisCoPy PR. This PR should focus on removing/deduplicating code while keeping all the tests and coverage in check.

- [x] Delete commented-out dead code (qubits.py, core/channel.py, utils/perceval_conversion.py) and fix the phantom `lo` docstring references
- [WIP] @34e6a827-2026-07-23 14:20 Default `Box.conjugate` to the box itself and delete the identical per-box bodies; delete `determine_output_dimensions` bodies that duplicate a parent
- [ ] Deduplicate `Scalar`, the `Id` helpers, the empty-sum grad fallbacks and the `from_bosonic_operator` operator loop
- [ ] Dissolve `utils/misc.py` into core modules and break the layering inversion in `utils/perceval_conversion.py`
- [ ] Rename `test/qpath.py` and `test/grad.py` to the `test_` convention and replace star imports with explicit imports
