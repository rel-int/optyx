# TODO

> _encode_wire and _encode_ty in optyx.channel are not needed, the code can directly be used in
> inflate. Hidden methods like this are not allowed, can you find other examples in optyx? I also
> want to start testing the pylint more seriously, for example Add is imported in that method, I'm
> wondering if there's a cleaner solution. Make a plan for a new PR

Scope agreed with the user: `optyx/channel.py`, `optyx/core/diagram.py` and the pylint config,
plus the one-line ripples the renames force elsewhere. CI keeps `--fail-under=9`; the inline
`# pylint: disable` comments go, and the policy moves into `pyproject.toml`. The rest of the
package is listed in the PR body, not fixed here.

## Baseline

- [ ] Install `.[test]` and record `pylint optyx` score + by-symbol census before any change

## No secrets

- [x] Inline `Measure._measure_wire` into `Measure.inflate`
- [x] Inline `Encode._encode_wire` into `Encode.inflate`
- [x] `Channel._decomp` -> `decomp` (shadows `Diagram.decomp`, a plain functor)
- [x] `Channel._to_dual_rail` -> `dual_rail` (`Diagram.to_dual_rail` is `functor . decomp`, so the
      box hook is a different map and needs its own name — sharing it breaks `channel.py:139`)
- [x] `Ob._classical` / `Ob._quantum` -> `Ob.classical` / `Ob.quantum`
- [x] `core/diagram.py`: drop `Box._array` and its pass-through property

## Pylint

- [x] Hoist the deferred imports in `channel.py` that were never circular (`core.path`, `core.zw`)
- [x] Hoist the deferred imports in `core/diagram.py` that were never circular (`core.path`,
      `utils.misc`, `sympy`, `copy`) and fix the import order
- [x] `good-names` in `pyproject.toml`; delete the `invalid-name` disables in the two files
- [x] Fix `bad-classmethod-argument`, `redefined-builtin` in `core/diagram.py`
- [x] `no-else-return` in `Box.photon_number_transform` (`Spider.determine_output_dimensions` was already clean)
- [ ] Leave the genuinely circular `import-outside-toplevel` visible; open the layering issue

## Verify

- [x] Tests for the inlined `inflate` branches
- [ ] `pflake8 optyx`, `pylint optyx --fail-under=9`, `coverage run -m pytest`,
      `coverage report --fail-under=95`, no `docs/_static` churn
