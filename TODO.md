# TODO

> Open a PR in optyx to solve this issue https://github.com/rel-int/optyx/issues/6

- [WIP] @claude-g9z1pv-2026-07-28 14:20 — `channel.Diagram.feedback(dom, cod, mem, initial_state=None, final_effect=None)`
      returning a composable diagram drawn with a feedback loop
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — `channel.Diagram.stream` cached property building the DisCoPy stream,
      constant when there are no feedback loops, no choice of initial state or final effect
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — `channel.Diagram.unroll(n_steps)` plugging `initial_state` / `final_effect`,
      open wires when they are `None`
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — feedback, stream and unroll on `core.diagram.Diagram` such that `double` and `unroll` commute
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — feedback on `core.path.Matrix` such that `to_path` and `unroll` commute
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — tensor evaluation raises on diagrams with feedback loops, asking to unroll first
- [WIP] @claude-g9z1pv-2026-07-28 14:20 — tests and doctests, including zero-photon `initial_state` and `Discard(mem)` `final_effect`
