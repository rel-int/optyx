# TODO

> Open a PR in optyx to solve this issue https://github.com/rel-int/optyx/issues/6

- [x] `channel.Diagram.feedback(dom, cod, mem, initial_state=None, final_effect=None)`
      returning a composable diagram drawn with a feedback loop
- [x] `channel.Diagram.stream` cached property building the DisCoPy stream,
      constant when there are no feedback loops, no choice of initial state or final effect
- [x] `channel.Diagram.unroll(n_steps)` plugging `initial_state` / `final_effect`,
      open wires when they are `None`
- [x] feedback, stream and unroll on `core.diagram.Diagram` such that `double` and `unroll` commute
- [x] feedback on `core.path.Matrix` such that `to_path` and `unroll` commute
- [x] tensor evaluation raises on diagrams with feedback loops, asking to unroll first
- [x] tests and doctests, including zero-photon `initial_state` and `Discard(mem)` `final_effect`
