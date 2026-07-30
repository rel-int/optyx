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

## Review round 1

- [x] remove `feedback_loops`: carry the loops' states on the stream instead
- [x] obtain `stream` by applying a Functor, not a handwritten fold
- [x] no pre-scan in `to_tensor`/`eval`: the feedback box raises when hit
- [x] drop the assert in `unroll` and the inline pylint comment in `stream`
- [x] tests with qubits and bits

## Review round 2

- [x] drop the `stream` method: `feedback` and `unroll` are the whole interface
- [x] build the stream inside `unroll` through `StreamFunctor`, uncached

## Review round 3

- [x] drop the Stream and StreamFunctor classes: call discopy.stream inline in `unroll`
- [x] `path.Matrix.feedback` returns a `path.Feedback` class, not a Stream
- [x] remove the pylint comments and the separate `map_feedback`

## Review round 4

- [x] simplify the Feedback init
- [x] CNOT-ladder doctests: core unrolling with |+>, channel double/unroll Equation
- [x] remove feedback from path, and the unused to_path branch
