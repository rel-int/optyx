# TODO

> Open a new pull request moving optyx/core/channel.py to optyx/channel.py in the library. All top modules (photonic, qubits, etc) depend on those definitions. Add a marimo notebook explaining the difference between optyx/channel.py and the implementation of CQ maps in discopy/quantum. Finish with a markdown paragraph on which implementation you prefer and how you would improve/unify them.

- [x] Move `optyx/core/channel.py` to `optyx/channel.py` and update every import in the library, the tests and the notebooks
- [x] Update the docs and AGENTS.md so that `channel` is listed as a top-level module rather than a `core` submodule
- [WIP] @10d78100-2026-07-26 18:42 Add `examples/channel_vs_cqmap.py`, a marimo notebook comparing `optyx.channel` with `discopy.quantum.channel`, ending with a paragraph on which one to prefer and how to unify them
