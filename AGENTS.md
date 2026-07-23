# AGENTS.md

## What

optyx is a Python library for networked quantum architectures, built on
[DisCoPy](https://github.com/discopy/discopy) string diagrams. It mixes qubit
registers with photonic modes, supports lossy channels, heralded measurements
and classical feedback, and evaluates diagrams as tensor networks.

## Context

Read the following documents before any work on the package:

- @RULES.md describes a protocol based on checkboxes as mutex, follow it exactly.
- @STYLE.md describes coding guidelines that all your work should try to follow.
- @README.md contains a high-level description of the features along with some examples.
- @CONTRIBUTING.md contains setup instructions and our general coding philosophy.
- The [refactor roadmap](https://github.com/rel-int/optyx/issues/5) tracks planned work; check whether your change overlaps with a planned item before starting.

## Where

- [optyx/core](optyx/core/) is the back-end: the pure Kraus layer ([diagram](optyx/core/diagram.py)), the doubled channel layer ([channel](optyx/core/channel.py)), the zw/zx/path calculi, and the evaluation [backends](optyx/core/backends.py)
- [optyx/photonic.py](optyx/photonic.py), [optyx/qubits.py](optyx/qubits.py) and [optyx/classical.py](optyx/classical.py) are the user-facing modules, built on the channel layer
- [optyx/compiler](optyx/compiler/) compiles MBQC patterns to hardware instructions
- [test](test/) contains the test suite; doctests in the source are also collected (`--doctest-modules`)
- [docs](docs/) contains the sphinx documentation and example notebooks

## How

Before writing any code, make sure that:

1) your change was first described in high-level mathematical terms
2) this description aligns with the data structures you plan to use

Before pushing anything, make sure that:

- you have reported any bugs or confusing docs that you encounter even if unrelated
- you have added docs and tests that are complete but concise as best as you can
- you have run both `pflake8 optyx` and `coverage run -m pytest` as described in @CONTRIBUTING.md
- you have respected the [code style guide](STYLE.md)
