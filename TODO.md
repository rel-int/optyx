# TODO

> Write an algorithm to compute the contextual fraction of any empirical model and push it in a new branch in optyx

- [x] `optyx/contextuality.py`: measurement scenarios, empirical models and the contextual fraction as a linear program (Abramsky, Barbosa & Mansfield 2017), with the Bell inequality read off the dual
- [x] `EmpiricalModel.from_diagram`: the empirical model of an optyx diagram with classical setting inputs and outcome outputs
- [x] tests in `test/test_contextuality.py`, `scipy` dependency, `docs/api.rst` entry, `pflake8`, `pylint` and `coverage run -m pytest`
- [x] report the bugs met on the way: `Diagram.eval` crashes on purely classical diagrams, filed as [#86](https://github.com/rel-int/optyx/issues/86)
- [x] differentiate through the contextual fraction: `EmpiricalModel.contextual_fraction_gradient` is the dual solution and `maximise_contextual_fraction` runs gradient ascent on the parameters of a setup, e.g. the measurement phases of a Bell test to the Tsirelson bound
- [x] `MessagePassingScenario`: the measurement scenario of a `CMap` of `optyx.interaction` for any physical system on its wires, the nodes as parties each with a family of local measurements of its prediction chosen by a classical wire, with `learn` finding the angles maximising the contextual fraction from the most contextual of a few random starts; `local_measurement` for qubits and `dual_rail_measurement` for pairs of modes as defaults, the Bell-pair map giving the Tsirelson bound on both; `optyx.interaction` is brought over verbatim from the branch of the closed #16 since it is not on `main`
- [x] `EmpiricalModel.conditionals` evaluates a diagram once with its bit inputs open rather than once per setting
