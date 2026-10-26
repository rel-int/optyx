# TODO

> Write an algorithm to compute the contextual fraction of any empirical model and push it in a new branch in optyx

- [x] `optyx/contextuality.py`: measurement scenarios, empirical models and the contextual fraction as a linear program (Abramsky, Barbosa & Mansfield 2017), with the Bell inequality read off the dual
- [x] `EmpiricalModel.from_diagram`: the empirical model of an optyx diagram with classical setting inputs and outcome outputs
- [x] tests in `test/test_contextuality.py`, `scipy` dependency, `docs/api.rst` entry, `pflake8`, `pylint` and `coverage run -m pytest`
- [x] report the bugs met on the way: `Diagram.eval` crashes on purely classical diagrams, filed as [#86](https://github.com/rel-int/optyx/issues/86)
- [x] differentiate through the contextual fraction: `EmpiricalModel.contextual_fraction_gradient` is the dual solution and `maximise_contextual_fraction` runs gradient ascent on the parameters of a setup, e.g. the measurement phases of a Bell test to the Tsirelson bound
