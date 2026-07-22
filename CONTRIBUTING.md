# Contributing

Thank you for considering contributing to optyx. The project follows the same
philosophy and conventions as [DisCoPy](https://github.com/discopy/discopy),
on which it is built — if you have contributed to DisCoPy, you will find
everything familiar. If you want guidance, don't hesitate to
[open an issue](https://github.com/rel-int/optyx/issues/new), even if it's to
ask a simple question.

## Get started

Clone the repository and install the development dependencies:

```shell
git clone https://github.com/rel-int/optyx.git
cd optyx
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

Note: optyx currently tracks the `main` branch of DisCoPy rather than a
released version; see `pyproject.toml` and [TODO.md](TODO.md) (item W1) for
the current pin. Packaging is planned to move to
[uv](https://docs.astral.sh/uv/), matching DisCoPy (TODO item W8) — these
instructions will be updated when that lands.

## Run the tests

```shell
pflake8 optyx
coverage run -m pytest
coverage report -m
```

Doctests in the source are part of the suite (`--doctest-modules` in
`pyproject.toml`), so keep the examples in docstrings runnable.

## Build the docs

You'll need [pandoc](https://pandoc.org/) as an external dependency:

```shell
pip install -e '.[docs]'
sphinx-build docs docs/_build/html
```

## Add documentation

We use the same convention as DisCoPy so that documentation images are
generated automatically when running doctests:

```
Example
-------
>>> from optyx import photonic
>>> circuit = photonic.BBS(0) >> photonic.Phase(0.5) @ photonic.Id(1)
>>> circuit.draw(path='docs/_static/photonic/bbs-example.png')

.. image:: /_static/photonic/bbs-example.png
    :align: center
```

Make sure you remember to push changes to these documentation images, but
don't push if the changes are only due to minor glitches, e.g. font aliasing.

## Code style guide

This is DisCoPy's style guide, which holds for optyx too:

- **optyx is pure.** Diagram composition should never cause side-effects, only functor application does when the codomain is effectful.
- **optyx is deterministic.** Even in their internal representation, data structures should not depend on sources of non-determinism (e.g. hashing).
- **optyx is transparent.** `eval(repr(x)) == x` should always be true and `eval(str(x)) == x` should be true assuming the obvious variable naming convention. This `str(x)` should be as close as possible to what a mathematician would write on the board.
- **optyx has no secrets.** We avoid using private or semiprivate attributes and let the user see the internals of each data structure. We expose the interface of every subprocedure as methods that can be tested and reused.
- **optyx cares about naming.** Classes and methods should have short descriptive names, when possible the names correspond to well-known mathematical definitions.
- **optyx speaks for itself.** The code should be clear enough that it doesn't need comments, only documentation with links to mathematical definitions.
- **optyx does not show off.** If there is a simpler way to name or explain something, don't make it sound more complicated.
- **optyx never repeats itself.** Identity and composition of diagrams are defined once, in DisCoPy. If there's duplicate code then you're probably working at the wrong level of abstraction.
- **optyx aims at never nesting.** We believe if your code goes beyond three levels deep then you're probably working at the wrong level of abstraction.

## LLM guidelines

We accept contributions from large language models so long as they are
explicitly indicated as such. We recommend using our [AGENTS.md](AGENTS.md) in
your prompts so that the model has enough context to give quality results.

LLMs have shifted the bottleneck of software development from writing code to
reviewing it, please ensure that your AI assistants save more human time than
they require to supervise them. In particular, AI contributions should be
small (a thousand lines is a red line not to cross lightly) and well-planned
(delegate the execution not the design).

One specific guideline for PR descriptions: it's fine to have the detailed
list of changes LLM-generated but the high-level description should be either
written by a human or quoting a human's prompt verbatim.
