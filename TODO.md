# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
session, per the discopy#513/#514 precedent (see also optyx#34).

Fixes #42: the module docstring of `optyx/channel.py` has one doctest line
`symbol="$\\mapsto$"` inside a non-raw docstring. Python collapses `\\` to a
single backslash while parsing the docstring itself, so the text doctest
actually re-compiles as Python source is `symbol="$\mapsto$"` — an invalid
escape sequence, reported as `SyntaxWarning: invalid escape sequence '\m'`
on every test run.

- [x] Double the escape (`\\\\mapsto`) on that one line so the text doctest
      sees a literal `\\mapsto`, a valid escape.
- [x] `pflake8 optyx/channel.py`: clean.
- [x] `python -W error::SyntaxWarning -m pytest --doctest-modules
      optyx/channel.py`: 4 passed (confirmed it fails the same way on
      unmodified `main` first, to make sure the fix is what closes it).
