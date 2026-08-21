# TODO

Human prompt (verbatim):

> Convert all the jupyter notebooks in optyx to marimo, in the same way that we did it in DisCoPy. You can use marimo convert command, but check that the result looks good and runs. Make a PR for the notebooks that are already in main. Then go through every open PR and change the notebooks there too. You can push the changes to every PR.

Work items:

- [x] Convert every `docs/notebooks/*.ipynb` to a marimo `.md` notebook (`marimo convert` + `marimo export md`), verifying each one still runs (or confirming any failure is a pre-existing bug unrelated to the conversion).
- [x] Convert every `examples/*.ipynb` to a marimo `.md` notebook the same way.
- [x] Wire up the docs build the way DisCoPy did (`docs/export_notebooks.py`, `docs/conf.py`, `docs/notebooks.rst`), dropping `nbsphinx`.
- [x] Update `.github/workflows/main.yml`, `pyproject.toml`, `.gitignore`, `CONTRIBUTING.md` for marimo instead of Jupyter/nbsphinx.
- [x] File GitHub issues for any pre-existing bugs/staleness found in the notebooks during conversion (#62, #63, #64, #65).
- [x] Open a draft PR against `main` with the conversion (#67).
- [x] Enumerate every open PR in `rel-int/optyx`.
- [ ] For each open PR that touches or adds notebooks, convert those notebooks to marimo too and push to the PR's branch: #58 (x), #39 (x), #36 (x), #15 (x), #61 (in progress).
