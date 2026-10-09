#!/usr/bin/env python
"""Render the marimo notebooks in ``docs/notebooks`` for the documentation.

For every marimo notebook ``docs/notebooks/*.md`` this module:

1. runs it and exports the *computed* result to a self-contained HTML file in
   ``docs/_static/notebooks/<name>.html`` (via ``marimo export html``), and
2. writes a small reStructuredText page ``docs/notebooks/<name>.rst`` that
   embeds that HTML in an ``<iframe>`` so Sphinx picks it up in the toctree.

Both are generated, not committed: ``conf.py`` calls :func:`generate` from a
``builder-inited`` hook, which renders the notebooks whose HTML is missing or
older than their source, so ``sphinx-build`` renders them as part of a normal
docs build. A notebook that cannot be executed falls back to a link to its
source rather than failing the build.

It can also be run as a script, the way CI does before building the docs::

    python docs/export_notebooks.py            # render every notebook
    python docs/export_notebooks.py fixpoints  # render a single notebook
    python docs/export_notebooks.py --check    # only check they execute
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
HTML_DIR = DOCS / "_static" / "notebooks"
SOURCE = "https://github.com/rel-int/optyx/blob/main/docs/notebooks"

IFRAME = """\
.. raw:: html

    <iframe class="marimo-notebook" src="../_static/notebooks/{name}.html"
            title="{title}" loading="lazy"
            style="width: 100%; height: 90vh; border: none;"></iframe>
"""

THEME_STYLE = """\
<style type="text/css">
/* DisCoPy SVGs turn the elements tagged by drawing.backend.DARK_MODE_STYLE
   white under a dark prefers-color-scheme; rendered inline in the notebook,
   they read the browser preference rather than the notebook theme, so these
   higher-specificity rules retag them by marimo's theme class instead. */
.light [id^="dark-stroke"] path { stroke: #000000 !important; }
.light [id^="dark-fill"] g, .light [id^="dark-fill"] use
{ fill: #000000 !important; stroke: #000000 !important; }
.dark [id^="dark-stroke"] path { stroke: #ffffff !important; }
.dark [id^="dark-fill"] g, .dark [id^="dark-fill"] use
{ fill: #ffffff !important; stroke: #ffffff !important; }
</style>"""

FALLBACK = """\
.. note::

    This notebook could not be rendered in this build (a dependency needed to
    execute it may be missing). You can read its source on GitHub:
    `{name}.md <{source}/{name}.md>`_.
"""


def notebooks() -> list:
    """The marimo notebooks, the ``.md`` files with ``{.marimo}`` cells."""
    return sorted(path for path in NOTEBOOKS.glob("*.md")
                  if "{.marimo}" in path.read_text())


def title_of(notebook: Path) -> str:
    """Read the ``title:`` field from a marimo notebook's YAML front-matter."""
    lines = notebook.read_text().splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip()
    return notebook.stem


def export(notebook: Path, *, check: bool) -> None:
    """Run ``notebook`` and export the computed HTML (unless ``check``).

    The notebook is run as the Python app ``marimo convert`` makes of it, so
    that a process pool, such as the one cotengra starts to optimise a
    contraction, imports Python when it spawns its workers rather than the
    Markdown source.
    """
    with TemporaryDirectory() as scratch:
        app = Path(scratch) / f"{notebook.stem}.py"
        if check:
            output = Path(scratch) / f"{notebook.stem}.html"
        else:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            output = HTML_DIR / f"{notebook.stem}.html"
        marimo = [sys.executable, "-m", "marimo"]
        subprocess.run(marimo + ["convert", notebook.name, "-o", str(app)],
                       cwd=NOTEBOOKS, check=True)
        subprocess.run(marimo + ["export", "html", str(app),
                                 "-o", str(output), "-f"],
                       cwd=NOTEBOOKS, check=True)
        add_theme_style(output)


def add_theme_style(output: Path) -> None:
    """Insert :data:`THEME_STYLE` in the head of an exported notebook."""
    text = output.read_text()
    output.write_text(text.replace("</head>", THEME_STYLE + "</head>", 1))


def write_page(notebook: Path, *, rendered: bool) -> None:
    """Write the reStructuredText page for ``notebook``."""
    title = title_of(notebook)
    body = (IFRAME if rendered else FALLBACK).format(
        name=notebook.stem, title=title, source=SOURCE)
    (NOTEBOOKS / f"{notebook.stem}.rst").write_text(
        f"{title}\n{'=' * len(title)}\n\n" + body)


def is_current(notebook: Path) -> bool:
    """Whether the HTML of ``notebook`` is newer than its source."""
    output = HTML_DIR / f"{notebook.stem}.html"
    return output.exists() \
        and output.stat().st_mtime >= notebook.stat().st_mtime


def generate(*, strict: bool = False) -> None:
    """Render every notebook to HTML and write its page (used by ``conf.py``).

    When ``strict`` is false a notebook that fails to execute falls back to a
    link page instead of aborting the build.
    """
    for notebook in notebooks():
        try:
            if not is_current(notebook):
                export(notebook, check=False)
            write_page(notebook, rendered=True)
        except (subprocess.CalledProcessError, OSError):
            if strict:
                raise
            write_page(notebook, rendered=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*",
                        help="notebook stems to render (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="only check the notebooks execute, write nothing")
    args = parser.parse_args()

    stems = set(args.notebooks)
    selected = [notebook for notebook in notebooks()
                if not stems or notebook.stem in stems]
    if not selected:
        print("no matching notebooks found", file=sys.stderr)
        return 1

    for notebook in selected:
        print(f"rendering {notebook.name} ...", flush=True)
        export(notebook, check=args.check)
        if not args.check:
            write_page(notebook, rendered=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
