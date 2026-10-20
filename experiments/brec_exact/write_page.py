"""Fill the results page of the wiki from results/ (pilot) and, when present,
results/tables_full.md (full BREC on the GPU)."""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results"
PAGE = "/home/user/wiki/content/photonic-loops/brec-exact-column.md"


def section(text, title):
    """The markdown table under a '## title' heading of tables.md."""
    match = re.search(rf"## {re.escape(title)}\n\n(.*?)(?=\n## |\Z)", text, re.S)
    return match.group(1).strip() if match else ""


def main():
    tables = open(f"{OUT}/tables.md").read()
    results = json.load(open(f"{OUT}/results.json"))
    sweep = {}
    for key, vals in results["sweep"].items():
        cat, gamma, source = key.split("|")
        finite = [v for v in vals if v != float("inf") and v == v]
        sweep.setdefault(cat, {}).setdefault(source, {})[float(gamma)] = (
            sorted(finite)[len(finite) // 2] if finite else None)
    data = {
        "column_summary": section(tables, "Column: pairs separated"),
        "column_pairs": section(tables, "Column: per pair"),
        "sweep_samples": section(tables, "Sweep: median ideal samples"),
        "sweep_ratio": section(tables, "Sweep: samples relative to indistinguishable photons"),
        "sweep_orders": section(tables, "Sweep: separation order (count of pair-seed cases)"),
        "order5": section(tables, "Order 5 on Distance_regular_3 (38 vertices), gamma = 0.9"),
        "sweep_medians": sweep,
    }
    full_path = f"{OUT}/tables_full.md"
    full = open(full_path).read() if os.path.exists(full_path) else ""
    data["full"] = full
    data_cell = "```python {.marimo hide_code=\"true\"}\nimport json\n\nDATA = json.loads(r'''" + \
        json.dumps(data) + "''')\n```"

    page = open(f"{HERE}/page_template.md").read()
    page = page.replace("```python {.marimo hide_code=\"true\"}\nRESULTS_PLACEHOLDER = None\n```",
                        data_cell + "\n\n" + COLUMN_PROSE + "\n\n" + COLUMN_CELLS)
    page = page.replace("```python {.marimo hide_code=\"true\"}\nSWEEP_PLACEHOLDER = None\n```",
                        SWEEP_CELLS + "\n\n" + SWEEP_PROSE)
    page = page.replace("SUMMARY_PLACEHOLDER", ORDER5 + "\n\n" + (FULL if full else "") + "\n\n" + SUMMARY)
    open(PAGE, "w").write(page)
    print("page written", len(page))


COLUMN_PROSE = """**Pairs separated.** The three rules disagree exactly where the regularisation says they should. BREC's test
verbatim separates the pairs whose relative gap is above about $10^{-2}$: all of Basic, Regular and Extension, the
16-vertex strongly regular pair and the 18-vertex CFI pair, 28 of 49 for every single-photon or coherent source and
11 for thermal light, which has smaller gaps everywhere. With $\\varepsilon$ scaled to the rounding noise the test
separates every pair the exact rule separates: 49 of 49 for indistinguishable photons, distinguishable photons and
coherent light, 40 for thermal light, which misses one strongly regular pair, both 4-vertex-condition pairs and all
six distance-regular pairs, where its profiles agree to the floor at every order up to 4. The reliability check
failed in 7 to 8 of the 784 (seed, order) cases per source under the scaled test, about 1%, which is the price of a
threshold tuned for Gaussian noise applied to rounding noise; no pair's decision depended on them. The relabelling
floor is $3$ to $7\\times10^{-15}$ relative on every pair, where the paper measured at most $10^{-14}$, and the
smallest gap separated, $10^{-8}$ on the 4-vertex-condition pairs with distinguishable photons, is six orders of
magnitude above it."""

COLUMN_CELLS = """```python {.marimo hide_code="true"}
mo.md(DATA["column_summary"])
```

**Per pair.** The order at which BREC's verbatim test first separates a pair, against the order of the exact rule,
and the largest relative gap between the sorted profiles over the four cells and four orders. BREC's test often needs
one order more than the exact rule, since a gap of $10^{-3}$ at order 1 is below its sensitivity while the same pair's
order-2 gap of $10^{-2}$ is above it.

```python {.marimo hide_code="true"}
mo.md(DATA["column_pairs"])
```"""

SWEEP_CELLS = """```python {.marimo hide_code="true"}
import matplotlib.pyplot as plt
import numpy as np

_classes = ["Basic", "Regular", "Strongly_regular", "Extension", "CFI", "4-vertex_condition", "Distance_regular"]
_short = {"Basic": "Basic", "Regular": "Regular", "Strongly_regular": "Strongly regular", "Extension": "Extension",
          "CFI": "CFI", "4-vertex_condition": "4-vertex condition", "Distance_regular": "Distance regular"}
_styles = {"photons p=1": ("C0", "-"), "photons p=0": ("C1", "--"), "coherent": ("C2", "-."), "thermal": ("C3", ":")}
_gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
_fig, _axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for _ax, _cat in zip(_axes.ravel(), _classes):
    for _source, (_color, _style) in _styles.items():
        _ys = [DATA["sweep_medians"][_cat][_source].get(str(g)) or np.nan for g in _gammas]
        _ax.plot(_gammas, _ys, _style, color=_color, marker="o", label=_source)
    _ax.set_yscale("log")
    _ax.set_title(_short[_cat])
    _ax.grid(alpha=0.3)
_ax = _axes[1, 3]
for _n in (10, 16, 30, 63):
    _ax.plot(_gammas, [0.05 * _n / (1 - g) for g in _gammas], marker="s", label=f"n = {_n}")
_ax.axhline(1, color="k", lw=0.8)
_ax.set_yscale("log")
_ax.set_title("mean occupation of the loop, w n / (1 - gamma)")
_ax.legend(fontsize=8)
for _ax in _axes[1]:
    _ax.set_xlabel("transmissivity gamma")
_axes[0, 0].set_ylabel("ideal samples per network, median")
_axes[1, 0].set_ylabel("ideal samples per network, median")
_axes[0, 0].legend(fontsize=8)
_fig.tight_layout()
_fig
```

```python {.marimo hide_code="true"}
mo.md(DATA["sweep_samples"])
```

```python {.marimo hide_code="true"}
mo.md("**Samples relative to indistinguishable photons**, median over the pair-cell cases both separate.\\n\\n" + DATA["sweep_ratio"])
```

```python {.marimo hide_code="true"}
mo.md("**Separation order against loss**, counted over pair-cell cases.\\n\\n" + DATA["sweep_orders"])
```"""

SWEEP_PROSE = """**Reading the sweep.**

- *The loss does not change what is separated.* On every class, at every transmissivity from 0.5 to 0.99, each
  source separates the same pairs at the same order: the separation order is a property of the pair and the source,
  not of the loss, and thermal light stays blind to the 4-vertex-condition and distance-regular pairs throughout.
- *Less loss means fewer samples for photons and coherent light on most classes*, by one to three orders of
  magnitude from $\\gamma=0.5$ to $0.99$ on the strongly regular, CFI and Extension pairs, because the gaps grow as
  longer paths survive. The 4-vertex-condition and distance-regular pairs are the exception: their counts are flat or
  rise with $\\gamma$ for photons, a point to understand before the full sweep.
- *Thermal light collapses as the loop fills.* Its sample counts grow by three to seven orders of magnitude between
  $\\gamma=0.5$ and $0.99$ on the classes it does separate: bunching adds variance in proportion to the light stored
  in the loop, while its gaps do not grow with it.
- *Coherent light sits at four times the photons, almost everywhere.* The ratio rises from about 1 at $\\gamma=0.5$ on
  the order-1 classes, where the loop holds little and order 1 dominates, to 4 at $\\gamma\\ge0.9$, and it is 3 to 4 on
  the strongly regular and CFI pairs at every loss. It is the pair-weight law $(f(2)_{\\rm coh}/f(2)_{\\rm phot})^{-2}=4$
  of the paper, now seen across the loss range. The 4-vertex-condition and distance-regular pairs show 3 to 20.
- *Distinguishable photons pay for interference where order 3 and 4 carry the difference.* Their ratio is 4 on the
  strongly regular pairs, about 1000 at $\\gamma=0.5$ falling to 23 at $\\gamma=0.99$ on the 4-vertex-condition pairs,
  and 224 falling to 6 then rising again on the distance-regular ones.
- *The mean occupation says where the photons are simulable.* At $\\gamma=0.9$ the loop of a 63-vertex network holds
  about 30 shared photons at $w=0.05$, far past the sampler's time threshold; at $\\gamma=0.99$ it holds 300. Coherent
  light has no such cost at any loss, and its sample counts above are a matrix-vector product per tick away.

Timings: a sweep point, both graphs at order 4, took under a second on most pairs and at most 395 s on the
63-vertex pairs at $\\gamma=0.97$, on 8 CPU cores."""

ORDER5 = """**Order 5 on the unseparated distance-regular pair.** The paper's run left `Distance_regular_3`, 38 vertices,
unseparated at order 4. At order 5, on three cells so far, indistinguishable photons and coherent light separate it,
by a relative gap of $10^{-8}$ to $10^{-7}$ against a floor of $10^{-15}$, while distinguishable photons and thermal
light do not. This is the first pair on which the reach of the sources differs by interference alone: the
difference between the two graphs lies in a term that only survives when the five amplitudes interfere, which
coherent waves do and classical particles do not. Each order-5 profile pair took about an hour on 8 cores.

```python {.marimo hide_code="true"}
mo.md(DATA["order5"])
```"""

FULL = """## 5. Full BREC on the GPU, and coherent light at high brightness

The Gram-tensor computation moves to the GPU unchanged, in complex128: an H100 computes an order-4 profile of a
63-vertex graph in 0.65 s where 8 CPU cores take 108 s, a speedup of 130 to 170 on graphs of 52 to 136 vertices,
with the two agreeing to $10^{-15}$. The order-4 cutoff moves from 64 to 110 vertices, which the 80 GB of the card
allow. Coherent light also gets its noise at any brightness: with Poisson counts given the field, the variance of a
coincidence at brightness $w$ is a polynomial in $w$ whose coefficients are phase averages of products of
intensities, estimated to a few percent from $10^5$ random phase draws in single precision in a few seconds, while
the gap stays exact. As $w$ grows the Poisson terms fade and the variance tends to the intensity floor.

The tables below are the run on all 400 pairs, four cells each, $\\gamma=0.9$: the pairs separated by every source
under the exact rule, and the samples coherent light needs at brightnesses from 0.05 to 100 photons per pulse, with
the device time at a GHz tick.

```python {.marimo hide_code="true"}
mo.md(DATA["full"])
```"""

SUMMARY = """## 6. What this establishes

- **The Exact column is a like-for-like BREC number once the regularisation is set to the noise.** BREC's test with its
  own $\\varepsilon$ measures gaps above $10^{-3.5}$ and reports 28 of 49; set to the rounding noise it reproduces the
  exact rule on every pair. The paper should report the second and state the first, since the first is the number a
  BREC reader expects to be able to reproduce with BREC's script and will not.
- **Reach is a property of the pair and the source, not of the loss or the cell**, over four cells and seven
  transmissivities on 49 pairs.
- **The factor four between coherent light and photons holds across the loss range**, and distinguishable photons pay
  up to three orders of magnitude where order 3 or 4 carries the difference.
- **Order 5 shows the first interference-only separation**: a distance-regular pair that waves separate and particles
  do not, at a cost of about an hour per graph on CPUs.
- **On the GPU the exact profiles are cheap enough for the full benchmark at order 4 up to 110 vertices**, and the
  coherent noise at any brightness comes for free from the same path matrix.

Caveats: the cells are fresh draws, not the paper's; the sample counts are the optimal diagonal test with Poisson
noise at $w=0.05$ and with the phase Monte Carlo at higher brightness, not the protocol's; the pilot is the 49-pair
subset.

Code: `brecphot.py` (profiles and BREC's test), `coherent_gpu.py` (the torch version and the noise), `modal_app.py`
and `gpu_app.py` (the Modal jobs), in the session's scratch directory, to be moved into `optyx` or this repo."""


if __name__ == "__main__":
    main()
