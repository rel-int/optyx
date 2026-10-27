---
title: Equilibrium Propagation
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# Equilibrium propagation on photonic loops

*Training a photonic network with feedback from a nudged run of the same
device, and when that gives the gradient.*

A diagram with feedback, fed by a source at every tick, settles into a
stationary regime: its memory forgets where it started, and every
observable of the output stream has a stationary mean. Training such a
device means following the gradient of a stationary cost, here a weighted
sum of the mean counts of its outputs,

$$J(\theta) = c \cdot o(\theta).$$

Backpropagating through the loop means differentiating through every tick
the light has gone round. Equilibrium propagation asks the device instead:
nudge the sources by the error, $q \to q \pm \beta c$, and the response of
the memory, $\hat\lambda = (g_{+\beta} - g_{-\beta}) / 2\beta$, plays the
part of the adjoint of the loop. Contracted with the derivative of a single
tick, it estimates the gradient:

$$\hat\partial_a J = c \cdot \partial_a o + \hat\lambda \cdot \partial_a g'.$$

This is exact when the response of the means is reciprocal. It is
reciprocal when the mode matrix of a tick is symmetric and the memory is
dephased at the cut, because the map of mean photon numbers over one tick
is then $|U_{ij}|^2$, a symmetric matrix. This notebook checks that, then
measures how far the estimate drifts when the memory keeps its coherence,
when the optics is not reciprocal, and when the detectors feed their
counts forward into the next tick.

```python {.marimo}
import numpy as np
from optyx import classical
from optyx.channel import Diagram, bit, mode, qmode
from optyx.photonic import (
    Create, Dephasing, MZI, NumberResolvingMeasurement, Phase, PhotonLoss)
from optyx.sampling import Coherent, Fock, Thermal
from optyx.equilibrium import Equilibrium

import warnings

warnings.filterwarnings("ignore", message="Trial error")  # cotengra probes
```

## A loop of two memory modes and one port

The step is a palindrome: the layers that act on the port and the two
memory modes, then the same layers in reverse, between losses on the
memory. Each layer is a symmetric matrix, so the step is symmetric, and the
port is where light both enters and leaves. With `dephased`, the memory
loses its coherence at the cut, and with `reciprocal=False` the second half
is a different interferometer.

```python {.marimo}
def palindrome(dephased=True, reciprocal=True, feedforward=False):
    def loop(theta):
        layers = [MZI(theta[0], 0) @ qmode, qmode @ MZI(theta[1], 0),
                  qmode @ qmode @ Phase(theta[2])]
        back = layers[::-1] if reciprocal else [
            MZI(.1, .3) @ qmode, qmode @ MZI(.4, .7)]
        loss = qmode @ PhotonLoss(.8) @ PhotonLoss(.8)
        start = [qmode @ Dephasing() @ Dephasing()] if dephased else []
        if not feedforward:
            return Diagram.then(*start, loss, *layers, *back, loss).feedback(
                mem=qmode @ qmode, state=Create(0, 0))
        middle = qmode @ qmode @ Diagram.swap(qmode, bit) \
            >> qmode @ qmode @ classical.BitControlledGate(Phase(.25))
        parity = classical.ClassicalFunction(lambda n: [n[0] % 2], mode, bit)
        readout = NumberResolvingMeasurement(1) @ qmode @ qmode \
            >> classical.CopyN(2) @ qmode @ qmode \
            >> mode @ parity @ qmode @ qmode \
            >> mode @ Diagram.permutation([1, 2, 0], bit @ qmode @ qmode)
        return Diagram.then(
            *[box @ bit for box in start + [loss] + layers], middle, *back,
            loss, readout).feedback(
                mem=qmode @ qmode @ bit,
                state=Create(0, 0) @ classical.Bit(0))
    return loop


THETA = (.2, .3, .1)
palindrome()(THETA).one_step()
```

## Passive loops, exactly

Without feed-forward, the mean photon numbers are one-particle quantities:
they follow the linear Stein equation of the loop's one-particle
correlation matrix, whatever the statistics of the source and the
indistinguishability of its photons. So `exact=True` computes both the
gradient and the estimate with no sampling. The table gives the cosine
between the two and the largest error relative to the norm of the gradient.

```python {.marimo}
def compare(estimate, gradient):
    return (estimate @ gradient / np.linalg.norm(estimate)
            / np.linalg.norm(gradient),
            np.max(np.abs(estimate - gradient)) / np.linalg.norm(gradient))


exact_rows = []
for _reciprocal in (True, False):
    for _dephased in (True, False):
        _ep = Equilibrium(palindrome(_dephased, _reciprocal), Fock((1, )),
                          cost=(1., ))
        _cosine, _error = compare(_ep.propagate(THETA, exact=True),
                                  _ep.gradient(THETA, exact=True))
        exact_rows.append(
            f"| {_reciprocal} | {_dephased} | {_cosine:.4f} | {_error:.1e} |")
mo.md("| reciprocal | dephased | cosine | relative error |\n|---|---|---|---|\n"
      + "\n".join(exact_rows))
```

The same numbers come out for any source: here the stationary cost of the
reciprocal dephased loop under four of them.

```python {.marimo}
mo.md(", ".join(
    f"{name}: {Equilibrium(palindrome(), source, (1., ), indistinguishability=p).value(THETA, exact=True):.6f}"
    for name, source, p in [("Fock", Fock((1, )), 1.),
                            ("distinguishable Fock", Fock((1, )), 0.),
                            ("coherent", Coherent((1., )), 1.),
                            ("thermal", Thermal((1., )), 1.)]))
```

Dephasing is what makes the estimate exact. With the coherence of the
memory kept, the means at the cut no longer determine the next tick:
$\partial_a g'$ depends on the off-diagonal correlations, while the nudged
response only sees their diagonal. The estimate then misses the gradient,
by a cosine of about 0.3 here. A non-reciprocal step that is dephased still
gives a good estimate, but not an exact one.

## Feed-forward, sampled

With feed-forward the means are no longer linear in the source, and the
statistics of the photons matter. Below, the parity of each tick's count
sets a phase of $\pi/2$ in the middle of the next tick. The run is sampled
by the `Unravelling`, with common random numbers between the runs it
compares. It is kept short so that the notebook renders quickly, so its
error bars are large.

```python {.marimo}
_ep = Equilibrium(palindrome(feedforward=True), Fock((1, )), cost=(1., ),
                  indistinguishability=.5, cap=8)
_sampling = dict(ticks=60, burn_in=30, seed=0, shots=4)
sampled = (_ep.gradient(THETA, step=.05, **_sampling),
           _ep.propagate(THETA, beta=.3, step=.02, **_sampling))
mo.md(f"gradient {np.round(sampled[0], 3)}, estimate {np.round(sampled[1], 3)}")
```

GRID
