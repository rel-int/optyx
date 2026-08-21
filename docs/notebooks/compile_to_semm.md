---
title: Compile To Semm
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# Example showing how to use the compiler

First start by creating an open graph

```python {.marimo}
import networkx as nx
```

```python {.marimo}
g = nx.Graph([(0, 1), (0, 2), (2, 1)])
nx.draw(g)
```

Create the open graph on top by assigning measurements for

```python {.marimo}
from optyx.compiler import OpenGraph, Measurement

meas = {i: Measurement(0.5 * i, "XY") for i in range (3)}
inputs = {0}
outputs = {2}

og = OpenGraph(g, meas, inputs, outputs)
```

Now compile the open graph to instructions that can be executed on a machine with a single resource state emitter and unlimited measurement devices

```python {.marimo}
from optyx.compiler.semm import compile_to_semm
compile_to_semm(og)
```