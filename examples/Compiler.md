---
title: Compiler
marimo-version: 0.24.0
header: |-
  # /// script
  # dependencies = ["networkx"]
  # ///
---

```python {.marimo}
import marimo as mo
```

# Example showing how to use the compiler

First start by creating an open graph

```python {.marimo}
# packages added via marimo's package management: networkx !pip install networkx
```

```python {.marimo}
import networkx as nx
```

```python {.marimo}
g = nx.Graph([(0, 1), (0, 2), (2, 1)])
nx.draw(g)
```

```python {.marimo}
from optyx.compiler import OpenGraph, Measurement

meas = {i: Measurement(i) for i in range (2)}
inputs = [0]
outputs = [2]

og = OpenGraph(g, meas, inputs, outputs)
```

```python {.marimo}
from optyx.compiler.semm import compile_to_semm
compile_to_semm(og)
```