---
title: Zw Examples
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# Evaluating photonic circuits using the permanent method and tensor networks
<!---->
By representing the circuits in $\bf{ZW}_\infty$, we can treat them as a tensor networks which can be contracted using different methods. Here, we simply use the tensor class from DisCoPy, but tools such as Quimb could be used.

```python {.marimo}
import sys
sys.path.append('..')

from optyx import circuit
from optyx import qpath
from optyx import zw
import numpy as np
```

## Hong-Ou-Mandel effect
<!---->
We can define the optical circuit to represent the Hong-Ou-Mandel effect in QPath, then translate the circuit to $\bf{ZW}_\infty$:

```python {.marimo}
hom = qpath.Create(1, 1) >> circuit.BBS(0) >> qpath.Select(1, 1)
```

```python {.marimo}
hom.draw()
```

Convert the QPath diagram to a $\bf{ZW}_\infty$ diagram:

```python {.marimo}
hom.to_zw().draw()
```

Evaluating the permanent:

```python {.marimo}
hom.to_path().prob_with_perceval().array
```

```python {.marimo}
hom.to_path().prob().array
```

Evaluating the tensor network:

```python {.marimo}
hom.to_zw().to_tensor().eval().array
```

# Number operator

```python {.marimo}
number_operator = (qpath.Split() >> 
                   qpath.Select(1) @ qpath.Id(1) >> 
                   qpath.Create(1) @ qpath.Id(1) >> 
                   qpath.Merge())
```

```python {.marimo}
number_operator.draw()
```

```python {.marimo}
# Create a 2-photon state
two_p = qpath.Create(2) >> number_operator >> qpath.Select(2)
np.sqrt(two_p.prob().array)
```

```python {.marimo}
two_p.to_zw().to_tensor().eval().array
```

```python {.marimo}
# Create a 3-photon state
three_p = qpath.Create(3) >> number_operator >> qpath.Select(3)
np.sqrt(three_p.prob().array)
```

```python {.marimo}
three_p.to_zw().to_tensor().eval().array
```

```python {.marimo}
# Create a 12-photon state
n_p = qpath.Create(12) >> number_operator >> qpath.Select(12)
np.sqrt(n_p.prob().array)
```

```python {.marimo}
n_p.to_zw().to_tensor().eval().array
```

## More complicated circuit

```python {.marimo}
circ = (
    qpath.Create(1, 1, 1, 1) >> 
    circuit.BBS(0.3) @ circuit.MZI(0.34, 0.10) >>
    circuit.Phase(np.pi/2) @ qpath.Id(1) @ qpath.Id(1) @ circuit.Phase(np.pi/4)
)
```

```python {.marimo}
circ.draw()
```

```python {.marimo}
circ.to_zw().draw()
```

```python {.marimo}
diagram_qpath = circ
diagram_zw = diagram_qpath.to_zw()

prob_zw_ = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
prob_zw = zw.tn_output_2_perceval_output(prob_zw_, diagram_zw)

prob_perceval = diagram_qpath.to_path().prob_with_perceval().array
prob_eval = diagram_qpath.to_path().prob().array

assert np.allclose(prob_zw, prob_perceval)
```