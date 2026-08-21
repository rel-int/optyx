---
title: Bosonic Vqe 2
marimo-version: 0.24.0
---

```python {.marimo}
from optyx import photonic

circuit = photonic.Create(1, 1, 1) >> photonic.ansatz(3, 4) >> photonic.Id(2) @ photonic.Select(1)
circuit.draw()
```

```python {.marimo}
from optyx import Diagram

matrix = [
    [0, 1],
    [1, 0]
]

n = len(matrix)
assert len(matrix[0]) == n

terms = []

for i in range(n):
    for j in range(n):
        term = Diagram.from_bosonic_operator(
            n_modes=n,
            operators=[(i, False), (j, True)],
            scalar=matrix[i][j]
        )
        terms.append(term)

hamiltonian = Diagram.sum_factory(terms)

hamiltonian.draw()
```

```python {.marimo}
expectation = circuit >> hamiltonian >> circuit.dagger()
```

```python {.marimo}
from optyx.core.backends import PermanentBackend

def to_float(x):
    if isinstance(x, complex):
        assert x.imag < 1e-10, x
        return x.real
    return x

free_syms = list(expectation.free_symbols)

f_exp = lambda xs: to_float(
    expectation.lambdify(*free_syms)(*xs)
    .eval(PermanentBackend())
    .tensor
    .array
)

def d_f_exp(xs):
    return [
        expectation.grad(s).lambdify(*free_syms)(*xs)
        .eval(PermanentBackend())
        .tensor
        .array
        for s in free_syms
    ]
```

```python {.marimo}
from tqdm import tqdm

xs = []
fxs = []
dfxs = []

def optimize(x0):
    x = x0

    for _ in tqdm(range(10)):
        fx = f_exp(x)
        dfx = d_f_exp(x)

        xs.append(x[::])
        fxs.append(fx)
        dfxs.append(dfx)
        for i, dfxx in enumerate(dfx):
            x[i] = to_float(x[i] - 0.01 * dfxx)

    xs.append(x[::])
    fxs.append(f_exp(x))
    dfxs.append(d_f_exp(x))
```

```python {.marimo}
optimize([0.1] * len(free_syms))
```

```python {.marimo}
import matplotlib.pyplot as plt

plt.plot(range(len(xs)),fxs,'b.')
plt.show()
```