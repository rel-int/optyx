---
title: Distributed 412 Code
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

```python {.marimo}
import numpy as np
from optyx.channel import Ty
from optyx import (
    CQMap
)
```

```python {.marimo}
from optyx.qubits import BitFlipError, Bra, Channel, DephasingError, Diagram, Discard, DiscardChannel, H, Id, Ket, Measure, Scalar, X, Z, bit, diagram, qubit, zx
```

```python {.marimo}
Ket_0 = Ket(0)
Ket_1 = Ket(1)
Ket_plus = Ket('+')
Ket_minus = Ket('-')
Bra_0 = Bra(0)
Bra_1 = Bra(1)
Bra_plus = Bra('+')
Bra_minus = Bra('-')
Bit_0 = Bra(0, bit)
Bit_1 = Bra(1, bit)
CNOT = Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1) @ Scalar(np.sqrt(2))
NOTC = X(1, 2) @ Id(1) >> Id(1) @ Z(2, 1) @ Scalar(np.sqrt(2))
_X_ = X(1, 1, 0.5)
_Z_ = Z(1, 1, 0.5)
H_ = H()
```

### Encoder for the [4,1,2] code

```python {.marimo}
_encoder = Z(1, 2) >> X(1, 2) @ X(1, 2)
logical_0 = X(0, 1) @ Scalar(1 / np.sqrt(2)) >> _encoder
# Defining the logical |0> and <0|
logical_0_measurement = logical_0.dagger()
logical_plus = Z(0, 1) @ Scalar(1 / np.sqrt(2)) >> _encoder
logical_plus_measurement = logical_0.dagger()
# Defining the logical |+> and <+|
(logical_0 @ logical_plus).draw()
```

```python {.marimo}
X_module = Id(1) @ CNOT >> Diagram.permutation([1, 0, 2], qubit ** 3) >> Id(1) @ CNOT >> Diagram.permutation([1, 0, 2], qubit ** 3)
X_parity_check = Id(2) @ Ket_0 >> X_module >> Id(2) @ Measure(1)
Z_parity_check = H_ @ H_ @ Ket_0 >> X_module >> H_ @ H_ @ Measure(1)

X_detector = X_parity_check @ X_parity_check >> Diagram.permutation([0,1,3,4,2,5], qubit ** 2 @ bit @ qubit ** 2 @ bit)
Z_detector = Z_parity_check @ Z_parity_check >> Diagram.permutation([0,1,3,4,2,5], qubit ** 2 @ bit @ qubit ** 2 @ bit)

Z_detector.draw()
```

### Defining the circuit with probabilistic bit-flips on data qubits

```python {.marimo}
p_error = 0.95
Imperfect_circuit = logical_plus >> BitFlipError(p_error) ** 4 >> X_detector
Imperfect_circuit.draw()
```

### Computing the fidelity after one round of X error detection

```python {.marimo}
_P_s_eq_0 = Imperfect_circuit >> Discard(4) @ Bit_0 @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit >> logical_plus_measurement @ Bit_0 @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity:', np.sqrt(fidelity).real)
```

## Calculating the fidelity for the distributed [4,2,2] code with an imperfect Bell state on the network

### Defining the network Bell state from its density matrix

![Density Matrix](densityMatrix.png)

The state has a fidelity of 96.89(8)% to the |Ψ+⟩ Bell state.
https://www.nature.com/articles/s41586-024-08404-x

```python {.marimo}
bell_density_re = np.array([
    [0.012, 0.014, 0.014, 0.000],
    [0.014, 0.508, 0.475, 0.008],
    [0.014, 0.475, 0.479, 0.009],
    [0.000, 0.008, 0.009, 0.000]
])
bell_density_im = np.exp(1j * np.pi * np.array([
    [0.000, -1.850, -1.825, -0.985],
    [1.850, 0.000, -0.002, -0.902],
    [1.825, 0.002, 0.000, -0.931],
    [0.985, 0.902, 0.931, 0.000]
]))

bell_density = np.multiply(bell_density_re, bell_density_im)
np.set_printoptions(precision=3)
print(bell_density)
```

```python {.marimo}
from discopy import tensor

bell = diagram.Box(name="Bell", dom=diagram.bit ** 2, cod=diagram.bit ** 2)
bell.array = bell_density
bell.determine_output_dimensions = lambda input_dims: input_dims
bell.truncation = lambda input_dims, output_dims: tensor.Box("Bell", dom=tensor.Dim(2)**2, cod=tensor.Dim(2)**2, data=bell_density)
bell = (
    diagram.Spider(0, 2, typ=diagram.bit) >>
    diagram.Id(diagram.bit) @ diagram.Spider(0, 2, typ=diagram.bit) @ diagram.Id(diagram.bit) >>
    diagram.Diagram.permutation([0,1,3,2], diagram.bit**4) >>
    diagram.Id(diagram.bit ** 2) @ bell >>
    diagram.Diagram.permutation([0,2,1,3], diagram.bit**4)
)

Noisy_bell = CQMap('Physical Bell', bell, dom=Ty(), cod=qubit ** 2)
```

#### Verifying fidelity

```python {.marimo}
Perfect_Bell = Channel("Perfect Bell", diagram.Spider(0,2,typ=diagram.bit) @ diagram.Scalar(1 / np.sqrt(2)))
Perfect_Bell_Effect = Channel("Perfect Bell Effect", diagram.Id(diagram.bit) @ zx.X(1,1,0.5) >> diagram.Spider(2,0,typ=diagram.bit) @ diagram.Scalar(1 / np.sqrt(2)))

print("Calculated fidelity:", (Noisy_bell >> Perfect_Bell_Effect).double().to_tensor().eval().array.real)
```

### Defining the circuit for the distributed physical implementation

```python {.marimo}
from optyx import classical
cnot = Id(1) @ X(1, 2) >> Z(2, 1) @ Id(1)
left_module_ = Id(1) @ cnot >> Diagram.permutation([1, 0, 2], qubit ** 3) >> Id(1) @ cnot >> Diagram.permutation([1, 0, 2], qubit ** 3)
notc = X(1, 2) @ Id(1) >> Id(1) @ Z(2, 1)
right_module_ = notc @ Id(1) >> Diagram.permutation([0, 2, 1], qubit ** 3) >> notc @ Id(1) >> Diagram.permutation([0, 2, 1], qubit ** 3)
left_module = left_module_ >> Id(2) @ Measure(1)
right_module = right_module_ >> Measure(1) @ Id(2)
_X_ = X(1, 1, 0.5)
_Z_ = Z(1, 1, 0.5)
H__1 = H()
Left_X_module = left_module
Right_X_module = right_module
Left_Z_module = H__1 ** 2 @ Id(1) >> left_module_ >> H__1 ** 2 @ Id(1) >> Id(2) @ Measure(1)
Right_Z_module = Id(1) @ H__1 ** 2 >> right_module_ >> Id(1) @ H__1 ** 2 >> Measure(1) @ Id(2)
ClassicalDetector = classical.X(2, 1, 0)
ClassicalCopy = classical.X(2, 1, 0)
Physical_X_detector = Id(2) @ (Noisy_bell >> Id(1) @ _X_ >> H__1 @ H__1) @ Id(2) >> Left_X_module @ Right_X_module >> Id(2) @ ClassicalDetector @ Id(2) >> Diagram.permutation([0, 1, 3, 4, 2], qubit ** 2 @ bit @ qubit ** 2)
Physical_Z_detector = Id(2) @ (Noisy_bell >> Id(1) @ _X_ >> H__1 @ H__1) @ Id(2) >> Left_Z_module @ Right_Z_module >> Id(2) @ ClassicalDetector @ Id(2) >> Diagram.permutation([0, 1, 3, 4, 2], qubit ** 2 @ bit @ qubit ** 2)
Physical_X_detector.draw()
```

### Calculating fidelity for Z error detection

```python {.marimo}
_encoder = Z(1, 2) @ Z(1, 2) @ Z(0, 4) >> Diagram.permutation([0, 2, 4, 1, 5, 3, 6, 7], qubit ** 8) >> X(3, 1) @ X(2, 1) @ X(2, 1) @ Id(1)
logical_bell = Z(0, 2) @ Scalar(2) >> _encoder
logical_bell_measurement = logical_bell.dagger()
Imperfect_circuit_1 = logical_bell >> DephasingError(p_error) ** 4 >> Physical_Z_detector
Imperfect_circuit_1.draw()
```

```python {.marimo}
_P_s_eq_0 = Imperfect_circuit_1 >> Discard(4) @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit_1 >> logical_bell_measurement @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity_1 = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity for Z error detection:', np.sqrt(fidelity_1.real))
```

### Calculating fidelity for a whole round of Z and X error detection

```python {.marimo}
Imperfect_circuit_2 = logical_bell >> DephasingError(p_error) ** 4 >> Physical_Z_detector >> BitFlipError(p_error) ** 4 @ Diagram.id(bit) >> Physical_X_detector @ Diagram.id(bit)
_P_s_eq_0 = Imperfect_circuit_2 >> Discard(4) @ Bit_0 @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit_2 >> logical_bell_measurement @ Bit_0 @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity_2 = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity a whole round of error detection:', np.sqrt(fidelity_2.real))
```

```python {.marimo}
Imperfect_circuit_3 = logical_bell >> DephasingError(p_error) ** 4 >> Physical_Z_detector
Imperfect_circuit_3.draw()
```

## Two-qubit bit-flip with a two-qubit phase-flip repetition code

```python {.marimo}
Left_Z_detector = Id(2) @ Ket_0 >> Left_Z_module
Logical_ket_0 = Z(0, 2) @ Scalar(1 / np.sqrt(2))
Logical_bra_0 = Z(2, 0) @ Scalar(1 / np.sqrt(2))
Imperfect_circuit_4 = Logical_ket_0 >> DephasingError(p_error) @ DephasingError(p_error) >> Left_Z_detector
Imperfect_circuit_4.draw()
```

```python {.marimo}
_P_s_eq_0 = Imperfect_circuit_4 >> Discard(2) @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit_4 >> Logical_bra_0 @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity_3 = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity a whole round of error detection:', fidelity_3.real)
```

```python {.marimo}
Z_repetition_detector = Left_Z_detector @ Left_Z_detector >> Diagram.permutation([0, 1, 3, 4, 2, 5], qubit ** 2 @ bit @ qubit ** 2 @ bit)
X_repetition_detector = Physical_X_detector
Imperfect_circuit_5 = Logical_ket_0 @ Logical_ket_0 >> DephasingError(p_error) ** 4 >> Z_repetition_detector
Imperfect_circuit_5.draw()
```

```python {.marimo}
_P_s_eq_0 = Imperfect_circuit_5 >> Discard(4) @ Bit_0 @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit_5 >> Logical_bra_0 @ Logical_bra_0 @ Bit_0 @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity_4 = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity a whole round of error detection:', fidelity_4.real)
```

```python {.marimo}
logical_ket_00 = Z(0, 4) @ Scalar(1 / np.sqrt(2))
logical_bra_00 = Z(4, 0) @ Scalar(1 / np.sqrt(2))
Imperfect_circuit_6 = logical_ket_00 >> DephasingError(p_error) ** 4 >> Z_repetition_detector >> BitFlipError(p_error) ** 4 @ Diagram.id(bit ** 2) >> Physical_X_detector @ Diagram.id(bit ** 2)
_P_s_eq_0 = Imperfect_circuit_6 >> Discard(4) @ Bit_0 @ Bit_0 @ Bit_0
_p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
_P_00_and_s_eq_0 = Imperfect_circuit_6 >> logical_bra_00 @ Bit_0 @ Bit_0 @ Bit_0
_p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
fidelity_5 = _p_00_and_s_eq_0 / _p_s_eq_0
print('Calculated fidelity a whole round of error detection:', fidelity_5.real)
Imperfect_circuit_6.draw()
```

```python {.marimo}
def fidelity_6(input_state, p_error=0.95, n_rounds=1):
    Imperfect_circuit = input_state >> DephasingError(p_error) ** 4 >> Z_repetition_detector >> BitFlipError(p_error) ** 4 @ Diagram.id(bit ** 2) >> Physical_X_detector @ Diagram.id(bit ** 2)
    _P_s_eq_0 = Imperfect_circuit >> Discard(4) @ Bit_0 @ Bit_0 @ Bit_0
    _p_s_eq_0 = _P_s_eq_0.double().to_tensor().eval().array
    _P_00_and_s_eq_0 = Imperfect_circuit >> input_state.dagger() @ Bit_0 @ Bit_0 @ Bit_0
    _p_00_and_s_eq_0 = _P_00_and_s_eq_0.double().to_tensor().eval().array
    fidelity = _p_00_and_s_eq_0 / _p_s_eq_0
    return fidelity.real
fidelity_6(logical_ket_00, 0.95, 2)
```

```python {.marimo}
xs = np.linspace(0.9, 1, 20)
ys = [fidelity_6(logical_ket_00, error) for error in xs]
```

```python {.marimo}
import matplotlib.pyplot as plt

plt.plot(xs, ys)
plt.plot(xs, xs)

plt.title("Fidelities after error detection")
plt.xlabel("Physical fidelity")
plt.ylabel("Logical fidelity")
plt.show()
```

```python {.marimo}
xs_1 = np.linspace(0.99, 1, 10)
ys_1 = [fidelity_6(logical_ket_00, error) for error in xs_1]
```

```python {.marimo}
plt.plot(xs_1, ys_1)
plt.plot(xs_1, xs_1)
plt.title('Fidelities after error detection')
plt.xlabel('Physical fidelity')
plt.ylabel('Logical fidelity')
plt.show()
```

```python {.marimo}
#(Ket_plus**4 >> Physical_X_detector >> H_**4).double().to_tensor().eval().array
```

```python {.marimo}
(Ket_plus**4 >> Physical_X_detector ).draw()
```

```python {.marimo}
Left_X_module.double().draw()
```

```python {.marimo}
Physical_X_detector_1 = Id(2) @ (Perfect_Bell >> H__1 @ H__1) @ Id(2) >> Left_X_module @ Right_X_module >> Id(2) @ DiscardChannel(bit ** 2) @ Id(2)
Physical_X_detector_1.double().draw()
```