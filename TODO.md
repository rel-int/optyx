# Metal tensor contraction benchmark

> Use optyx to do some fancy tensor contraction on the mac mini GPU to impress me
>
> open a PR so Alexis can look at it too

## Design

Construct a deterministic brickwork circuit

\[
\lvert\psi\rangle =
\prod_{d=1}^{18}
\left(\prod_{(i,j)\in E_d} CZ_{ij}\right)
\left(\bigotimes_{i=1}^{22} R_{P_{i,d}}(\theta_{i,d})\right)
H^{\otimes 22}\lvert 0^{22}\rangle,
\]

with alternating \(X/Z\) rotations and nearest-neighbour \(CZ\) matchings.
Compile the Optyx Kraus diagram to a Quimb tensor network, choose the
contraction tree with Cotengra, move its arrays to an MLX Metal device, and
contract the complete \(2^{22}\)-amplitude statevector. Compare the result
against the same contraction tree evaluated with NumPy.

The exploratory run on an Apple M4 produced 2,498 tensors and an estimated
3.01 billion-FLOP contraction. The cached Metal contraction took 25.8 ms
versus 153.1 ms with NumPy (5.94x), with relative L2 disagreement
\(1.48\times10^{-6}\) in complex64.

## Work

- [ ] Decide whether this belongs in `examples/` or as an optional benchmark.
- [ ] Add a deterministic, parameterised CPU/Metal contraction entry point.
- [ ] Keep MLX optional and report clearly when Metal is unavailable.
- [ ] Test circuit normalisation, CPU/GPU agreement, and benchmark metadata.
- [ ] Document dependencies, invocation, cold/warm timing, and interpretation.
- [ ] Run `pflake8 optyx` and `coverage run -m pytest`.
