# TODO

> Two problems: 1) I see you added a lot of pylint disables, inline comments are not allowed in optyx as in discopy! remove them and check the pylint, 2) The tensor backend is a large addition, write an issue and open a separate PR directly on main that proposes a new routine for evaluating optyx tensor contractions. The logic needs to be simple and flexible accross quimb, cotengra, jax and pytorch

## Tensor contraction routine

- [x] Implement issue #20 as one deterministic routine
  over ``discopy.tensor.Diagram`` for exact NumPy, JAX, PyTorch and Quimb
  contraction, with arbitrary Cotengra optimizers and optional bond limits.
- [x] Reuse the routine from the existing Quimb and
  DisCoPy evaluators, add
  concise documentation and tests, and run lint, pylint and coverage.
- [x] Open a draft pull request directly against ``main``
  and leave every
  checklist item complete.
- [x] Cover structural spider materialisation when the
  optional JAX and PyTorch test dependencies are unavailable in CI.
