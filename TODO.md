# TODO

> Your next task is to solve the same sudoku dataset using photonic channels
> instead of qubits. Each cell is interpreted as a small recurrent linear
> optical circuit with one or two photons injected at each time step. Open a
> separate PR for this experiment. Again, let's be efficient in our use of
> the GPU. The previous experiment should help us narrow down the choice of
> dimensions for the ansatz and the unroll length, so that we run few
> tailored experiments on the GPU, making sure the contraction is doable
> beforehand.

- [WIP] @session_01GRttRw1nfYw86K7h4bfMRx-2026-08-14 17:00 Implement the
      photonic ansatz: cells as recurrent linear optical circuits -- a
      one-photon memory register recirculating between ticks, one or two
      fresh photons injected at each tick, an orthogonal interferometer
      mesh with feedforward control by the measured incoming digits,
      photon counting on the work modes with fixed pattern-to-digit
      lookups; constraints as one photon through a feedforward mesh with
      the verdict a designated-mode click. Fock-sector tensors via
      symmetric powers of the mesh unitary, differentiable in JAX.
- [ ] Keep the dimensions established by the previous experiment: dim-4
      measured message wires, two ticks, feedback rank two -- the proven
      contraction regime -- and confirm the widths on CPU before any GPU
      run. Two variants: measured memory (all-classical wires) and
      coherent memory (the memory photon interferes with the injected
      photons, doubled memory wire of dimension sixteen).
- [ ] Certify the solver inside the trainable mesh ansatz on CPU: mode
      permutations are Givens meshes at right angles, so digit broadcast
      and the all-different verdict routing are closed-form settings;
      the certificate must decode held-out puzzles exactly.
- [ ] CPU smoke training, then few tailored Modal runs: the two variants
      with structured and random initialisation at two ticks, one seed
      replicate for the headline.
- [ ] Report the results in a separate draft PR stacked on the sudoku
      experiment branch, against the qubit and CMap GNN numbers.
