# TODO

> Your next task is to solve the same sudoku dataset using photonic channels
> instead of qubits. Each cell is interpreted as a small recurrent linear
> optical circuit with one or two photons injected at each time step. Open a
> separate PR for this experiment. Again, let's be efficient in our use of
> the GPU. The previous experiment should help us narrow down the choice of
> dimensions for the ansatz and the unroll length, so that we run few
> tailored experiments on the GPU, making sure the contraction is doable
> beforehand.

- [x] Implement the
      photonic ansatz: cells as recurrent linear optical circuits -- a
      one-photon memory register recirculating between ticks, one or two
      fresh photons injected at each tick, an orthogonal interferometer
      mesh with feedforward control by the measured incoming digits,
      photon counting on the work modes with fixed pattern-to-digit
      lookups; constraints as one photon through a feedforward mesh with
      the verdict a designated-mode click. Fock-sector tensors via
      symmetric powers of the mesh unitary, differentiable in JAX.
- [x] Keep the dimensions established by the previous experiment: dim-4
      measured message wires, two ticks, feedback rank two -- the proven
      contraction regime -- and confirm the widths on CPU before any GPU
      run. Two variants: measured memory (all-classical wires) and
      coherent memory (the memory photon interferes with the injected
      photons, doubled memory wire of dimension sixteen).
- [x] Certify the solver inside the trainable mesh ansatz on CPU: mode
      permutations are Givens meshes at right angles, so digit broadcast
      and the all-different verdict routing are closed-form settings;
      the certificate must decode held-out puzzles exactly.
- [x] CPU smoke training, then few tailored Modal runs: the two variants
      with structured and random initialisation at two ticks, one seed
      replicate for the headline.
- [x] Report the results in a separate draft PR stacked on the sudoku
      experiment branch, against the qubit and CMap GNN numbers.

> Try to do it without classical feedforward , purely quantum, and if that
> doesn't work then you can try by controlling the unitary with the
> classical outcomes. Three models to bencark: only classical loop, only
> quantum loop, both classical and quantum loop. Max two runs each if you
> don't get the result, so we don't use too much gpu

> Let's run only the purely quantum model. Both the memory and the messages
> should be coherent (quantum links), and for the prediction we can use a
> photon counting measurement, possibly followed by some local classical
> post-processing. Go ahead and run this experiment on the modal GPUs

- [x] Benchmark the three loop models within two GPU runs each: the
      purely quantum model (coherent links, counters with learned local
      post-processing, a traced constraint ancilla) reaches 0.541 cell
      accuracy at 109 parameters on the local light cones; the measured
      model solves at 648; the mixed model reaches 0.637 at 872.

> How can we scale up the pure quantum photonic experiment? The number of
> parameters seems too low to hope to solve the task, at the same time the
> contractions need to be doable on the GPU, could truncation help?

- [x] Implement the scaled purely quantum family: internal traced
      vacuum ancillas, complex phased-Givens meshes, more sweeps, and
      fixed pure-loss channels of transmittivity at least 0.9 on the
      coherent links -- all growing boxes rather than wires, verified
      to reduce exactly to the pure family.
- [x] Probe the width side: Fock cutoff two on the message wires makes
      the doubled cell tensor a 4.4 GB dense array before contraction,
      out of reach of the A100 -- its budget slot went to a bigger mesh
      (twelve-mode cells, eight sweeps, 1,520 parameters) instead.
- [x] Run the four-run GPU ladder at two ticks on the local light
      cones: 680 parameters lossless 0.463, with fixed loss 0.95
      0.459, 1,520 parameters 0.482, trained twice as long 0.396 --
      the 109-parameter plateau of 0.541 does not move, so the
      bottleneck is the coherent structure, not the parameter count.
