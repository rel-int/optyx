## Pairs separated by the protocol against the budget (any of four cells)

| Model | T = 1e+03 | T = 1e+04 | T = 1e+05 | T = 1e+06 | T = 1e+07 | exact |
|---|---|---|---|---|---|---|
| coherent light, w = 100 | 5 | 7 | 12 | 28 | 30 | - |
| coherent light, w = 1 | 1 | 1 | 3 | 18 | 29 | - |
| GNN, noise every step, tanh | 0 | 9 | 1 | 1 | 5 | - |
| GNN, noise every step, linear | 1 | 5 | 3 | 2 | 9 | 40 |
| GNN, noise at init, tanh | 26 | 28 | 30 | 33 | 36 | - |
| GNN, noise at init, linear | 26 | 28 | 29 | 33 | 33 | 40 |
| pairs run | 49 | 49 | 49 | 49 | 49 | |

## Pairs separated per class at T = 1e+07, protocol, any cell

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 10 | 6 | 0 | 10 | 4 | 0 | 0 | 30 |
| coherent light, w = 1 | 10 | 6 | 0 | 10 | 3 | 0 | 0 | 29 |
| GNN, noise every step, tanh | 0 | 2 | 2 | 0 | 1 | 0 | 0 | 5 |
| GNN, noise every step, linear | 3 | 2 | 0 | 1 | 1 | 0 | 2 | 9 |
| GNN, noise at init, tanh | 10 | 6 | 4 | 10 | 6 | 0 | 0 | 36 |
| GNN, noise at init, linear | 10 | 6 | 1 | 10 | 6 | 0 | 0 | 33 |
| pairs | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |

## Pairs separated by BREC's test on sampled copies (1e5 samples per copy), any cell

| Model | eps = 1e-7 | eps at the sampling noise | unreliable (cells) |
|---|---|---|---|
| coherent light, w = 100 | 12 | 12 | 0 |
| coherent light, w = 1 | 1 | 1 | 0 |
| GNN, noise every step, tanh | 0 | 0 | 0 |
| GNN, noise every step, linear | 0 | 0 | 0 |
| GNN, noise at init, tanh | 0 | 27 | 0 |
| GNN, noise at init, linear | 0 | 27 | 0 |

## Smallest budget that separates (median over pairs the model separates at some budget, best cell)

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 6e+05 (10) | 1e+05 (6) | 1e+03 (3) | 1e+06 (10) | 5e+06 (4) | 1e+04 (1) | 5e+04 (4) |
| coherent light, w = 1 | 1e+06 (10) | 1e+06 (6) | 1e+05 (1) | 1e+06 (10) | 1e+07 (3) | - | - |
| GNN, noise every step, tanh | - | 5e+06 (4) | 1e+07 (3) | 1e+06 (1) | 1e+04 (5) | - | 1e+04 (2) |
| GNN, noise every step, linear | 1e+07 (3) | 1e+07 (2) | 1e+05 (1) | 5e+06 (2) | 1e+04 (4) | - | 5e+06 (4) |
| GNN, noise at init, tanh | 1e+03 (10) | 1e+03 (6) | 1e+07 (4) | 1e+03 (10) | 1e+06 (9) | 1e+06 (2) | - |
| GNN, noise at init, linear | 1e+03 (10) | 1e+03 (6) | 1e+07 (1) | 1e+03 (10) | 1e+06 (8) | 1e+06 (2) | - |

## Linear against tanh at T = 1e+07

| Noise | both | linear only | tanh only | neither |
|---|---|---|---|---|
| every step | 1 | 8 | 4 | 36 |

tanh only: CFI_73, Regular_10, Strongly_regular_10, Strongly_regular_5

| at init | 33 | 0 | 3 | 13 |

tanh only: Strongly_regular_0, Strongly_regular_30, Strongly_regular_5


## Coherent light at w = 100 against each network at T = 1e+07

| Network | both | coherent only | network only |
|---|---|---|---|
| coherent light, w = 1 | 29 | 1 | 0 |
| GNN, noise every step, tanh | 2 | 28 | 3 |
| GNN, noise every step, linear | 6 | 24 | 3 |
| GNN, noise at init, tanh | 29 | 1 | 7 |
| GNN, noise at init, linear | 29 | 1 | 4 |

## False-positive rate of the protocol on G against relabelled G (10 runs per cell)

| Model | T = 1e+03 | T = 1e+04 | T = 1e+05 | T = 1e+06 |
|---|---|---|---|---|
| coherent light, w = 100 | 0.021 (24) | 0.025 (24) | 0.021 (24) | 0.029 (24) |
| coherent light, w = 1 | 0.013 (24) | 0.033 (24) | 0.029 (24) | 0.033 (24) |
| GNN, noise every step, tanh | 0.008 (24) | 0.029 (24) | 0.029 (24) | 0.025 (24) |
| GNN, noise every step, linear | 0.008 (24) | 0.029 (24) | 0.037 (24) | 0.067 (24) |
| GNN, noise at init, tanh | 0.008 (24) | 0.013 (24) | 0.025 (24) | 0.013 (24) |
| GNN, noise at init, linear | 0.008 (24) | 0.008 (24) | 0.021 (24) | 0.017 (24) |

## GPU seconds per pair and cell, all budgets and the Hotelling copies

| Model | median | max |
|---|---|---|
| coherent light, w = 100 | 16 | 212 |
| coherent light, w = 1 | 16 | 208 |
| GNN, noise every step, tanh | 19 | 111 |
| GNN, noise every step, linear | 19 | 115 |
| GNN, noise at init, tanh | 77 | 143 |
| GNN, noise at init, linear | 73 | 143 |