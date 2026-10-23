## Pairs separated by the protocol against the budget (any cell, alpha / (J cells))

| Model | T = 1e+03 | T = 1e+04 | T = 1e+05 | T = 1e+06 | T = 1e+07 | T = 1e+08 | exact |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 2 | 1 | 7 | 20 | 28 | 29 | - |
| GNN, noise every step, tanh | 0 | 1 | 2 | 13 | 25 | 29 | - |
| GNN, noise every step, linear | 1 | 2 | 11 | 25 | 29 | 31 | 40 |
| GNN, noise at init, tanh | 2 | 5 | 14 | 23 | 29 | 31 | - |
| GNN, noise at init, linear | 2 | 5 | 16 | 25 | 29 | 31 | 40 |
| pairs run | 43 | 43 | 43 | 43 | 43 | 43 | |

## Pairs separated with the studentised profile added (retest from checkpoints or native), any cell

| Model | T = 1e+03 | T = 1e+04 | T = 1e+05 | T = 1e+06 | T = 1e+07 | T = 1e+08 |
|---|---|---|---|---|---|---|
| coherent light, w = 100 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 3 | 29 of 36 |
| GNN, noise every step, tanh | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 3 | 28 of 36 |
| GNN, noise every step, linear | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 3 | 31 of 36 |
| GNN, noise at init, tanh | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 3 | 31 of 36 |
| GNN, noise at init, linear | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 3 | 31 of 36 |

## Pairs separated per class at T = 1e+08, protocol, any cell

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 10 | 6 | 1 | 10 | 2 | 0 | 0 | 29 |
| GNN, noise every step, tanh | 10 | 6 | 2 | 10 | 1 | 0 | 0 | 29 |
| GNN, noise every step, linear | 10 | 6 | 3 | 10 | 2 | 0 | 0 | 31 |
| GNN, noise at init, tanh | 10 | 6 | 3 | 10 | 2 | 0 | 0 | 31 |
| GNN, noise at init, linear | 10 | 6 | 3 | 10 | 2 | 0 | 0 | 31 |
| pairs | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |

## Pairs separated by BREC's test on sampled copies (1e5 samples per copy), any cell

| Model | eps = 1e-7 | eps at the sampling noise | unreliable (cells) |
|---|---|---|---|
| coherent light, w = 100 | 11 | 11 | 0 |
| GNN, noise every step, tanh | 2 | 2 | 0 |
| GNN, noise every step, linear | 8 | 8 | 0 |
| GNN, noise at init, tanh | 26 | 27 | 0 |
| GNN, noise at init, linear | 26 | 27 | 0 |

## Smallest budget from which the pair stays separated at every larger budget (median over such pairs, best cell)

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 1e+06 (10) | 6e+05 (6) | 1e+08 (1) | 1e+06 (10) | 1e+07 (2) | - | - |
| GNN, noise every step, tanh | 1e+06 (10) | 1e+06 (6) | 6e+07 (2) | 1e+07 (10) | 1e+07 (1) | - | - |
| GNN, noise every step, linear | 1e+06 (10) | 1e+05 (6) | 1e+08 (3) | 1e+06 (10) | 6e+04 (2) | - | - |
| GNN, noise at init, tanh | 1e+05 (10) | 6e+04 (6) | 1e+07 (3) | 1e+06 (10) | 5e+07 (2) | - | - |
| GNN, noise at init, linear | 1e+05 (10) | 6e+04 (6) | 1e+07 (3) | 1e+06 (10) | 5e+07 (2) | - | - |

## Linear against tanh at T = 1e+07

| Noise | both | linear only | tanh only | neither |
|---|---|---|---|---|
| every step | 25 | 4 | 0 | 20 |
| at init | 29 | 0 | 0 | 20 |

## Coherent light at w = 100 against each network at T = 1e+07

| Network | both | coherent only | network only |
|---|---|---|---|
| GNN, noise every step, tanh | 24 | 4 | 1 |
| GNN, noise every step, linear | 28 | 0 | 1 |
| GNN, noise at init, tanh | 27 | 1 | 2 |
| GNN, noise at init, linear | 27 | 1 | 2 |

## Ideal sample counts (exact gap, estimated noise, all orders), median per class, best cell

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 9.7e+03 (10) | 3.2e+03 (6) | 4e+05 (6) | 1.8e+04 (10) | 7.9e+07 (9) | 9.8e+12 (2) | 2.2e+09 (6) |
| GNN, noise every step, linear | - | - | - | - | - | - | - |
| GNN, noise at init, linear | - | - | - | - | - | - | - |

**Ratio of a network's ideal count to coherent light's, median over pairs both have finite**

| Network | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| GNN, noise every step, linear | - | - | - | - | - | - | - |
| GNN, noise at init, linear | - | - | - | - | - | - | - |

## Synthetic batches from exact moments: pairs separated in at least half the draws, any cell, original statistic / with the studentised profile

| Model | 1e+03 | 1e+04 | 1e+05 | 1e+06 | 1e+07 | 1e+08 | 1e+09 | 1e+10 | 1e+11 | 1e+12 | 1e+13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 13 / 13 | 20 / 18 | 33 / 32 | 36 / 35 | 35 / 35 | 34 / 34 | 33 / 33 | 37 / 37 | 38 / 38 | 38 / 38 | 38 / 38 |
| GNN, noise every step, linear | 38 / 38 | 38 / 38 | 39 / 39 | 36 / 37 | 36 / 37 | 37 / 37 | 39 / 39 | 39 / 39 | 39 / 39 | 39 / 39 | 39 / 39 |
| GNN, noise at init, linear | 39 / 39 | 39 / 39 | 39 / 39 | 37 / 38 | 36 / 38 | 39 / 39 | 39 / 39 | 39 / 39 | 39 / 39 | 39 / 39 | 39 / 39 |

**Median budget to separate (smallest T separated in half the draws and at every larger budget), per class, original / studentised**

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 6e+04 (10) / 1e+05 (10) | 6e+04 (6) / 6e+04 (6) | 5e+06 (6) / 6e+06 (6) | 1e+05 (10) / 1e+05 (10) | 1e+10 (6) / 1e+10 (6) | - | - |
| GNN, noise every step, linear | 1e+03 (10) / 1e+03 (10) | 1e+03 (6) / 1e+03 (6) | 1e+03 (6) / 1e+03 (6) | 1e+03 (10) / 1e+03 (10) | 5e+07 (6) / 1e+03 (6) | - | 1e+03 (1) / 1e+03 (1) |
| GNN, noise at init, linear | 1e+03 (10) / 1e+03 (10) | 1e+03 (6) / 1e+03 (6) | 1e+03 (6) / 1e+03 (6) | 1e+03 (10) / 1e+03 (10) | 5e+07 (6) / 1e+03 (6) | - | 1e+03 (1) / 1e+03 (1) |

## GPU seconds per pair and cell, all budgets and the Hotelling copies

| Model | median | max |
|---|---|---|
| coherent light, w = 100 | 10 | 1383 |
| GNN, noise every step, tanh | 8 | 1310 |
| GNN, noise every step, linear | 8 | 1313 |
| GNN, noise at init, tanh | 8 | 1428 |
| GNN, noise at init, linear | 8 | 1318 |