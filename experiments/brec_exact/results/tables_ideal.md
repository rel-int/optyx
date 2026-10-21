## Exact rule on full BREC (relative gap > 1e-12, any of four cells, orders up to 4 below 65 vertices, 3 above)

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|
| indistinguishable photons | 60 | 50 | 50 | 100 | 76 | 20 | 19 | 375 |
| distinguishable photons | 60 | 50 | 50 | 100 | 71 | 20 | 19 | 370 |
| coherent light | 60 | 50 | 50 | 100 | 73 | 20 | 19 | 372 |
| thermal light | 60 | 50 | 33 | 100 | 61 | 0 | 1 | 305 |
| GNN, noise every step, linear | 60 | 50 | 33 | 100 | 51 | 0 | 1 | 295 |
| GNN, noise at init, linear | 60 | 50 | 33 | 100 | 44 | 0 | 1 | 288 |
| pairs | 60 | 50 | 50 | 100 | 100 | 20 | 20 | 400 |

## Pairs the photonic sources and the linear networks separate, exact rule

| Comparison | both | first only | second only | neither |
|---|---|---|---|---|
| coherent light vs GNN, noise every step, linear | 295 | 77 | 0 | 28 |
| coherent light vs GNN, noise at init, linear | 288 | 84 | 0 | 28 |
| indistinguishable photons vs GNN, noise every step, linear | 295 | 80 | 0 | 25 |
| distinguishable photons vs GNN, noise every step, linear | 295 | 75 | 0 | 30 |
| thermal light vs GNN, noise every step, linear | 295 | 10 | 0 | 95 |
| GNN, noise every step, linear vs GNN, noise at init, linear | 288 | 7 | 0 | 105 |

Coherent light only: ['4-vertex_condition_0', '4-vertex_condition_1', '4-vertex_condition_10', '4-vertex_condition_11', '4-vertex_condition_12', '4-vertex_condition_13', '4-vertex_condition_14', '4-vertex_condition_15', '4-vertex_condition_16', '4-vertex_condition_17', '4-vertex_condition_18', '4-vertex_condition_19', '4-vertex_condition_2', '4-vertex_condition_3', '4-vertex_condition_4', '4-vertex_condition_5', '4-vertex_condition_6', '4-vertex_condition_7', '4-vertex_condition_8', '4-vertex_condition_9', 'CFI_0', 'CFI_1', 'CFI_15', 'CFI_20', 'CFI_22', 'CFI_26', 'CFI_28', 'CFI_3', 'CFI_33', 'CFI_4', 'CFI_52', 'CFI_60', 'CFI_61', 'CFI_62', 'CFI_63', 'CFI_65', 'CFI_66', 'CFI_67', 'CFI_69', 'CFI_75', 'CFI_78', 'CFI_79', 'Distance_regular_1', 'Distance_regular_10', 'Distance_regular_11', 'Distance_regular_12', 'Distance_regular_13', 'Distance_regular_14', 'Distance_regular_15', 'Distance_regular_16', 'Distance_regular_17', 'Distance_regular_18', 'Distance_regular_19', 'Distance_regular_2', 'Distance_regular_4', 'Distance_regular_5', 'Distance_regular_6', 'Distance_regular_7', 'Distance_regular_8', 'Distance_regular_9', 'Strongly_regular_1', 'Strongly_regular_13', 'Strongly_regular_15', 'Strongly_regular_17', 'Strongly_regular_18', 'Strongly_regular_19', 'Strongly_regular_20', 'Strongly_regular_22', 'Strongly_regular_23', 'Strongly_regular_24', 'Strongly_regular_25', 'Strongly_regular_26', 'Strongly_regular_29', 'Strongly_regular_31', 'Strongly_regular_32', 'Strongly_regular_33', 'Strongly_regular_34']

A network only: []

## Separation order, pairs per order: coherent light / step-lin / init-lin

| Class | order 1 | order 2 | order 3 | order 4 | none |
|---|---|---|---|---|---|
| Basic | 60 / 60 / 60 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Regular | 50 / 50 / 50 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Str. regular | 0 / 0 / 0 | 0 / 0 / 0 | 50 / 0 / 0 | 0 / 33 / 33 | 0 / 17 / 17 |
| Extension | 100 / 100 / 99 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| CFI | 6 / 6 / 0 | 46 / 19 / 7 | 17 / 19 / 20 | 4 / 7 / 17 | 27 / 49 / 56 |
| 4-vertex | 0 / 0 / 0 | 0 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 20 / 20 |
| Dist. regular | 0 / 0 / 0 | 0 / 0 / 0 | 16 / 0 / 0 | 3 / 1 / 1 | 1 / 19 / 19 |

## Ideal sample counts, exact gap over the estimated noise, all orders, best cell: median per class (pairs with a finite count)

| Model | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular |
|---|---|---|---|---|---|---|---|
| coherent light, w = 100 | 9.4e+03 (60) | 3.9e+03 (50) | 7.1e+05 (50) | 8.4e+03 (100) | 2.9e+11 (100) | 6.5e+12 (20) | 3e+10 (20) |
| GNN, noise every step, linear | 3.9e+04 (60) | 6.7e+04 (50) | 1e+05 (50) | 7.4e+04 (100) | 3.7e+10 (100) | 2.6e+25 (20) | 7.2e+25 (20) |
| GNN, noise at init, linear | 3.5e+03 (60) | 1e+04 (50) | 5.9e+04 (50) | 6.2e+03 (100) | 4e+29 (100) | 9.2e+29 (20) | 1.2e+31 (20) |

**Ratio of a network's ideal count to coherent light's at w = 100, median over pairs where both are finite (pairs)**

| Network | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | all |
|---|---|---|---|---|---|---|---|---|
| GNN, noise every step, linear | 3.89 (60) | 18.7 (50) | 0.197 (50) | 11 (100) | 0.0211 (100) | 3.55e+12 (20) | 7.86e+14 (20) | 8.28 (400) |
| GNN, noise at init, linear | 0.324 (60) | 3.32 (50) | 0.136 (50) | 1.05 (100) | 18.1 (100) | 1.43e+17 (20) | 2.09e+20 (20) | 1.52 (400) |

**Pairs within a budget, best cell, all orders**

| Model | T <= 1e+06 | T <= 1e+08 | T <= 1e+10 | T <= 1e+12 |
|---|---|---|---|---|
| coherent light, w = 100 | 242 | 254 | 284 | 324 |
| GNN, noise every step, linear | 248 | 273 | 292 | 295 |
| GNN, noise at init, linear | 247 | 267 | 285 | 288 |
