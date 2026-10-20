## Pairs separated (exact test, relative gap > 1e-12, any of four cells)

| Source | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|
| photons p=1 | 60 | 50 | 50 | 100 | 76 | 20 | 19 | 375 |
| photons p=0 | 60 | 50 | 50 | 100 | 71 | 20 | 19 | 370 |
| coherent | 60 | 50 | 50 | 100 | 73 | 20 | 19 | 372 |
| thermal | 60 | 50 | 33 | 100 | 61 | 0 | 1 | 305 |
| pairs run | 60 | 50 | 50 | 100 | 100 | 20 | 20 | 400 |
| at order 4 | 60 | 50 | 50 | 100 | 72 | 20 | 20 | 372 |

## Pairs separated under BREC's test computed from the profiles

| Source | Test | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|---|
| photons p=1 | eps = 1e-7 | 60 | 50 | 1 | 99 | 1 | 0 | 0 | 211 |
| photons p=1 | eps at the pilot's rounding noise | 60 | 50 | 50 | 100 | 78 | 20 | 19 | 377 |
| photons p=0 | eps = 1e-7 | 60 | 50 | 1 | 97 | 1 | 0 | 0 | 209 |
| photons p=0 | eps at the pilot's rounding noise | 60 | 50 | 50 | 100 | 73 | 20 | 19 | 372 |
| coherent | eps = 1e-7 | 60 | 50 | 1 | 97 | 1 | 0 | 0 | 209 |
| coherent | eps at the pilot's rounding noise | 60 | 50 | 50 | 100 | 78 | 20 | 19 | 377 |
| thermal | eps = 1e-7 | 34 | 27 | 0 | 23 | 1 | 0 | 0 | 85 |
| thermal | eps at the pilot's rounding noise | 60 | 50 | 33 | 100 | 61 | 0 | 1 | 305 |

## Separation order (pairs per order), indistinguishable photons / coherent

| Class | order 1 | order 2 | order 3 | order 4 | none |
|---|---|---|---|---|---|
| Basic | 60 / 60 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Regular | 50 / 50 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Str. regular | 0 / 0 | 0 / 0 | 50 / 50 | 0 / 0 | 0 / 0 |
| Extension | 100 / 100 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| CFI | 6 / 6 | 46 / 46 | 19 / 17 | 5 / 4 | 24 / 27 |
| 4-vertex | 0 / 0 | 0 / 0 | 20 / 20 | 0 / 0 | 0 / 0 |
| Dist. regular | 0 / 0 | 0 / 0 | 16 / 16 | 3 / 3 | 1 / 1 |

## Coherent light: median samples per network against brightness (best of four cells)

| Class | top order only, Poisson noise, w = 0.05 | all orders, w = 0.05 | all orders, w = 0.2 | all orders, w = 1.0 | all orders, w = 5.0 | all orders, w = 20.0 | all orders, w = 100.0 |
|---|---|---|---|---|---|---|---|
| Basic (60) | 1.3e+09 | 2.3e+07 | 1.5e+06 | 8.2e+04 | 1.7e+04 | 1.1e+04 | 9.4e+03 |
| Regular (50) | 6.6e+08 | 9.6e+06 | 6.3e+05 | 3.4e+04 | 7.1e+03 | 4.5e+03 | 3.9e+03 |
| Str. regular (50) | 3.9e+10 | 5e+10 | 3.9e+08 | 7.1e+06 | 1.3e+06 | 8.1e+05 | 7.1e+05 |
| Extension (100) | 9.9e+08 | 2.9e+07 | 1.5e+06 | 7.6e+04 | 1.5e+04 | 9.6e+03 | 8.4e+03 |
| CFI (73) | 6e+14 | 7.8e+14 | 6.4e+12 | 1.3e+11 | 2.5e+10 | 1.7e+10 | 1.5e+10 |
| 4-vertex (20) | 2.5e+18 | 1.4e+18 | 1.5e+16 | 1.9e+14 | 1.9e+13 | 8.6e+12 | 6.5e+12 |
| Dist. regular (19) | 3.3e+15 | 4e+15 | 2.9e+13 | 4.1e+11 | 5.5e+10 | 3.1e+10 | 2.6e+10 |

## Coherent light: pairs separated within a device-time budget at 1 GHz

| Brightness | an hour | a day | a year |
|---|---|---|---|
| w = 0.05 | 253 | 268 | 316 |
| w = 0.2 | 282 | 304 | 360 |
| w = 1.0 | 310 | 337 | 364 |
| w = 5.0 | 332 | 357 | 365 |
| w = 20.0 | 337 | 359 | 365 |
| w = 100.0 | 337 | 359 | 365 |

## Coherent light: gain in samples from brightness, median per class

| Class | w = 0.2 vs 0.05 | w = 1.0 vs 0.05 | w = 5.0 vs 0.05 | w = 20.0 vs 0.05 | w = 100.0 vs 0.05 |
|---|---|---|---|---|---|
| Basic | 16.2 | 328 | 1.65e+03 | 2.58e+03 | 2.96e+03 |
| Regular | 17 | 328 | 1.57e+03 | 2.47e+03 | 2.83e+03 |
| Str. regular | 128 | 7.01e+03 | 3.94e+04 | 6.14e+04 | 7.01e+04 |
| Extension | 18 | 354 | 1.72e+03 | 2.71e+03 | 3.11e+03 |
| CFI | 96.9 | 5.03e+03 | 2.64e+04 | 4e+04 | 4.51e+04 |
| 4-vertex | 91.4 | 6.81e+03 | 6.9e+04 | 1.54e+05 | 2.04e+05 |
| Dist. regular | 96.7 | 6.47e+03 | 4.9e+04 | 8.95e+04 | 1.08e+05 |

## GPU time per pair and cell (seconds, median over cells)

| size | order | pairs | median | max |
|---|---|---|---|---|
| 0-19 | 4 | 212 | 0 | 2 |
| 20-39 | 4 | 59 | 3 | 9 |
| 40-63 | 4 | 66 | 47 | 68 |
| 64-109 | 3,4 | 43 | 172 | 383 |
| 110-199 | 3 | 20 | 15 | 59 |