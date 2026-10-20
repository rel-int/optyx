## Column: pairs separated

| Source | Test | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|---|
| photons p=1 | BREC, eps = 1e-7 | 10 | 6 | 1 | 10 | 1 | 0 | 0 | 28 |
| photons p=1 | BREC, eps at the noise | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| photons p=1 | exact, gap > 1e-12 | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| photons p=0 | BREC, eps = 1e-7 | 10 | 6 | 1 | 10 | 1 | 0 | 0 | 28 |
| photons p=0 | BREC, eps at the noise | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| photons p=0 | exact, gap > 1e-12 | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| coherent | BREC, eps = 1e-7 | 10 | 6 | 1 | 10 | 1 | 0 | 0 | 28 |
| coherent | BREC, eps at the noise | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| coherent | exact, gap > 1e-12 | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| thermal | BREC, eps = 1e-7 | 5 | 4 | 0 | 1 | 1 | 0 | 0 | 11 |
| thermal | BREC, eps at the noise | 10 | 6 | 5 | 10 | 9 | 0 | 0 | 40 |
| thermal | exact, gap > 1e-12 | 10 | 6 | 5 | 10 | 9 | 0 | 0 | 40 |
| pairs run | | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |

Unreliable (seed, order) cases under the noise-scaled test: {'photons p=1': 8, 'photons p=0': 8, 'coherent': 7, 'thermal': 8}

## Column: per pair

| Pair | n | seeds | photons p=1: order (BREC / exact), max rel. gap | photons p=0: order (BREC / exact), max rel. gap | coherent: order (BREC / exact), max rel. gap | thermal: order (BREC / exact), max rel. gap | relabel floor |
|---|---|---|---|---|---|---|---|
| Basic_0 | 10 | 4 | 2 / 1, 2.8e-02 | 2 / 1, 1.7e-02 | 2 / 1, 1.6e-02 | None / 1, 2.9e-03 | 5e-15 |
| Basic_6 | 10 | 4 | 1 / 1, 5.4e-02 | 1 / 1, 3.3e-02 | 1 / 1, 3.1e-02 | 1 / 1, 3.9e-03 | 5e-15 |
| Basic_12 | 10 | 4 | 2 / 1, 3.8e-02 | 2 / 1, 2.3e-02 | 2 / 1, 2.2e-02 | None / 1, 7.3e-04 | 5e-15 |
| Basic_18 | 10 | 4 | 2 / 1, 3.0e-02 | 2 / 1, 2.0e-02 | 2 / 1, 1.9e-02 | None / 1, 7.3e-04 | 6e-15 |
| Basic_24 | 10 | 4 | 1 / 1, 1.1e-01 | 1 / 1, 6.9e-02 | 1 / 1, 6.2e-02 | 1 / 1, 4.6e-03 | 3e-15 |
| Basic_30 | 10 | 4 | 2 / 1, 2.4e-02 | 2 / 1, 1.3e-02 | 2 / 1, 1.3e-02 | None / 1, 7.3e-04 | 5e-15 |
| Basic_36 | 10 | 4 | 2 / 1, 1.9e-02 | 2 / 1, 1.1e-02 | 2 / 1, 1.0e-02 | None / 1, 2.1e-03 | 6e-15 |
| Basic_42 | 10 | 4 | 2 / 1, 5.9e-02 | 2 / 1, 3.7e-02 | 2 / 1, 3.6e-02 | 4 / 1, 3.8e-03 | 5e-15 |
| Basic_48 | 10 | 4 | 1 / 1, 1.2e-01 | 1 / 1, 6.8e-02 | 1 / 1, 6.5e-02 | 1 / 1, 5.7e-03 | 5e-15 |
| Basic_54 | 10 | 4 | 1 / 1, 2.2e-02 | 1 / 1, 1.5e-02 | 1 / 1, 1.4e-02 | 1 / 1, 4.0e-03 | 5e-15 |
| Regular_0 | 10 | 4 | 2 / 1, 4.3e-02 | 2 / 1, 2.4e-02 | 2 / 1, 2.4e-02 | 2 / 1, 3.1e-03 | 6e-15 |
| Regular_10 | 9 | 4 | 2 / 1, 6.5e-02 | 2 / 1, 3.3e-02 | 2 / 1, 3.3e-02 | None / 1, 1.7e-03 | 5e-15 |
| Regular_20 | 10 | 4 | 2 / 1, 6.4e-02 | 2 / 1, 3.7e-02 | 2 / 1, 3.4e-02 | None / 1, 1.6e-03 | 5e-15 |
| Regular_30 | 10 | 4 | 1 / 1, 1.4e-01 | 1 / 1, 8.2e-02 | 1 / 1, 7.6e-02 | 1 / 1, 6.1e-03 | 4e-15 |
| Regular_40 | 9 | 4 | 1 / 1, 9.2e-02 | 1 / 1, 5.0e-02 | 1 / 1, 4.9e-02 | 1 / 1, 9.7e-03 | 4e-15 |
| Regular_47 | 7 | 4 | 1 / 1, 2.2e-01 | 1 / 1, 1.3e-01 | 1 / 1, 1.2e-01 | 1 / 1, 5.2e-03 | 4e-15 |
| Strongly_regular_0 | 16 | 4 | 4 / 3, 2.2e-02 | 4 / 3, 1.1e-02 | 4 / 3, 1.1e-02 | None / 4, 1.6e-04 | 4e-15 |
| Strongly_regular_5 | 25 | 4 | None / 3, 5.0e-03 | None / 3, 2.5e-03 | None / 3, 2.5e-03 | None / 4, 4.3e-05 | 6e-15 |
| Strongly_regular_10 | 26 | 4 | None / 3, 8.0e-03 | None / 3, 4.0e-03 | None / 3, 4.0e-03 | None / 4, 5.2e-05 | 5e-15 |
| Strongly_regular_20 | 29 | 4 | None / 3, 2.2e-06 | None / 3, 3.1e-07 | None / 3, 5.9e-07 | None / None, 4.3e-15 | 5e-15 |
| Strongly_regular_30 | 29 | 4 | None / 3, 3.7e-03 | None / 3, 1.9e-03 | None / 3, 1.9e-03 | None / 4, 3.2e-05 | 6e-15 |
| Strongly_regular_40 | 35 | 4 | None / 3, 2.3e-03 | None / 3, 1.2e-03 | None / 3, 1.2e-03 | None / 4, 2.2e-05 | 5e-15 |
| Extension_0 | 10 | 4 | 2 / 1, 2.1e-02 | 2 / 1, 1.2e-02 | 2 / 1, 1.1e-02 | None / 1, 4.6e-04 | 6e-15 |
| Extension_10 | 10 | 4 | 2 / 1, 2.3e-02 | 2 / 1, 1.5e-02 | 2 / 1, 1.5e-02 | None / 1, 6.8e-04 | 6e-15 |
| Extension_20 | 10 | 4 | 2 / 1, 1.8e-02 | 2 / 1, 9.8e-03 | 2 / 1, 9.4e-03 | None / 1, 5.9e-04 | 5e-15 |
| Extension_30 | 10 | 4 | 2 / 1, 3.1e-02 | 2 / 1, 1.6e-02 | 2 / 1, 1.6e-02 | None / 1, 8.7e-04 | 5e-15 |
| Extension_40 | 10 | 4 | 2 / 1, 5.3e-02 | 2 / 1, 2.7e-02 | 2 / 1, 2.7e-02 | None / 1, 1.2e-03 | 5e-15 |
| Extension_50 | 10 | 4 | 2 / 1, 1.7e-02 | 2 / 1, 9.5e-03 | 2 / 1, 9.1e-03 | None / 1, 4.2e-04 | 6e-15 |
| Extension_60 | 10 | 4 | 1 / 1, 1.9e-01 | 1 / 1, 1.0e-01 | 1 / 1, 9.9e-02 | 1 / 1, 2.1e-02 | 5e-15 |
| Extension_70 | 10 | 4 | 2 / 1, 5.9e-02 | 2 / 1, 3.7e-02 | 2 / 1, 3.3e-02 | None / 1, 1.6e-03 | 5e-15 |
| Extension_80 | 10 | 4 | 3 / 1, 4.2e-02 | 3 / 1, 2.8e-02 | 3 / 1, 2.5e-02 | None / 1, 3.8e-04 | 5e-15 |
| Extension_90 | 11 | 4 | 2 / 1, 5.2e-02 | 2 / 1, 2.9e-02 | 2 / 1, 2.7e-02 | None / 1, 8.5e-04 | 5e-15 |
| CFI_5 | 18 | 4 | 1 / 1, 2.4e-02 | 1 / 1, 1.4e-02 | 1 / 1, 1.3e-02 | 1 / 1, 6.0e-03 | 4e-15 |
| CFI_11 | 44 | 4 | None / 2, 4.5e-05 | None / 2, 2.3e-05 | None / 2, 2.2e-05 | None / 2, 1.8e-06 | 5e-15 |
| CFI_12 | 38 | 4 | None / 2, 1.5e-04 | None / 2, 8.5e-05 | None / 2, 8.0e-05 | None / 2, 3.6e-06 | 5e-15 |
| CFI_17 | 48 | 4 | None / 2, 1.1e-05 | None / 2, 5.6e-06 | None / 2, 5.5e-06 | None / 3, 5.5e-07 | 5e-15 |
| CFI_18 | 30 | 4 | None / 1, 5.3e-03 | None / 1, 3.5e-03 | None / 1, 3.4e-03 | None / 1, 1.7e-03 | 5e-15 |
| CFI_31 | 38 | 4 | None / 2, 1.8e-04 | None / 2, 8.7e-05 | None / 2, 1.1e-04 | None / 2, 2.9e-06 | 5e-15 |
| CFI_57 | 40 | 4 | None / 1, 8.6e-05 | None / 1, 4.3e-05 | None / 1, 4.3e-05 | None / 1, 2.1e-05 | 5e-15 |
| CFI_73 | 46 | 4 | None / 3, 4.4e-05 | None / 3, 2.1e-05 | None / 3, 2.2e-05 | None / 4, 1.7e-06 | 5e-15 |
| CFI_76 | 52 | 4 | None / 3, 2.2e-05 | None / 3, 1.8e-05 | None / 3, 1.9e-05 | None / 4, 1.3e-06 | 5e-15 |
| 4-vertex_condition_0 | 63 | 4 | None / 3, 6.7e-08 | None / 3, 1.3e-08 | None / 3, 4.6e-08 | None / None, 1.1e-15 | 7e-15 |
| 4-vertex_condition_10 | 63 | 4 | None / 3, 5.0e-08 | None / 3, 1.0e-08 | None / 3, 3.5e-08 | None / None, 9.4e-16 | 7e-15 |
| Distance_regular_1 | 30 | 4 | None / 3, 8.6e-05 | None / 3, 2.0e-05 | None / 3, 2.2e-05 | None / None, 2.5e-15 | 4e-15 |
| Distance_regular_2 | 30 | 4 | None / 3, 8.6e-05 | None / 3, 2.0e-05 | None / 3, 2.2e-05 | None / None, 2.5e-15 | 5e-15 |
| Distance_regular_4 | 38 | 4 | None / 3, 2.2e-05 | None / 3, 4.3e-06 | None / 3, 4.8e-06 | None / None, 3.8e-15 | 5e-15 |
| Distance_regular_5 | 38 | 4 | None / 3, 2.3e-05 | None / 3, 4.3e-06 | None / 3, 4.9e-06 | None / None, 3.8e-15 | 4e-15 |
| Distance_regular_18 | 32 | 4 | None / 4, 2.8e-04 | None / 4, 4.9e-05 | None / 4, 7.5e-05 | None / None, 5.5e-16 | 3e-15 |
| Distance_regular_19 | 32 | 4 | None / 4, 1.2e-05 | None / 4, 2.2e-06 | None / 4, 2.2e-06 | None / None, 3.2e-15 | 4e-15 |

## Sweep: median ideal samples

| Class | Source | gamma = 0.5 | gamma = 0.7 | gamma = 0.8 | gamma = 0.9 | gamma = 0.95 | gamma = 0.97 | gamma = 0.99 |
|---|---|---|---|---|---|---|---|---|
| Basic | photons p=1 | 8.2e+07 | 5.1e+07 | 3.7e+07 | 4.3e+07 | 3.3e+07 | 2.9e+07 | 2.4e+07 |
| Basic | photons p=0 | 1e+08 | 1.1e+08 | 1e+08 | 1.2e+08 | 1.4e+08 | 1.2e+08 | 9.9e+07 |
| Basic | coherent | 1.1e+08 | 1.1e+08 | 1e+08 | 1.2e+08 | 1.4e+08 | 1.2e+08 | 9.9e+07 |
| Basic | thermal | 1.1e+08 | 3.1e+08 | 4.8e+08 | 9.6e+08 | 3.7e+09 | 1.7e+10 | 5.6e+11 |
| Regular | photons p=1 | 1.5e+07 | 1.2e+07 | 1e+07 | 1.2e+07 | 1e+07 | 9e+06 | 7.9e+06 |
| Regular | photons p=0 | 1.7e+07 | 2.8e+07 | 3.2e+07 | 2.7e+07 | 3.8e+07 | 3.6e+07 | 3.2e+07 |
| Regular | coherent | 1.8e+07 | 2.8e+07 | 3.3e+07 | 2.8e+07 | 3.9e+07 | 3.7e+07 | 3.2e+07 |
| Regular | thermal | 2e+07 | 5.2e+07 | 9.1e+07 | 2.3e+08 | 1.2e+09 | 5.8e+09 | 2.1e+11 |
| Str. regular | photons p=1 | 1.1e+12 | 1e+11 | 4.4e+10 | 2e+10 | 1.3e+10 | 1.1e+10 | 9.2e+09 |
| Str. regular | photons p=0 | 3.2e+12 | 3.8e+11 | 1.7e+11 | 7.7e+10 | 5.1e+10 | 4.3e+10 | 3.6e+10 |
| Str. regular | coherent | 5.9e+12 | 4.9e+11 | 1.7e+11 | 8e+10 | 5.1e+10 | 4.3e+10 | 3.6e+10 |
| Str. regular | thermal | 2e+13 | 2.8e+13 | 6.3e+13 | 3.9e+14 | 6e+15 | 7.9e+16 | 4.5e+19 |
| Extension | photons p=1 | 4.3e+08 | 1.8e+08 | 1.3e+08 | 7.2e+07 | 5.6e+07 | 5e+07 | 4.3e+07 |
| Extension | photons p=0 | 5.4e+08 | 4.1e+08 | 3.3e+08 | 2.5e+08 | 2.2e+08 | 2e+08 | 1.7e+08 |
| Extension | coherent | 5.7e+08 | 4.6e+08 | 3.2e+08 | 2.6e+08 | 2.3e+08 | 2e+08 | 1.8e+08 |
| Extension | thermal | 7.5e+08 | 9.9e+08 | 2.9e+09 | 7.9e+09 | 6.4e+10 | 3.6e+11 | 1.1e+13 |
| CFI | photons p=1 | 2.4e+15 | 2.9e+13 | 5.6e+12 | 1.3e+12 | 9.3e+11 | 6.3e+11 | 5.4e+11 |
| CFI | photons p=0 | 3.9e+15 | 7.9e+13 | 1.6e+13 | 4.4e+12 | 2.9e+12 | 2.5e+12 | 2e+12 |
| CFI | coherent | 6e+15 | 1.7e+14 | 2.1e+13 | 4.7e+12 | 3e+12 | 2.4e+12 | 2e+12 |
| CFI | thermal | 2.1e+15 | 3.2e+14 | 5.6e+14 | 4.4e+15 | 4.5e+16 | 4.1e+17 | 6.9e+19 |
| 4-vertex | photons p=1 | 9.9e+16 | 1.4e+17 | 3.6e+17 | 1.5e+18 | 1.1e+18 | 4.8e+18 | 8e+18 |
| 4-vertex | photons p=0 | 9.4e+20 | 5.7e+20 | 4e+20 | 2.5e+20 | 1.9e+20 | 1.6e+20 | 1.5e+20 |
| 4-vertex | coherent | 3.6e+17 | 5e+17 | 9.2e+17 | 3.1e+19 | 1.2e+19 | 1.3e+19 | 1.6e+20 |
| 4-vertex | thermal | 3.4e+33 | 1.6e+33 | 4.8e+32 | 3.5e+32 | 3.3e+32 | 2.7e+32 | 1.6e+32 |
| Dist. regular | photons p=1 | 1.4e+14 | 1.3e+14 | 2.1e+14 | 9.3e+14 | 1.8e+14 | 7.6e+13 | 5.2e+13 |
| Dist. regular | photons p=0 | 1.7e+16 | 6.5e+15 | 4.4e+15 | 3.1e+15 | 2.8e+15 | 2.7e+15 | 2.6e+15 |
| Dist. regular | coherent | 2.2e+14 | 3.6e+14 | 7.6e+14 | 1.7e+15 | 1.2e+16 | 1.5e+15 | 5.7e+14 |
| Dist. regular | thermal | 2e+34 | 6.7e+33 | 3.3e+33 | 2.5e+33 | 1.6e+33 | 1.8e+33 | 1.4e+33 |

## Sweep: samples relative to indistinguishable photons

| Class | Source | gamma = 0.5 | gamma = 0.7 | gamma = 0.8 | gamma = 0.9 | gamma = 0.95 | gamma = 0.97 | gamma = 0.99 |
|---|---|---|---|---|---|---|---|---|
| Basic | photons p=0 | 1.07 | 1.65 | 2.75 | 3.22 | 3.77 | 3.99 | 4.01 |
| Basic | coherent | 1.07 | 1.98 | 2.79 | 3.30 | 3.83 | 4.02 | 4.09 |
| Basic | thermal | 1.05 | 2.81 | 6.02 | 18.28 | 138.81 | 627.95 | 27427.28 |
| Regular | photons p=0 | 1.15 | 1.61 | 2.98 | 3.81 | 3.90 | 4.00 | 4.02 |
| Regular | coherent | 1.15 | 1.55 | 3.35 | 3.91 | 3.99 | 4.04 | 4.05 |
| Regular | thermal | 1.14 | 1.99 | 8.42 | 45.32 | 131.93 | 504.72 | 19199.48 |
| Str. regular | photons p=0 | 3.95 | 3.95 | 3.81 | 3.88 | 3.92 | 3.89 | 3.88 |
| Str. regular | coherent | 3.32 | 3.96 | 4.00 | 4.00 | 3.94 | 3.96 | 3.96 |
| Str. regular | thermal | 33.57 | 379.67 | 2850.01 | 105273.70 | 3118650.20 | 33103277.88 | 15816482430.53 |
| Extension | photons p=0 | 1.16 | 1.91 | 3.08 | 3.77 | 4.03 | 4.04 | 4.02 |
| Extension | coherent | 1.12 | 2.38 | 3.15 | 3.84 | 4.06 | 4.07 | 4.06 |
| Extension | thermal | 1.18 | 3.52 | 20.42 | 85.95 | 680.25 | 3673.83 | 251777.84 |
| CFI | photons p=0 | 1.19 | 1.85 | 2.60 | 2.87 | 3.31 | 3.54 | 3.92 |
| CFI | coherent | 2.48 | 4.12 | 4.05 | 3.18 | 3.23 | 3.62 | 3.92 |
| CFI | thermal | 1.21 | 5.09 | 30.02 | 705.79 | 7322.04 | 170801.60 | 74060273.31 |
| 4-vertex | photons p=0 | 1078.32 | 501.74 | 344.24 | 124.32 | 34.57 | 22.25 | 23.46 |
| 4-vertex | coherent | 3.57 | 3.25 | 3.05 | 3.36 | 10.15 | 7.06 | 16.26 |
| 4-vertex | thermal | 47772545532518016.00 | 13352311313867804.00 | 1695542925358437.00 | 269272075168986.31 | 308890813518378.12 | 447799565836920.44 | 156450939892058.81 |
| Dist. regular | photons p=0 | 224.30 | 53.41 | 13.78 | 6.45 | 15.51 | 24.91 | 34.35 |
| Dist. regular | coherent | 3.06 | 2.44 | 2.21 | 6.81 | 19.93 | 15.72 | 10.92 |
| Dist. regular | thermal | 144057471025788043264.00 | 43928539885641236480.00 | 8835501643233951744.00 | 2787584292825281536.00 | 8326321344962699264.00 | 13251232244133957632.00 | 22823612043725553664.00 |

## Sweep: separation order (count of pair-seed cases)

| Class | Source | gamma = 0.5 | gamma = 0.7 | gamma = 0.8 | gamma = 0.9 | gamma = 0.95 | gamma = 0.97 | gamma = 0.99 |
|---|---|---|---|---|---|---|---|---|
| Basic | photons p=1 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Basic | photons p=0 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Basic | coherent | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Basic | thermal | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Regular | photons p=1 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 |
| Regular | photons p=0 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 |
| Regular | coherent | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 |
| Regular | thermal | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 | 1: 24 |
| Str. regular | photons p=1 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 |
| Str. regular | photons p=0 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 |
| Str. regular | coherent | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 | 3: 24 |
| Str. regular | thermal | 4: 20, none: 4 | 4: 20, none: 4 | 4: 20, none: 4 | 4: 20, none: 4 | 4: 20, none: 4 | 4: 20, none: 4 | 4: 20, none: 4 |
| Extension | photons p=1 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Extension | photons p=0 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Extension | coherent | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| Extension | thermal | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 | 1: 40 |
| CFI | photons p=1 | 1: 12, 2: 15, 3: 9 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 |
| CFI | photons p=0 | 1: 12, 2: 15, 3: 9 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 |
| CFI | coherent | 1: 12, 2: 15, 3: 9 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 | 1: 12, 2: 16, 3: 8 |
| CFI | thermal | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 | 1: 12, 2: 12, 3: 4, 4: 8 |
| 4-vertex | photons p=1 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 |
| 4-vertex | photons p=0 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 |
| 4-vertex | coherent | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 | 3: 8 |
| 4-vertex | thermal | none: 8 | none: 8 | none: 8 | none: 8 | none: 8 | none: 8 | none: 8 |
| Dist. regular | photons p=1 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 |
| Dist. regular | photons p=0 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 |
| Dist. regular | coherent | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 | 3: 16, 4: 8 |
| Dist. regular | thermal | none: 24 | none: 24 | none: 24 | none: 24 | none: 24 | none: 24 | none: 24 |

## Sweep timings (seconds per pair and seed, both graphs)

| gamma | median | max |
|---|---|---|
| 0.5 | 0 | 96 |
| 0.7 | 0 | 125 |
| 0.8 | 0 | 145 |
| 0.9 | 0 | 192 |
| 0.95 | 0 | 218 |
| 0.97 | 0 | 395 |
| 0.99 | 1 | 312 |
## Order 5 on Distance_regular_3 (38 vertices), gamma = 0.9

| seed | source | order 1: max rel. gap | order 2: max rel. gap | order 3: max rel. gap | order 4: max rel. gap | order 5: max rel. gap | seconds |
|---|---|---|---|---|---|---|---|
| 100 | photons p=1 | 1.8e-16 | 2.9e-16 | 2.3e-16 | 3.8e-16 | 9.3e-07 | 3958 |
| 100 | photons p=0 | 1.8e-16 | 2.9e-16 | 3.5e-16 | 3.7e-16 | 4.5e-16 | 3958 |
| 100 | coherent | 1.8e-16 | 2.9e-16 | 3.5e-16 | 5.6e-16 | 2.3e-07 | 3958 |
| 100 | thermal | 1.8e-16 | 2.9e-16 | 3.4e-16 | 3.7e-16 | 4.4e-16 | 3958 |
| 101 | photons p=1 | 1.9e-16 | 3.2e-16 | 4.1e-16 | 5.8e-16 | 3.6e-07 | 3020 |
| 101 | photons p=0 | 1.9e-16 | 3.2e-16 | 4.1e-16 | 5.8e-16 | 7.9e-16 | 3020 |
| 101 | coherent | 1.9e-16 | 3.2e-16 | 4.1e-16 | 5.8e-16 | 8.8e-08 | 3020 |
| 101 | thermal | 1.9e-16 | 3.2e-16 | 2.7e-16 | 4.6e-16 | 3.9e-16 | 3020 |
| 102 | photons p=1 | 1.9e-16 | 3.1e-16 | 3.9e-16 | 6.5e-16 | 8.4e-08 | 4113 |
| 102 | photons p=0 | 1.9e-16 | 3.1e-16 | 3.9e-16 | 6.5e-16 | 5.4e-16 | 4113 |
| 102 | coherent | 1.9e-16 | 3.1e-16 | 3.9e-16 | 6.5e-16 | 2.1e-08 | 4113 |
| 102 | thermal | 1.9e-16 | 3.1e-16 | 2.6e-16 | 2.1e-16 | 3.6e-16 | 4113 |
| 103 | photons p=1 | 1.8e-16 | 4.4e-16 | 3.5e-16 | 5.7e-16 | 3.1e-08 | 3657 |
| 103 | photons p=0 | 1.8e-16 | 4.4e-16 | 3.5e-16 | 5.7e-16 | 6.2e-16 | 3657 |
| 103 | coherent | 1.8e-16 | 4.4e-16 | 3.5e-16 | 5.7e-16 | 7.6e-09 | 3657 |
| 103 | thermal | 1.8e-16 | 4.4e-16 | 4.7e-16 | 3.8e-16 | 4.6e-16 | 3657 |
