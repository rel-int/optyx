## Pairs separated by BREC's test, rule permutation, any of four cells

| Source | R = 1e+03 | R = 1e+04 | R = 1e+05 | R = 1e+06 | profiles, exact |
|---|---|---|---|---|---|
| coherent light, w = 1 | 3 | 4 | 14 | 17 | 49 |
| coherent light, w = 100 | 5 | 6 | 20 | 23 | 49 |
| coherent light, intensities | 6 | 9 | 17 | 26 | 49 |
| thermal light, intensities | 1 | 4 | 2 | 3 | 40 |
| pairs run | 49 | 49 | 49 | 49 | 49 |

## Pairs separated by BREC's test, rule hotelling, any of four cells

| Source | R = 1e+03 | R = 1e+04 | R = 1e+05 | R = 1e+06 | profiles, exact |
|---|---|---|---|---|---|
| coherent light, w = 1 | 9 | 4 | 15 | 23 | 49 |
| coherent light, w = 100 | 9 | 14 | 24 | 28 | 49 |
| coherent light, intensities | 6 | 17 | 22 | 27 | 49 |
| thermal light, intensities | 5 | 1 | 2 | 3 | 40 |
| pairs run | 49 | 49 | 49 | 49 | 49 |

## Pairs separated by BREC's test, rule scaled, any of four cells

| Source | R = 1e+03 | R = 1e+04 | R = 1e+05 | R = 1e+06 | profiles, exact |
|---|---|---|---|---|---|
| coherent light, w = 1 | 0 | 0 | 0 | 3 | 49 |
| coherent light, w = 100 | 0 | 1 | 3 | 15 | 49 |
| coherent light, intensities | 0 | 1 | 4 | 14 | 49 |
| thermal light, intensities | 0 | 0 | 0 | 0 | 40 |
| pairs run | 49 | 49 | 49 | 49 | 49 |

## Pairs separated by BREC's test, rule verbatim, any of four cells

| Source | R = 1e+03 | R = 1e+04 | R = 1e+05 | R = 1e+06 | profiles, exact |
|---|---|---|---|---|---|
| coherent light, w = 1 | 0 | 0 | 0 | 1 | 49 |
| coherent light, w = 100 | 0 | 1 | 3 | 8 | 49 |
| coherent light, intensities | 0 | 0 | 3 | 9 | 49 |
| thermal light, intensities | 0 | 0 | 0 | 0 | 40 |
| pairs run | 49 | 49 | 49 | 49 | 49 |

## Pairs separated per class at R = 1e+06, permutation rule, any cell

| Source | Basic | Regular | Str. regular | Extension | CFI | 4-vertex | Dist. regular | Total |
|---|---|---|---|---|---|---|---|---|
| coherent light, w = 1 | 7 | 5 | 0 | 5 | 0 | 0 | 0 | 17 |
| coherent light, w = 100 | 10 | 6 | 0 | 7 | 0 | 0 | 0 | 23 |
| coherent light, intensities | 10 | 6 | 0 | 8 | 1 | 1 | 0 | 26 |
| thermal light, intensities | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 3 |
| profiles, coherent, exact | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |
| profiles, thermal, exact | 10 | 6 | 5 | 10 | 9 | 0 | 0 | 40 |
| pairs | 10 | 6 | 6 | 10 | 9 | 2 | 6 | 49 |

## Agreement with the exact profiles at R = 1e+06, permutation rule

| Source | both | test only | profiles only | neither |
|---|---|---|---|---|
| coherent light, w = 1 | 17 | 0 | 32 | 0 |
| coherent light, w = 100 | 23 | 0 | 26 | 0 |
| coherent light, intensities | 26 | 0 | 23 | 0 |
| thermal light, intensities | 3 | 0 | 37 | 9 |

## Reliability failures (pair-cell cases of 196) and median T² at the largest budget

| Source | unreliable, hotelling | unreliable, permutation | unreliable, verbatim | median T² pair (BREC's, no factor q) | median T² reliability |
|---|---|---|---|---|---|
| coherent light, w = 1 | 2 | 2 | 0 | 1.06 | 0.804 |
| coherent light, w = 100 | 6 | 4 | 0 | 1.46 | 0.793 |
| coherent light, intensities | 2 | 3 | 0 | 1.23 | 0.78 |
| thermal light, intensities | 5 | 1 | 0 | 0.769 | 0.816 |

## Which feature carries the gap: the feature with the largest gap over noise, counted over the pairs separated at the largest budget

| Source | e1(I^1) | e2(I^1) | e3(I^1) | e4(I^1) | e1(I^2) | e2(I^2) | e3(I^2) | e4(I^2) | e1(I^3) | e2(I^3) | e3(I^3) | e4(I^3) | e1(I^4) | e2(I^4) | e3(I^4) | e4(I^4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coherent light, w = 1 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| coherent light, w = 100 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| coherent light, intensities | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 2 | 1 | 0 | 1 | 0 | 9 | 0 | 0 | 0 |
| thermal light, intensities | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

## Exact anchors: relative gap of the exact readout between G and H (cell 100), against the profile's exact order

| Pair | n | coherent e3(I^1) | coherent e3(I^2) | thermal e3(I^1) | thermal e3(I^2) | thermal e3(I^3) | thermal e4(I^1) | thermal e4(I^2) | thermal e4(I^3) | profile order coherent / thermal |
|---|---|---|---|---|---|---|---|---|---|---|
| Basic_0 | 10 | 1.9e-03 | 3.0e-03 | -6.8e-06 | -2.9e-05 | -6.6e-05 | -1.4e-05 | -5.7e-05 | -1.3e-04 | 1 / 1 |
| Basic_6 | 10 | -2.8e-03 | -6.3e-03 | 1.8e-06 | 6.9e-05 | 2.0e-04 | 3.9e-06 | 9.7e-05 | 2.8e-04 | 1 / 1 |
| Basic_12 | 10 | 9.8e-04 | 2.1e-03 | 4.0e-06 | 1.6e-05 | 3.5e-05 | 8.0e-06 | 3.1e-05 | 6.9e-05 | 1 / 1 |
| Basic_18 | 10 | -7.1e-04 | -1.4e-03 | -1.9e-06 | -8.3e-06 | -1.9e-05 | -3.7e-06 | -1.6e-05 | -3.5e-05 | 1 / 1 |
| Basic_24 | 10 | -5.9e-03 | -1.6e-02 | -1.5e-05 | 1.1e-04 | 3.7e-04 | -2.9e-05 | 1.0e-04 | 4.0e-04 | 1 / 1 |
| Basic_30 | 10 | 9.7e-04 | 1.8e-03 | 1.1e-06 | 4.9e-06 | 1.1e-05 | 2.1e-06 | 8.9e-06 | 2.0e-05 | 1 / 1 |
| Basic_36 | 10 | 6.2e-04 | 5.8e-04 | -9.8e-06 | -3.9e-05 | -8.8e-05 | -2.0e-05 | -7.9e-05 | -1.8e-04 | 1 / 1 |
| Basic_42 | 10 | -2.2e-03 | -4.6e-03 | 8.9e-06 | 3.3e-05 | 7.2e-05 | 1.8e-05 | 6.8e-05 | 1.5e-04 | 1 / 1 |
| Basic_48 | 10 | 5.0e-03 | 1.2e-02 | -2.5e-06 | -5.8e-06 | -1.0e-05 | -5.2e-06 | -1.6e-05 | -3.4e-05 | 1 / 1 |
| Basic_54 | 10 | 1.3e-03 | 2.7e-03 | -1.7e-05 | -6.6e-05 | -1.4e-04 | -3.5e-05 | -1.3e-04 | -3.0e-04 | 1 / 1 |
| Regular_0 | 10 | -5.4e-04 | -1.0e-03 | -1.2e-05 | -4.6e-05 | -1.1e-04 | -2.3e-05 | -9.3e-05 | -2.1e-04 | 1 / 1 |
| Regular_10 | 9 | -1.4e-03 | -3.4e-03 | -3.1e-06 | -1.1e-05 | -2.4e-05 | -6.1e-06 | -2.3e-05 | -4.9e-05 | 1 / 1 |
| Regular_20 | 10 | -2.5e-03 | -5.0e-03 | -8.7e-06 | -3.5e-05 | -7.8e-05 | -1.7e-05 | -6.9e-05 | -1.6e-04 | 1 / 1 |
| Regular_30 | 10 | 4.9e-03 | 1.2e-02 | -5.7e-06 | -2.4e-05 | -5.7e-05 | -1.1e-05 | -4.9e-05 | -1.1e-04 | 1 / 1 |
| Regular_40 | 9 | 4.0e-03 | 8.3e-03 | -3.0e-05 | -1.2e-04 | -2.8e-04 | -5.9e-05 | -2.4e-04 | -5.6e-04 | 1 / 1 |
| Regular_47 | 7 | -9.5e-03 | -2.2e-02 | -3.5e-06 | -1.2e-05 | -2.3e-05 | -6.6e-06 | -2.0e-05 | -3.5e-05 | 1 / 1 |
| Strongly_regular_0 | 16 | -2.9e-15 | 2.0e-08 | 0.0e+00 | 0.0e+00 | 4.5e-15 | 2.2e-16 | -4.4e-16 | -1.5e-14 | 3 / 4 |
| Strongly_regular_5 | 25 | - | - | 0.0e+00 | -2.2e-16 | 2.5e-15 | 4.4e-16 | -4.4e-16 | 2.3e-14 | 3 / 4 |
| Strongly_regular_10 | 26 | - | - | 2.2e-16 | 0.0e+00 | 1.8e-15 | 0.0e+00 | 4.4e-16 | -1.5e-14 | 3 / 4 |
| Strongly_regular_20 | 29 | - | - | 0.0e+00 | 0.0e+00 | -2.6e-15 | 0.0e+00 | 0.0e+00 | -1.6e-14 | 3 / None |
| Strongly_regular_30 | 29 | - | - | -8.9e-16 | -1.8e-15 | -4.9e-15 | -1.3e-15 | -2.4e-15 | 3.4e-14 | 3 / 4 |
| Strongly_regular_40 | 35 | - | - | 0.0e+00 | 4.4e-16 | -1.1e-15 | -2.2e-16 | 4.4e-16 | 1.1e-14 | 3 / 4 |
| Extension_0 | 10 | -5.0e-04 | -9.1e-04 | -1.5e-06 | -5.9e-06 | -1.3e-05 | -2.9e-06 | -1.2e-05 | -2.6e-05 | 1 / 1 |
| Extension_10 | 10 | 9.9e-04 | 1.9e-03 | 3.1e-06 | 1.2e-05 | 2.6e-05 | 6.1e-06 | 2.3e-05 | 5.1e-05 | 1 / 1 |
| Extension_20 | 10 | 8.0e-04 | 1.4e-03 | 8.3e-07 | 3.5e-06 | 8.0e-06 | 1.6e-06 | 6.5e-06 | 1.4e-05 | 1 / 1 |
| Extension_30 | 10 | -1.1e-03 | -2.1e-03 | -3.2e-06 | -1.3e-05 | -2.8e-05 | -6.3e-06 | -2.5e-05 | -5.5e-05 | 1 / 1 |
| Extension_40 | 10 | 1.7e-03 | 3.4e-03 | 1.9e-06 | 8.2e-06 | 1.9e-05 | 3.7e-06 | 1.5e-05 | 3.3e-05 | 1 / 1 |
| Extension_50 | 10 | -5.6e-04 | -1.0e-03 | -1.2e-06 | -4.7e-06 | -1.1e-05 | -2.4e-06 | -9.2e-06 | -2.0e-05 | 1 / 1 |
| Extension_60 | 10 | 1.1e-02 | 4.9e-02 | 2.9e-05 | 1.1e-04 | 2.5e-04 | 5.9e-05 | 2.3e-04 | 5.2e-04 | 1 / 1 |
| Extension_70 | 10 | -3.0e-03 | -6.7e-03 | -8.2e-06 | -3.1e-05 | -6.8e-05 | -1.6e-05 | -6.3e-05 | -1.4e-04 | 1 / 1 |
| Extension_80 | 10 | 4.6e-05 | 4.8e-04 | 2.6e-06 | 1.1e-05 | 2.4e-05 | 5.2e-06 | 2.1e-05 | 4.9e-05 | 1 / 1 |
| Extension_90 | 11 | 3.4e-05 | 4.0e-04 | 6.3e-06 | 2.3e-05 | 4.9e-05 | 1.3e-05 | 4.7e-05 | 1.0e-04 | 1 / 1 |
| CFI_5 | 18 | -1.7e-05 | 6.8e-05 | -2.6e-05 | -1.0e-04 | -2.4e-04 | -5.2e-05 | -2.1e-04 | -4.9e-04 | 1 / 1 |
| CFI_11 | 44 | - | - | 0.0e+00 | 6.7e-16 | 1.0e-14 | -2.2e-16 | 1.3e-15 | -9.8e-15 | 2 / 2 |
| CFI_12 | 38 | - | - | 0.0e+00 | 2.4e-15 | 2.3e-14 | 0.0e+00 | 3.6e-15 | 4.2e-14 | 2 / 2 |
| CFI_17 | 48 | - | - | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 2.2e-16 | 0.0e+00 | 2 / 3 |
| CFI_18 | 30 | - | - | -2.0e-06 | -7.9e-06 | -1.8e-05 | -4.0e-06 | -1.6e-05 | -3.7e-05 | 1 / 1 |
| CFI_31 | 38 | - | - | 0.0e+00 | 4.8e-14 | 4.6e-13 | 0.0e+00 | 7.2e-14 | 7.8e-13 | 2 / 2 |
| CFI_57 | 40 | - | - | -9.1e-13 | 1.4e-11 | 4.2e-11 | -1.8e-12 | 1.5e-11 | 4.8e-11 | 1 / 1 |
| CFI_73 | 46 | - | - | 0.0e+00 | -2.2e-16 | -2.6e-16 | 0.0e+00 | -2.2e-16 | 1.9e-15 | 3 / 4 |
| CFI_76 | 52 | - | - | 0.0e+00 | 4.4e-16 | -6.6e-16 | 0.0e+00 | 0.0e+00 | 1.4e-15 | 3 / 4 |
| 4-vertex_condition_0 | 63 | - | - | 0.0e+00 | 0.0e+00 | -2.2e-15 | 0.0e+00 | -2.2e-16 | 1.6e-14 | 3 / None |
| 4-vertex_condition_10 | 63 | - | - | 0.0e+00 | -2.2e-16 | 5.3e-16 | -2.2e-16 | -2.2e-16 | 2.8e-15 | 3 / None |
| Distance_regular_1 | 30 | - | - | 0.0e+00 | 0.0e+00 | 1.3e-15 | 0.0e+00 | 0.0e+00 | 5.3e-16 | 3 / None |
| Distance_regular_2 | 30 | - | - | 0.0e+00 | 0.0e+00 | 2.6e-16 | 0.0e+00 | 0.0e+00 | 4.6e-15 | 3 / None |
| Distance_regular_4 | 38 | - | - | 0.0e+00 | 2.2e-16 | -2.6e-16 | 0.0e+00 | 2.2e-16 | -1.1e-15 | 3 / None |
| Distance_regular_5 | 38 | - | - | 0.0e+00 | 0.0e+00 | -5.3e-16 | -2.2e-16 | -2.2e-16 | -1.1e-14 | 3 / None |
| Distance_regular_18 | 32 | - | - | 2.2e-16 | 0.0e+00 | 1.3e-15 | 0.0e+00 | 0.0e+00 | -9.3e-15 | 4 / None |
| Distance_regular_19 | 32 | - | - | -2.2e-16 | 0.0e+00 | 1.7e-15 | 0.0e+00 | 0.0e+00 | -1.9e-15 | 4 / None |

## Per pair: smallest budget that separates, permutation rule, any cell

| Pair | n | slots | coherent light, w = 1 | coherent light, w = 100 | coherent light, intensities | thermal light, intensities | profile order coherent / thermal |
|---|---|---|---|---|---|---|---|
| Basic_0 | 10 | 920 | 1e+06 | 1e+05 | 1e+05 | - | 1 / 1 |
| Basic_6 | 10 | 870 | 1e+03 | 1e+05 | 1e+05 | - | 1 / 1 |
| Basic_12 | 10 | 930 | - | 1e+05 | 1e+06 | - | 1 / 1 |
| Basic_18 | 10 | 920 | - | 1e+06 | 1e+06 | - | 1 / 1 |
| Basic_24 | 10 | 840 | 1e+05 | 1e+04 | 1e+03 | - | 1 / 1 |
| Basic_30 | 10 | 930 | - | 1e+03 | 1e+03 | 1e+04 | 1 / 1 |
| Basic_36 | 10 | 930 | 1e+06 | 1e+05 | 1e+05 | - | 1 / 1 |
| Basic_42 | 10 | 920 | 1e+06 | 1e+05 | 1e+03 | 1e+04 | 1 / 1 |
| Basic_48 | 10 | 920 | 1e+05 | 1e+05 | 1e+05 | - | 1 / 1 |
| Basic_54 | 10 | 910 | 1e+05 | 1e+05 | 1e+05 | - | 1 / 1 |
| Regular_0 | 10 | 910 | 1e+06 | 1e+04 | 1e+05 | - | 1 / 1 |
| Regular_10 | 9 | 810 | 1e+04 | 1e+06 | 1e+06 | - | 1 / 1 |
| Regular_20 | 10 | 920 | 1e+05 | 1e+05 | 1e+05 | - | 1 / 1 |
| Regular_30 | 10 | 910 | 1e+05 | 1e+03 | 1e+03 | 1e+04 | 1 / 1 |
| Regular_40 | 9 | 828 | 1e+04 | 1e+04 | 1e+04 | - | 1 / 1 |
| Regular_47 | 7 | 623 | 1e+05 | 1e+04 | 1e+04 | - | 1 / 1 |
| Strongly_regular_0 | 16 | 1504 | 1e+05 | - | - | - | 3 / 4 |
| Strongly_regular_5 | 25 | 2375 | - | 1e+04 | - | 1e+04 | 3 / 4 |
| Strongly_regular_10 | 26 | 2470 | - | - | - | - | 3 / 4 |
| Strongly_regular_20 | 29 | 2755 | - | - | - | - | 3 / None |
| Strongly_regular_30 | 29 | 2755 | - | 1e+05 | - | - | 3 / 4 |
| Strongly_regular_40 | 35 | 3325 | - | - | - | 1e+06 | 3 / 4 |
| Extension_0 | 10 | 920 | 1e+05 | - | 1e+06 | - | 1 / 1 |
| Extension_10 | 10 | 930 | - | 1e+06 | 1e+04 | - | 1 / 1 |
| Extension_20 | 10 | 930 | - | 1e+06 | 1e+06 | - | 1 / 1 |
| Extension_30 | 10 | 910 | 1e+06 | 1e+06 | 1e+06 | - | 1 / 1 |
| Extension_40 | 10 | 910 | 1e+06 | 1e+03 | 1e+03 | - | 1 / 1 |
| Extension_50 | 10 | 920 | 1e+05 | - | - | 1e+03 | 1 / 1 |
| Extension_60 | 10 | 870 | 1e+04 | 1e+03 | 1e+03 | - | 1 / 1 |
| Extension_70 | 10 | 900 | 1e+05 | 1e+05 | 1e+05 | - | 1 / 1 |
| Extension_80 | 10 | 910 | - | - | - | 1e+06 | 1 / 1 |
| Extension_90 | 11 | 1012 | 1e+06 | 1e+05 | 1e+06 | - | 1 / 1 |
| CFI_5 | 18 | 1548 | - | - | - | - | 1 / 1 |
| CFI_11 | 44 | 3828 | - | - | - | - | 2 / 2 |
| CFI_12 | 38 | 3306 | - | - | 1e+06 | - | 2 / 2 |
| CFI_17 | 48 | 4272 | - | - | - | - | 2 / 3 |
| CFI_18 | 30 | 2580 | - | - | - | - | 1 / 1 |
| CFI_31 | 38 | 3344 | 1e+05 | 1e+05 | 1e+04 | - | 2 / 2 |
| CFI_57 | 40 | 3520 | - | - | - | - | 1 / 1 |
| CFI_73 | 46 | 4094 | - | - | - | - | 3 / 4 |
| CFI_76 | 52 | 4628 | - | 1e+03 | 1e+04 | - | 3 / 4 |
| 4-vertex_condition_0 | 63 | 6048 | 1e+03 | - | - | - | 3 / None |
| 4-vertex_condition_10 | 63 | 6048 | 1e+03 | - | 1e+06 | - | 3 / None |
| Distance_regular_1 | 30 | 2790 | - | 1e+05 | - | - | 3 / None |
| Distance_regular_2 | 30 | 2790 | - | - | - | - | 3 / None |
| Distance_regular_4 | 38 | 3534 | - | - | - | - | 3 / None |
| Distance_regular_5 | 38 | 3534 | - | 1e+05 | 1e+05 | 1e+05 | 3 / None |
| Distance_regular_18 | 32 | 2976 | - | - | 1e+05 | - | 4 / None |
| Distance_regular_19 | 32 | 2944 | - | - | - | - | 4 / None |

## GPU seconds per pair and cell, all four sources at every budget

median 277, max 1103, total 22.47 GPU-hours over 196 pair-cell jobs
