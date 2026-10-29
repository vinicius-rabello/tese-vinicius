# Métricas espectrais: sem janela × Hann (teste)

27 runs/baselines com as duas avaliações. Faixa principal k = 11–46; k > 46 à parte.

## Métricas pontuais (devem ser idênticas)

| run | máx. dif. relativa por amostra (MSE, PSNR, SSIM, LPIPS) |
|---|---|
| A_dscms_res_lam0_seed0 | 9.9e-07 |
| A_prusr_res_lam0_seed0 | 0.00022 |
| A_sofono_res_lam0_seed0 | 0 |
| A_srcnn_res_lam0_seed0 | 3.3e-07 |
| A_srno_res_lam0_seed0 | 0 |
| B_srcnn_res_lam0.001_seed0 | 0.0002 |
| B_srcnn_res_lam0.01_seed0 | 0.00047 |
| B_srcnn_res_lam0.1_seed0 | 0.00055 |
| B_srcnn_res_lam1_seed0 | 0.00056 |
| B_srno_res_lam0.001_seed0 | 0 |
| B_srno_res_lam0.01_seed0 | 0 |
| B_srno_res_lam0.1_seed0 | 0 |
| B_srno_res_lam1_seed0 | 0 |
| baseline_identity | 0 |
| baseline_identity_noise0.0001 | 0 |
| baseline_identity_noise0.001 | 0 |
| baseline_identity_noise0.003 | 0 |
| baseline_identity_noise0.01 | 0 |
| baseline_identity_noise0.03 | 0 |
| baseline_identity_noise0.1 | 0 |
| baseline_wiener | 0 |
| baseline_wiener_noise0.0001 | 0 |
| baseline_wiener_noise0.001 | 0 |
| baseline_wiener_noise0.003 | 0 |
| baseline_wiener_noise0.01 | 0 |
| baseline_wiener_noise0.03 | 0 |
| baseline_wiener_noise0.1 | 0 |

## Antigo → Hann

| run | k_c | k_c_01 | eps_w_10_40 | eps_hf | T_11_46_geo | T_gt46_geo | slope_11_46 | ke |
|---|---|---|---|---|---|---|---|---|
| A_dscms_res_lam0_seed0 | 63 → 26 | 22 → 22 | 0.0855 → 0.123 | 0.26 → 0.692 | 0.679 → 0.00548 | 0.424 → 1.76e-06 | -2.69 → -13.5 | 0.355 → 24.5 |
| A_prusr_res_lam0_seed0 | 63 → 27 | 22 → 22 | 0.0716 → 0.103 | 0.215 → 0.656 | 0.735 → 0.101 | 0.708 → 0.000325 | -2.55 → -7.55 | 0.137 → 15.4 |
| A_sofono_res_lam0_seed0 | 63 → 30 | 25 → 25 | 0.0483 → 0.0688 | 0.186 → 0.593 | 0.76 → 0.172 | 0.758 → 0.00082 | -2.57 → -6.61 | 0.11 → 12.5 |
| A_srcnn_res_lam0_seed0 | 63 → 25 | 20 → 20 | 0.0971 → 0.142 | 0.248 → 0.705 | 0.631 → 0.0116 | 0.582 → 1.32e-05 | -2.58 → -11.6 | 0.263 → 23 |
| A_srno_res_lam0_seed0 | 63 → 29 | 24 → 24 | 0.0602 → 0.0831 | 0.204 → 0.621 | 0.73 → 0.145 | 0.71 → 0.000124 | -2.58 → -6.84 | 0.139 → 14 |
| B_srcnn_res_lam0.001_seed0 | 63 → 25 | 21 → 21 | 0.0927 → 0.133 | 0.244 → 0.694 | 0.649 → 0.0184 | 0.585 → 2.07e-05 | -2.6 → -10.9 | 0.246 → 21.8 |
| B_srcnn_res_lam0.01_seed0 | 63 → 27 | 22 → 22 | 0.0846 → 0.118 | 0.236 → 0.676 | 0.748 → 0.34 | 0.617 → 0.002 | -2.48 → -4.92 | 0.148 → 8.95 |
| B_srcnn_res_lam0.1_seed0 | 63 → 30 | 25 → 25 | 0.0646 → 0.0876 | 0.225 → 0.676 | 0.864 → 0.691 | 0.738 → 0.032 | -2.32 → -3.46 | 0.0577 → 3.51 |
| B_srcnn_res_lam1_seed0 | 63 → 30 | 25 → 26 | 0.0627 → 0.0821 | 0.235 → 0.713 | 0.892 → 0.747 | 0.821 → 0.157 | -2.27 → -3.26 | 0.0339 → 1.2 |
| B_srno_res_lam0.001_seed0 | 63 → 30 | 24 → 25 | 0.055 → 0.0726 | 0.197 → 0.598 | 0.765 → 0.257 | 0.727 → 0.000442 | -2.56 → -5.69 | 0.112 → 11.2 |
| B_srno_res_lam0.01_seed0 | 63 → 34 | 28 → 28 | 0.0355 → 0.0437 | 0.175 → 0.563 | 0.874 → 0.66 | 0.755 → 0.0309 | -2.37 → -3.6 | 0.0535 → 3.75 |
| B_srno_res_lam0.1_seed0 | 63 → 43 | 37 → 36 | 0.00863 → 0.00626 | 0.114 → 0.399 | 0.98 → 0.906 | 0.861 → 0.373 | -2.2 → -3.06 | 0.0148 → 0.466 |
| B_srno_res_lam1_seed0 | 63 → 51 | 48 → 47 | 0.00183 → 0.00045 | 0.0621 → 0.231 | 1.02 → 1.09 | 0.888 → 0.448 | -2.06 → -2.42 | 0.00647 → 0.231 |
| baseline_identity | 15 → 15 | 8 → 8 | 0.532 → 0.504 | 0.874 → 0.875 | 9.17e-05 → 0.000194 | 5.9e-12 → 1.32e-07 | -16 → -14.3 | 42.7 → 31.5 |
| baseline_identity_noise0.0001 | 15 → 15 | 8 → 8 | 0.532 → 0.504 | 0.874 → 0.875 | 0.000389 → 0.000549 | 7.99e-06 → 4.12e-05 | -11.9 → -11.5 | 42.5 → 31.4 |
| baseline_identity_noise0.001 | 15 → 15 | 8 → 8 | 0.532 → 0.504 | 0.875 → 0.877 | 0.00228 → 0.00372 | 0.000799 → 0.00411 | -7.7 → -7.14 | 34.3 → 24.6 |
| baseline_identity_noise0.003 | 15 → 15 | 8 → 8 | 0.532 → 0.505 | 0.878 → 0.891 | 0.00712 → 0.0124 | 0.00719 → 0.037 | -5.48 → -4.9 | 21.5 → 14.2 |
| baseline_identity_noise0.01 | 15 → 15 | 8 → 8 | 0.536 → 0.513 | 0.912 → 1.05 | 0.0314 → 0.0593 | 0.0799 → 0.411 | -3.09 → -2.56 | 8.93 → 5.06 |
| baseline_identity_noise0.03 | 15 → 14 | 8 → 8 | 0.574 → 0.581 | 1.21 → 2.42 | 0.155 → 0.312 | 0.719 → 3.7 | -1.17 → -0.766 | 2.33 → 2 |
| baseline_identity_noise0.1 | 13 → 12 | 7 → 7 | 1 → 1.36 | 4.63 → 18.1 | 1.16 → 2.51 | 7.99 → 41.1 | 0.294 → 0.495 | 1.91 → 5.83 |
| baseline_wiener | 51 → 49 | 49 → 47 | 7.48e-08 → 1.73e-07 | 0.218 → 0.356 | 1 → 1 | 0.0249 → 0.105 | -2.11 → -2.64 | 5.32 → 2.57 |
| baseline_wiener_noise0.0001 | 35 → 34 | 32 → 31 | 0.033 → 0.0342 | 0.516 → 0.551 | 0.247 → 0.356 | 3.21e-08 → 4.78e-06 | -6.08 → -5.63 | 20.5 → 12.4 |
| baseline_wiener_noise0.001 | 29 → 28 | 25 → 24 | 0.107 → 0.092 | 0.637 → 0.654 | 0.0329 → 0.056 | 6.03e-10 → 7.84e-07 | -10.4 → -9.78 | 27.3 → 17.5 |
| baseline_wiener_noise0.003 | 26 → 24 | 22 → 21 | 0.155 → 0.15 | 0.691 → 0.727 | 0.0291 → 0.0533 | 5.57e-10 → 6.48e-07 | -10.5 → -9.85 | 27.6 → 17.6 |
| baseline_wiener_noise0.01 | 22 → 21 | 17 → 17 | 0.249 → 0.238 | 0.767 → 0.789 | 0.00666 → 0.0134 | 1.18e-10 → 3.29e-07 | -12.5 → -11.8 | 31.7 → 20.8 |
| baseline_wiener_noise0.03 | 18 → 17 | 13 → 13 | 0.38 → 0.409 | 0.84 → 0.881 | 0.00579 → 0.0125 | 1.09e-10 → 2.46e-07 | -12.4 → -11.8 | 32 → 21 |
| baseline_wiener_noise0.1 | 13 → 12 | 9 → 8 | 0.674 → 0.839 | 0.955 → 1.04 | 0.00645 → 0.0148 | 1.11e-10 → 1.73e-07 | -12.3 → -12 | 31.7 → 20.8 |

## Ranking entre runs treinadas (baselines fora)

| métrica | ordem sem janela | ordem com Hann | muda? | Spearman |
|---|---|---|---|---|
| k_c | A_dscms_lam0, A_prusr_lam0, A_sofono_lam0, A_srcnn_lam0, A_srno_lam0, B_srcnn_lam0.001, B_srcnn_lam0.01, B_srcnn_lam0.1, B_srcnn_lam1, B_srno_lam0.001, B_srno_lam0.01, B_srno_lam0.1, B_srno_lam1 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srcnn_lam0.1, B_srcnn_lam1, B_srno_lam0.001, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.01, A_dscms_lam0, A_srcnn_lam0, B_srcnn_lam0.001 | sim | -0.69 |
| k_c_01 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srcnn_lam0.1, B_srcnn_lam1, A_srno_lam0, B_srno_lam0.001, A_dscms_lam0, A_prusr_lam0, B_srcnn_lam0.01, B_srcnn_lam0.001, A_srcnn_lam0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, B_srcnn_lam1, A_sofono_lam0, B_srcnn_lam0.1, B_srno_lam0.001, A_srno_lam0, A_dscms_lam0, A_prusr_lam0, B_srcnn_lam0.01, B_srcnn_lam0.001, A_srcnn_lam0 | sim | 0.98 |
| k_c_fase | A_prusr_lam0, A_sofono_lam0, A_srno_lam0, B_srno_lam0.001, B_srno_lam0.01, B_srno_lam0.1, B_srno_lam1, B_srcnn_lam0.001, B_srcnn_lam1, B_srcnn_lam0.01, A_srcnn_lam0, B_srcnn_lam0.1, A_dscms_lam0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, A_srno_lam0, B_srno_lam0.001, A_srcnn_lam0, A_dscms_lam0, B_srcnn_lam0.001, B_srcnn_lam1, B_srcnn_lam0.1, A_prusr_lam0, B_srcnn_lam0.01 | sim | 0.34 |
| eps_w_10_40 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, A_srno_lam0, B_srcnn_lam1, B_srcnn_lam0.1, A_prusr_lam0, B_srcnn_lam0.01, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, B_srcnn_lam1, A_srno_lam0, B_srcnn_lam0.1, A_prusr_lam0, B_srcnn_lam0.01, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0 | sim | 0.99 |
| eps_hf | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.1, B_srcnn_lam1, B_srcnn_lam0.01, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.1, B_srcnn_lam0.01, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0, B_srcnn_lam1 | sim | 0.93 |
| eps_k_ge_D0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, A_srno_lam0, B_srcnn_lam0.1, B_srcnn_lam1, A_prusr_lam0, B_srcnn_lam0.01, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0 | B_srno_lam1, B_srno_lam0.1, B_srno_lam0.01, A_sofono_lam0, B_srno_lam0.001, A_srno_lam0, B_srcnn_lam1, B_srcnn_lam0.1, A_prusr_lam0, B_srcnn_lam0.01, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0 | sim | 0.99 |
| T_11_46_geo | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srno_lam0.01, B_srcnn_lam0.1, B_srno_lam0.001, A_sofono_lam0, B_srcnn_lam0.01, A_prusr_lam0, A_srno_lam0, A_dscms_lam0, B_srcnn_lam0.001, A_srcnn_lam0 | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srcnn_lam0.1, B_srno_lam0.01, B_srcnn_lam0.01, B_srno_lam0.001, A_sofono_lam0, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | sim | 0.96 |
| ke | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srno_lam0.01, B_srcnn_lam0.1, A_sofono_lam0, B_srno_lam0.001, A_prusr_lam0, A_srno_lam0, B_srcnn_lam0.01, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srcnn_lam0.1, B_srno_lam0.01, B_srcnn_lam0.01, B_srno_lam0.001, A_sofono_lam0, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | sim | 0.93 |
| wke | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srno_lam0.01, B_srcnn_lam0.1, A_sofono_lam0, B_srno_lam0.001, A_prusr_lam0, A_srno_lam0, B_srcnn_lam0.01, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | B_srno_lam1, B_srno_lam0.1, B_srcnn_lam1, B_srcnn_lam0.1, B_srno_lam0.01, B_srcnn_lam0.01, B_srno_lam0.001, A_sofono_lam0, A_srno_lam0, A_prusr_lam0, B_srcnn_lam0.001, A_srcnn_lam0, A_dscms_lam0 | sim | 0.93 |
