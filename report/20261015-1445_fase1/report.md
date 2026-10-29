# Results report

## Alerts

- A_dscms_res_lam0_seed0: stopped at max_epochs (may not have converged).
- B_srcnn_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 86.8).
- B_srno_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 54.9).

## Runs (test set, m/s, fixed R), sorted by MSE

k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the threshold up to the last shell, so k_c is only a lower bound).

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_srno_res_lam1_seed0 | 170 | early_stopping | 0.0001179 | 54.03 | 0.998 | 0.006474 | 0.01081 | 0.0001166 | 63+ | 63 | 0.01454 | 0.0001073 | 48 |
| B_srno_res_lam0.1_seed0 | 125 | early_stopping | 0.0002318 | 51.08 | 0.9958 | 0.01483 | 0.02272 | 0.0002295 | 63+ | 63 | 0.03091 | 0.0002123 | 37 |
| B_srno_res_lam0.01_seed0 | 96 | early_stopping | 0.0004346 | 48.36 | 0.9916 | 0.05353 | 0.07664 | 0.0004294 | 63+ | 63 | 0.06055 | 0.0003989 | 28 |
| A_sofono_res_lam0_seed0 | 174 | early_stopping | 0.0005075 | 47.72 | 0.9902 | 0.11 | 0.1369 | 0.0004983 | 63+ | 63 | 0.07057 | 0.0004652 | 25 |
| B_srno_res_lam0.001_seed0 | 128 | early_stopping | 0.0005489 | 47.35 | 0.9897 | 0.1121 | 0.1465 | 0.0005419 | 63+ | 63 | 0.07708 | 0.0005042 | 24 |
| A_srno_res_lam0_seed0 | 89 | early_stopping | 0.0005818 | 47.11 | 0.9889 | 0.1389 | 0.1761 | 0.0005747 | 63+ | 63 | 0.0819 | 0.0005351 | 24 |
| B_srcnn_res_lam0.1_seed0 | 223 | early_stopping | 0.0006432 | 46.65 | 0.9879 | 0.05769 | 0.083 | 0.0006257 | 63+ | 62.87 | 0.08945 | 0.0005919 | 25 |
| A_prusr_res_lam0_seed0 | 192 | early_stopping | 0.0006489 | 46.64 | 0.987 | 0.1369 | 0.1746 | 0.00064 | 63+ | 63 | 0.09159 | 0.0005975 | 22 |
| B_srcnn_res_lam1_seed0 | 134 | early_stopping | 0.0006791 | 46.39 | 0.9878 | 0.03387 | 0.04645 | 0.0006364 | 63+ | 62.92 | 0.09093 | 0.0006207 | 25 |
| B_srcnn_res_lam0.01_seed0 | 198 | early_stopping | 0.0007532 | 45.97 | 0.9862 | 0.1477 | 0.2041 | 0.000733 | 63+ | 62.91 | 0.1052 | 0.000693 | 22 |
| A_dscms_res_lam0_seed0 | 300 | max_epochs | 0.0007806 | 45.82 | 0.9857 | 0.3547 | 0.5263 | 0.0007692 | 63+ | 62.74 | 0.1104 | 0.0007196 | 22 |
| B_srcnn_res_lam0.001_seed0 | 211 | early_stopping | 0.0008047 | 45.7 | 0.9852 | 0.2459 | 0.3114 | 0.0007851 | 63+ | 62.97 | 0.1129 | 0.0007421 | 21 |
| A_srcnn_res_lam0_seed0 | 213 | early_stopping | 0.0008302 | 45.57 | 0.9846 | 0.263 | 0.3272 | 0.0008116 | 63+ | 62.9 | 0.1168 | 0.0007652 | 20 |

Baselines:

| baseline | ruido | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | 0 | 0.004554 | 38.07 | 0.9614 | 42.68 | 56.37 | 0.003978 | 15 | 52.46 | 0.5811 | 8 |
| identity_noise0.0001 | 0.0001 | 0.004554 | 38.07 | 0.9614 | 42.53 | 56.14 | 0.003978 | 15 | 36.59 | 0.5812 | 8 |
| identity_noise0.001 | 0.001 | 0.004555 | 38.07 | 0.9614 | 34.32 | 44.03 | 0.003978 | 15 | 30.54 | 0.5813 | 8 |
| identity_noise0.003 | 0.003 | 0.004563 | 38.06 | 0.9611 | 21.49 | 26.02 | 0.003986 | 15 | 27.47 | 0.5822 | 8 |
| identity_noise0.01 | 0.01 | 0.004654 | 37.97 | 0.9586 | 8.93 | 9.481 | 0.004076 | 15 | 23.69 | 0.5924 | 8 |
| identity_noise0.03 | 0.03 | 0.005455 | 37.25 | 0.9369 | 2.333 | 1.785 | 0.004861 | 15 | 19.97 | 0.6826 | 8 |
| identity_noise0.1 | 0.1 | 0.01456 | 32.91 | 0.7488 | 1.909 | 2.793 | 0.01379 | 13 | 15.52 | 1.708 | 7 |
| wiener | 0 | 0.0003236 | 49.55 | 0.9968 | 5.322 | 10.03 | 0.0003236 | 51 | 52.7 | 0.04361 | 49 |
| wiener_noise0.0001 | 0.0001 | 0.0009436 | 44.9 | 0.9923 | 20.52 | 33.8 | 0.0009436 | 35 | 36.71 | 0.1348 | 32 |
| wiener_noise0.001 | 0.001 | 0.001398 | 43.2 | 0.9879 | 27.33 | 42.16 | 0.001398 | 29 | 30.69 | 0.2016 | 25 |
| wiener_noise0.003 | 0.003 | 0.00169 | 42.36 | 0.983 | 27.57 | 42.42 | 0.00169 | 26 | 27.56 | 0.2446 | 22 |
| wiener_noise0.01 | 0.01 | 0.002284 | 41.05 | 0.9768 | 31.72 | 46.8 | 0.002258 | 22 | 23.76 | 0.3282 | 17 |
| wiener_noise0.03 | 0.03 | 0.003299 | 39.43 | 0.9642 | 31.97 | 47.04 | 0.003054 | 18 | 20.03 | 0.4453 | 13 |
| wiener_noise0.1 | 0.1 | 0.006972 | 36.14 | 0.9338 | 31.72 | 46.79 | 0.004841 | 13 | 15.54 | 0.7082 | 9 |

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |
|---|---|---|---|---|---|---|---|---|
| srcnn | True | 0.001 | -2.548e-05 | +0.1297 | +0.0006662 | -0.01714 | -0.01588 | +0 |
| srcnn | True | 0.01 | -7.698e-05 | +0.4003 | +0.001598 | -0.1153 | -0.1231 | +0 |
| srcnn | True | 0.1 | -0.0001871 | +1.083 | +0.00337 | -0.2053 | -0.2442 | +0 |
| srcnn | True | 1 | -0.0001511 | +0.8228 | +0.003248 | -0.2292 | -0.2808 | +0 |
| srno | True | 0.001 | -3.295e-05 | +0.2434 | +0.0007663 | -0.02685 | -0.02961 | +0 |
| srno | True | 0.01 | -0.0001472 | +1.251 | +0.002687 | -0.0854 | -0.09946 | +0 |
| srno | True | 0.1 | -0.00035 | +3.978 | +0.006921 | -0.1241 | -0.1534 | +0 |
| srno | True | 1 | -0.0004639 | +6.922 | +0.009074 | -0.1325 | -0.1653 | +0 |

## Figures

- eps_k_mse+lpips_lam0.001_res.png
- eps_k_mse+lpips_lam0.01_res.png
- eps_k_mse+lpips_lam0.1_res.png
- eps_k_mse+lpips_lam1.0_res.png
- eps_k_mse_lam0.0_res.png
- error_by_band_mse+lpips_lam0.001_res.png
- error_by_band_mse+lpips_lam0.01_res.png
- error_by_band_mse+lpips_lam0.1_res.png
- error_by_band_mse+lpips_lam1.0_res.png
- error_by_band_mse_lam0.0_res.png
- lambda_sweep.png
- rho_k_mse+lpips_lam0.001_res.png
- rho_k_mse+lpips_lam0.01_res.png
- rho_k_mse+lpips_lam0.1_res.png
- rho_k_mse+lpips_lam1.0_res.png
- rho_k_mse_lam0.0_res.png
- spectra_mse+lpips_lam0.001_res.png
- spectra_mse+lpips_lam0.01_res.png
- spectra_mse+lpips_lam0.1_res.png
- spectra_mse+lpips_lam1.0_res.png
- spectra_mse_lam0.0_res.png
- train_curves.png
- transfer_k_mse+lpips_lam0.001_res.png
- transfer_k_mse+lpips_lam0.01_res.png
- transfer_k_mse+lpips_lam0.1_res.png
- transfer_k_mse+lpips_lam1.0_res.png
- transfer_k_mse_lam0.0_res.png
