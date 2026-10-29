# Results report

## Alerts

- A_dscms_res_lam0_seed0: stopped at max_epochs (may not have converged).
- B_srcnn_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 86.8).
- B_srno_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 54.9).

## Runs (test set, hann window, m/s, fixed R), sorted by MSE

k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the threshold up to the last shell, so k_c is only a lower bound).

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_srno_res_lam1_seed0 | 170 | early_stopping | 0.0001179 | 54.03 | 0.998 | 0.2309 | 0.4066 | 7.824e-05 | 51 | 52.87 | 0.01753 | 0.0001073 | 47 |
| B_srno_res_lam0.1_seed0 | 125 | early_stopping | 0.0002318 | 51.08 | 0.9958 | 0.4664 | 0.7383 | 0.000165 | 43 | 45.98 | 0.04148 | 0.0002123 | 36 |
| B_srno_res_lam0.01_seed0 | 96 | early_stopping | 0.0004346 | 48.36 | 0.9916 | 3.752 | 6.634 | 0.0003422 | 34 | 35.96 | 0.09184 | 0.0003989 | 28 |
| A_sofono_res_lam0_seed0 | 174 | early_stopping | 0.0005075 | 47.72 | 0.9902 | 12.5 | 19.75 | 0.0004166 | 30 | 35.68 | 0.1129 | 0.0004652 | 25 |
| B_srno_res_lam0.001_seed0 | 128 | early_stopping | 0.0005489 | 47.35 | 0.9897 | 11.15 | 18.15 | 0.0004277 | 30 | 33.43 | 0.116 | 0.0005042 | 25 |
| A_srno_res_lam0_seed0 | 89 | early_stopping | 0.0005818 | 47.11 | 0.9889 | 14.05 | 22 | 0.0004657 | 29 | 33.6 | 0.1267 | 0.0005351 | 24 |
| B_srcnn_res_lam0.1_seed0 | 223 | early_stopping | 0.0006432 | 46.65 | 0.9879 | 3.508 | 6.15 | 0.0005034 | 30 | 31.93 | 0.1374 | 0.0005919 | 25 |
| A_prusr_res_lam0_seed0 | 192 | early_stopping | 0.0006489 | 46.64 | 0.987 | 15.43 | 23.54 | 0.000536 | 27 | 30.48 | 0.1466 | 0.0005975 | 22 |
| B_srcnn_res_lam1_seed0 | 134 | early_stopping | 0.0006791 | 46.39 | 0.9878 | 1.197 | 2.05 | 0.0005035 | 30 | 32.36 | 0.1372 | 0.0006207 | 26 |
| B_srcnn_res_lam0.01_seed0 | 198 | early_stopping | 0.0007532 | 45.97 | 0.9862 | 8.952 | 14.88 | 0.0005843 | 27 | 29.26 | 0.1602 | 0.000693 | 22 |
| A_dscms_res_lam0_seed0 | 300 | max_epochs | 0.0007806 | 45.82 | 0.9857 | 24.5 | 34.11 | 0.000605 | 26 | 32.73 | 0.1661 | 0.0007196 | 22 |
| B_srcnn_res_lam0.001_seed0 | 211 | early_stopping | 0.0008047 | 45.7 | 0.9852 | 21.79 | 31.18 | 0.0006362 | 25 | 32.6 | 0.1749 | 0.0007421 | 21 |
| A_srcnn_res_lam0_seed0 | 213 | early_stopping | 0.0008302 | 45.57 | 0.9846 | 23.04 | 32.49 | 0.0006679 | 25 | 33 | 0.1838 | 0.0007652 | 20 |

Baselines:

| baseline | ruido | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | 0 | 0.004554 | 38.07 | 0.9614 | 31.55 | 39.59 | 0.001888 | 15 | 39.06 | 0.5284 | 8 |
| identity_noise0.0001 | 0.0001 | 0.004554 | 38.07 | 0.9614 | 31.42 | 39.4 | 0.001888 | 15 | 35.05 | 0.5284 | 8 |
| identity_noise0.001 | 0.001 | 0.004555 | 38.07 | 0.9614 | 24.56 | 29.57 | 0.001889 | 15 | 29.23 | 0.5286 | 8 |
| identity_noise0.003 | 0.003 | 0.004563 | 38.06 | 0.9611 | 14.18 | 15.57 | 0.001897 | 15 | 26.1 | 0.5303 | 8 |
| identity_noise0.01 | 0.01 | 0.004654 | 37.97 | 0.9586 | 5.057 | 4.447 | 0.001986 | 15 | 22.4 | 0.55 | 8 |
| identity_noise0.03 | 0.03 | 0.005455 | 37.25 | 0.9369 | 2 | 2.071 | 0.002771 | 14 | 18.64 | 0.7228 | 8 |
| identity_noise0.1 | 0.1 | 0.01456 | 32.91 | 0.7488 | 5.828 | 9.161 | 0.0117 | 12 | 14.26 | 2.691 | 7 |
| wiener | 0 | 0.0003236 | 49.55 | 0.9968 | 2.572 | 4.871 | 0.0001145 | 49 | 50.62 | 0.02753 | 47 |
| wiener_noise0.0001 | 0.0001 | 0.0009436 | 44.9 | 0.9923 | 12.41 | 20.4 | 0.0003062 | 34 | 35.16 | 0.08166 | 31 |
| wiener_noise0.001 | 0.001 | 0.001398 | 43.2 | 0.9879 | 17.48 | 26.7 | 0.0004999 | 28 | 29.26 | 0.1364 | 24 |
| wiener_noise0.003 | 0.003 | 0.00169 | 42.36 | 0.983 | 17.6 | 26.84 | 0.0006967 | 24 | 26.12 | 0.192 | 21 |
| wiener_noise0.01 | 0.01 | 0.002284 | 41.05 | 0.9768 | 20.81 | 30.31 | 0.000992 | 21 | 22.43 | 0.2754 | 17 |
| wiener_noise0.03 | 0.03 | 0.003299 | 39.43 | 0.9642 | 20.97 | 30.46 | 0.001567 | 17 | 18.66 | 0.4379 | 13 |
| wiener_noise0.1 | 0.1 | 0.006972 | 36.14 | 0.9338 | 20.79 | 30.28 | 0.003014 | 12 | 14.26 | 0.8466 | 8 |

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |
|---|---|---|---|---|---|---|---|---|
| srcnn | True | 0.001 | -2.548e-05 | +0.1297 | +0.0006663 | -1.247 | -1.306 | +0 |
| srcnn | True | 0.01 | -7.698e-05 | +0.4003 | +0.001598 | -14.09 | -17.61 | +2 |
| srcnn | True | 0.1 | -0.0001871 | +1.083 | +0.00337 | -19.53 | -26.34 | +5 |
| srcnn | True | 1 | -0.0001511 | +0.8228 | +0.003248 | -21.84 | -30.44 | +5 |
| srno | True | 0.001 | -3.295e-05 | +0.2434 | +0.0007663 | -2.894 | -3.85 | +1 |
| srno | True | 0.01 | -0.0001472 | +1.251 | +0.002687 | -10.3 | -15.36 | +5 |
| srno | True | 0.1 | -0.00035 | +3.978 | +0.006921 | -13.58 | -21.26 | +14 |
| srno | True | 1 | -0.0004639 | +6.922 | +0.009074 | -13.82 | -21.59 | +22 |

## Figures

- eps_k_mse+lpips_lam0.001_res_hann.png
- eps_k_mse+lpips_lam0.01_res_hann.png
- eps_k_mse+lpips_lam0.1_res_hann.png
- eps_k_mse+lpips_lam1.0_res_hann.png
- eps_k_mse_lam0.0_res_hann.png
- error_by_band_mse+lpips_lam0.001_res_hann.png
- error_by_band_mse+lpips_lam0.01_res_hann.png
- error_by_band_mse+lpips_lam0.1_res_hann.png
- error_by_band_mse+lpips_lam1.0_res_hann.png
- error_by_band_mse_lam0.0_res_hann.png
- lambda_sweep_hann.png
- rho_k_mse+lpips_lam0.001_res_hann.png
- rho_k_mse+lpips_lam0.01_res_hann.png
- rho_k_mse+lpips_lam0.1_res_hann.png
- rho_k_mse+lpips_lam1.0_res_hann.png
- rho_k_mse_lam0.0_res_hann.png
- spectra_mse+lpips_lam0.001_res_hann.png
- spectra_mse+lpips_lam0.01_res_hann.png
- spectra_mse+lpips_lam0.1_res_hann.png
- spectra_mse+lpips_lam1.0_res_hann.png
- spectra_mse_lam0.0_res_hann.png
- train_curves_hann.png
- transfer_k_mse+lpips_lam0.001_res_hann.png
- transfer_k_mse+lpips_lam0.01_res_hann.png
- transfer_k_mse+lpips_lam0.1_res_hann.png
- transfer_k_mse+lpips_lam1.0_res_hann.png
- transfer_k_mse_lam0.0_res_hann.png
