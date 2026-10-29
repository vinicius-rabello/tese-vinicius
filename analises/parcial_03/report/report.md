# Results report

## Alerts

- A_dscms_res_lam0_seed0: stopped at max_epochs (may not have converged).

## Runs (test set, m/s, fixed R), sorted by MSE

k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the threshold up to the last shell, so k_c is only a lower bound).

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_srno_res_lam0_seed0 | 89 | early_stopping | 0.0005818 | 47.11 | 0.9889 | 0.1389 | 0.1761 | 0.0005747 | 63+ | 63 | 0.0819 | 0.0005351 | 24 |
| A_prusr_res_lam0_seed0 | 192 | early_stopping | 0.0006489 | 46.64 | 0.987 | 0.1369 | 0.1746 | 0.00064 | 63+ | 63 | 0.09159 | 0.0005975 | 22 |
| A_dscms_res_lam0_seed0 | 300 | max_epochs | 0.0007806 | 45.82 | 0.9857 | 0.3547 | 0.5263 | 0.0007692 | 63+ | 62.74 | 0.1104 | 0.0007196 | 22 |
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

Not evaluated yet / incomplete: A_sofono_res_lam0_seed0 (incompleta)

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

(no architecture has both an MSE and an MSE+LPIPS run yet)

## Figures

- eps_k_mse_lam0.0_res.png
- error_by_band_mse_lam0.0_res.png
- rho_k_mse_lam0.0_res.png
- spectra_mse_lam0.0_res.png
- train_curves.png
- transfer_k_mse_lam0.0_res.png
