# Results report

## Alerts

- P_dscms_res_lam0.1_seed0: stopped at max_epochs (may not have converged).
- P_dscms_res_lam0_seed0: stopped at max_epochs (may not have converged).
- P_prusr_res_lam0.1_seed0: stopped at max_epochs (may not have converged).
- P_prusr_res_lam0_seed0: stopped at max_epochs (may not have converged).
- P_sofono_res_lam0.1_seed0: stopped at max_epochs (may not have converged).
- P_sofono_res_lam0_seed0: stopped at max_epochs (may not have converged).
- P_srcnn_res_lam0.1_seed0: stopped at max_epochs (may not have converged).
- P_srcnn_res_lam0_seed0: stopped at max_epochs (may not have converged).
- P_srno_res_lam0.1_seed0: stopped at max_epochs (may not have converged).
- P_srno_res_lam0_seed0: stopped at max_epochs (may not have converged).

## Runs (test set, m/s, fixed R), sorted by MSE

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_sofono_res_lam0.1_seed0 | 2 | max_epochs | 0.0006334 | 46.66 | 0.9892 | 0.03494 | 0.05146 | 0.0005706 | 63 | 63 | 0.08126 | 0.0005787 |
| P_srno_res_lam0.1_seed0 | 2 | max_epochs | 0.0007649 | 45.84 | 0.9877 | 0.03841 | 0.05632 | 0.0006822 | 63 | 62.95 | 0.0977 | 0.0007055 |
| P_sofono_res_lam0_seed0 | 2 | max_epochs | 0.001137 | 44.14 | 0.9815 | 0.2075 | 0.2031 | 0.001056 | 63 | 62.84 | 0.1527 | 0.001039 |
| P_srno_res_lam0_seed0 | 2 | max_epochs | 0.001241 | 43.75 | 0.9805 | 0.4752 | 0.5606 | 0.001171 | 63 | 62.64 | 0.1695 | 0.001117 |
| P_prusr_res_lam0.1_seed0 | 2 | max_epochs | 0.001645 | 42.49 | 0.9747 | 0.3342 | 0.4839 | 0.001463 | 27 | 58.52 | 0.2079 | 0.00151 |
| P_prusr_res_lam0_seed0 | 2 | max_epochs | 0.001995 | 41.67 | 0.9724 | 0.9094 | 1.138 | 0.001787 | 23 | 61.15 | 0.26 | 0.00182 |
| P_srcnn_res_lam0_seed0 | 2 | max_epochs | 0.002231 | 41.18 | 0.9729 | 1.707 | 2.156 | 0.001979 | 21 | 59.55 | 0.288 | 0.002001 |
| P_srcnn_res_lam0.1_seed0 | 2 | max_epochs | 0.002377 | 40.88 | 0.9741 | 1.073 | 1.521 | 0.001941 | 23 | 59.85 | 0.2824 | 0.002174 |
| P_dscms_res_lam0_seed0 | 2 | max_epochs | 0.002952 | 39.97 | 0.9679 | 3.578 | 4.617 | 0.002698 | 18 | 46.56 | 0.3935 | 0.002729 |
| P_dscms_res_lam0.1_seed0 | 2 | max_epochs | 0.003017 | 39.85 | 0.968 | 2.21 | 3.207 | 0.002637 | 19 | 27.35 | 0.3844 | 0.002744 |

Not evaluated yet / incomplete: configs (incompleta), report (incompleta)

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |
|---|---|---|---|---|---|---|---|---|
| dscms | True | 0.1 | +6.545e-05 | -0.1214 | +0.000141 | -1.368 | -1.411 | +1 |
| prusr | True | 0.1 | -0.0003505 | +0.8247 | +0.002332 | -0.5752 | -0.6544 | +4 |
| sofono | True | 0.1 | -0.0005032 | +2.519 | +0.0077 | -0.1726 | -0.1516 | +0 |
| srcnn | True | 0.1 | +0.0001467 | -0.2989 | +0.001203 | -0.6343 | -0.635 | +2 |
| srno | True | 0.1 | -0.0004759 | +2.087 | +0.007204 | -0.4368 | -0.5043 | +0 |

## Figures

- eps_k_mse+lpips_lam0.1_res.png
- eps_k_mse_lam0.0_res.png
- error_by_band_mse+lpips_lam0.1_res.png
- error_by_band_mse_lam0.0_res.png
- lambda_sweep.png
- rho_k_mse+lpips_lam0.1_res.png
- rho_k_mse_lam0.0_res.png
- spectra_mse+lpips_lam0.1_res.png
- spectra_mse_lam0.0_res.png
- train_curves.png
- transfer_k_mse+lpips_lam0.1_res.png
- transfer_k_mse_lam0.0_res.png
