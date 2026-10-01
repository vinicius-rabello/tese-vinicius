# Results report (validation set)

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

## Runs (validation set, m/s, fixed R), sorted by MSE

k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the threshold up to the last shell, so k_c is only a lower bound).

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_sofono_res_lam0.1_seed0 | 2 | max_epochs | 0.0005787 | 47.12 | 0.9897 | 0.04129 | 0.0611 | 0.0005253 | 63+ | 62.68 | 0.08408 | 0.0005787 | 24 |
| P_srno_res_lam0.1_seed0 | 2 | max_epochs | 0.0007055 | 46.24 | 0.9882 | 0.03334 | 0.04872 | 0.0006331 | 63+ | 62.48 | 0.1019 | 0.0007055 | 23 |
| P_sofono_res_lam0_seed0 | 2 | max_epochs | 0.001039 | 44.58 | 0.9822 | 0.2495 | 0.2501 | 0.0009669 | 63+ | 61.03 | 0.157 | 0.001039 | 17 |
| P_srno_res_lam0_seed0 | 2 | max_epochs | 0.001117 | 44.27 | 0.9813 | 0.431 | 0.5044 | 0.001057 | 63+ | 60.73 | 0.1718 | 0.001117 | 17 |
| P_prusr_res_lam0.1_seed0 | 2 | max_epochs | 0.00151 | 42.88 | 0.9755 | 0.3548 | 0.5174 | 0.001354 | 25 | 54.4 | 0.2159 | 0.00151 | 15 |
| P_prusr_res_lam0_seed0 | 2 | max_epochs | 0.00182 | 42.1 | 0.9733 | 0.841 | 1.046 | 0.001646 | 22 | 56.84 | 0.269 | 0.00182 | 13 |
| P_srcnn_res_lam0_seed0 | 2 | max_epochs | 0.002001 | 41.7 | 0.9738 | 1.698 | 2.143 | 0.001777 | 20 | 56.82 | 0.2905 | 0.002001 | 13 |
| P_srcnn_res_lam0.1_seed0 | 2 | max_epochs | 0.002174 | 41.32 | 0.975 | 1.15 | 1.649 | 0.001772 | 23 | 55.67 | 0.2897 | 0.002174 | 11 |
| P_dscms_res_lam0_seed0 | 2 | max_epochs | 0.002729 | 40.34 | 0.9691 | 3.781 | 4.812 | 0.002486 | 17 | 39.5 | 0.4073 | 0.002729 | 11 |
| P_dscms_res_lam0.1_seed0 | 2 | max_epochs | 0.002744 | 40.29 | 0.969 | 2.265 | 3.282 | 0.002408 | 19 | 25.88 | 0.3944 | 0.002744 | 11 |

Not evaluated yet / incomplete: configs (incompleta), report (incompleta), report_val (incompleta)

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |
|---|---|---|---|---|---|---|---|---|
| dscms | True | 0.1 | +1.437e-05 | -0.04882 | -3.6e-05 | -1.516 | -1.529 | +2 |
| prusr | True | 0.1 | -0.0003101 | +0.7775 | +0.002153 | -0.4862 | -0.5284 | +3 |
| sofono | True | 0.1 | -0.0004604 | +2.536 | +0.00746 | -0.2082 | -0.189 | +0 |
| srcnn | True | 0.1 | +0.0001731 | -0.38 | +0.001223 | -0.5484 | -0.4938 | +3 |
| srno | True | 0.1 | -0.0004117 | +1.965 | +0.006923 | -0.3977 | -0.4557 | +0 |

## Lambda selection (a priori rule, validation set)

- srcnn: candidates (MSE <= 1.1 x lambda=0): 0.1; lowest eps_hf 0.45939; within 2%: 0.1; chosen (smallest): 0.1 (`selecao_lambda_srcnn.csv`)
- srno: candidates (MSE <= 1.1 x lambda=0): 0.1; lowest eps_hf 0.24455; within 2%: 0.1; chosen (smallest): 0.1 (`selecao_lambda_srno.csv`)
- Decision: `python scripts/make_block_c.py --lambda 0.1 --sweep-arch srno srcnn`; srcnn 0.1, dscms 0.1, prusr 0.1, srno 0.1, sofono 0.1

## Figures

- eps_k_mse+lpips_lam0.1_res_val.png
- eps_k_mse_lam0.0_res_val.png
- error_by_band_mse+lpips_lam0.1_res_val.png
- error_by_band_mse_lam0.0_res_val.png
- lambda_sweep_val.png
- rho_k_mse+lpips_lam0.1_res_val.png
- rho_k_mse_lam0.0_res_val.png
- spectra_mse+lpips_lam0.1_res_val.png
- spectra_mse_lam0.0_res_val.png
- train_curves_val.png
- transfer_k_mse+lpips_lam0.1_res_val.png
- transfer_k_mse_lam0.0_res_val.png
