# Results report (validation set)

## Alerts

- A_dscms_res_lam0_seed0: stopped at max_epochs (may not have converged).
- B_srcnn_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 86.8).
- B_srno_res_lam1_seed0: LPIPS dominates the loss at the end (λ·LPIPS/MSE = 54.9).

## Runs (validation set, m/s, fixed R), sorted by MSE

k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the threshold up to the last shell, so k_c is only a lower bound).

| run_id | epocas | motivo_parada | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | best_val_mse | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_srno_res_lam1_seed0 | 170 | early_stopping | 0.0001073 | 54.51 | 0.9981 | 0.00813 | 0.01357 | 0.0001062 | 63+ | 63 | 0.0149 | 0.0001073 | 47 |
| B_srno_res_lam0.1_seed0 | 125 | early_stopping | 0.0002123 | 51.52 | 0.996 | 0.01744 | 0.02671 | 0.0002103 | 63+ | 63 | 0.03183 | 0.0002123 | 36 |
| B_srno_res_lam0.01_seed0 | 96 | early_stopping | 0.0003989 | 48.8 | 0.992 | 0.06113 | 0.08822 | 0.0003942 | 63+ | 62.95 | 0.06249 | 0.0003989 | 28 |
| A_sofono_res_lam0_seed0 | 174 | early_stopping | 0.0004652 | 48.17 | 0.9907 | 0.1238 | 0.1547 | 0.0004564 | 63+ | 62.78 | 0.07258 | 0.0004652 | 25 |
| B_srno_res_lam0.001_seed0 | 128 | early_stopping | 0.0005042 | 47.79 | 0.9901 | 0.1285 | 0.168 | 0.0004978 | 63+ | 62.79 | 0.07955 | 0.0005042 | 24 |
| A_srno_res_lam0_seed0 | 89 | early_stopping | 0.0005351 | 47.54 | 0.9894 | 0.1618 | 0.2053 | 0.0005284 | 63+ | 62.74 | 0.08463 | 0.0005351 | 24 |
| B_srcnn_res_lam0.1_seed0 | 223 | early_stopping | 0.0005919 | 47.07 | 0.9885 | 0.06242 | 0.09041 | 0.000576 | 35 | 61.86 | 0.09256 | 0.0005919 | 25 |
| A_prusr_res_lam0_seed0 | 192 | early_stopping | 0.0005975 | 47.06 | 0.9875 | 0.1717 | 0.2197 | 0.0005892 | 63+ | 62.4 | 0.09476 | 0.0005975 | 22 |
| B_srcnn_res_lam1_seed0 | 134 | early_stopping | 0.0006207 | 46.84 | 0.9884 | 0.03661 | 0.05079 | 0.0005859 | 37 | 61.92 | 0.09407 | 0.0006207 | 25 |
| B_srcnn_res_lam0.01_seed0 | 198 | early_stopping | 0.000693 | 46.39 | 0.9867 | 0.1604 | 0.2207 | 0.0006746 | 63+ | 61.72 | 0.1088 | 0.000693 | 22 |
| A_dscms_res_lam0_seed0 | 300 | max_epochs | 0.0007196 | 46.24 | 0.9863 | 0.3974 | 0.583 | 0.0007091 | 63+ | 61.2 | 0.1144 | 0.0007196 | 21 |
| B_srcnn_res_lam0.001_seed0 | 211 | early_stopping | 0.0007421 | 46.11 | 0.9858 | 0.2741 | 0.344 | 0.0007237 | 63+ | 61.83 | 0.1169 | 0.0007421 | 20 |
| A_srcnn_res_lam0_seed0 | 213 | early_stopping | 0.0007652 | 45.99 | 0.9852 | 0.2938 | 0.3621 | 0.0007478 | 63+ | 61.65 | 0.1209 | 0.0007652 | 20 |

Baselines:

| baseline | ruido | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c | k_c_fase | eps_k | k_c_01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | 0 | 0.004028 | 38.69 | 0.9628 | 41.3 | 54.34 | 0.003526 | 15 | 52.83 | 0.5788 | 8 |
| identity_noise0.01 | 0.01 | 0.004127 | 38.57 | 0.9599 | 8.318 | 8.704 | 0.003624 | 15 | 23.51 | 0.5914 | 8 |
| wiener | 0 | 0.0002832 | 50.25 | 0.997 | 5.084 | 9.583 | 0.0002832 | 52 | 53.15 | 0.0428 | 48 |
| wiener_noise0.01 | 0.01 | 0.00203 | 41.65 | 0.9775 | 30.35 | 44.81 | 0.002008 | 22 | 23.56 | 0.3279 | 17 |

## Difference to plain MSE (same architecture and residual flag, mean over seeds)

| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |
|---|---|---|---|---|---|---|---|---|
| srcnn | True | 0.001 | -2.314e-05 | +0.1248 | +0.0006488 | -0.01964 | -0.01801 | +0 |
| srcnn | True | 0.01 | -7.226e-05 | +0.4007 | +0.001578 | -0.1334 | -0.1414 | +0 |
| srcnn | True | 0.1 | -0.0001733 | +1.081 | +0.003311 | -0.2314 | -0.2716 | -28 |
| srcnn | True | 1 | -0.0001445 | +0.8535 | +0.003195 | -0.2572 | -0.3113 | -26 |
| srno | True | 0.001 | -3.085e-05 | +0.2448 | +0.0007411 | -0.03331 | -0.03723 | +0 |
| srno | True | 0.01 | -0.0001362 | +1.253 | +0.002592 | -0.1007 | -0.117 | +0 |
| srno | True | 0.1 | -0.0003228 | +3.981 | +0.006659 | -0.1443 | -0.1786 | +0 |
| srno | True | 1 | -0.0004278 | +6.966 | +0.008745 | -0.1537 | -0.1917 | +0 |

## Lambda selection (a priori rule, validation set)

- srcnn: candidates (MSE <= 1.1 x lambda=0): 0.001, 0.01, 0.1, 1; lowest eps_w_10_40 0.06501; within 2%: 1; chosen (smallest): 1 [sensitivity, eps_hf: eps_hf would choose 0.1] (`selecao_lambda_srcnn.csv`)
- srno: candidates (MSE <= 1.1 x lambda=0): 0.001, 0.01, 0.1, 1; lowest eps_w_10_40 0.0019104; within 2%: 1; chosen (smallest): 1 [sensitivity, eps_hf: same lambda] (`selecao_lambda_srno.csv`)
- Decision: `python scripts/make_block_c.py --lambda 1 --sweep-arch srno srcnn`; srcnn 1, dscms 1, prusr 1, srno 1, sofono 1

## Figures

- eps_k_mse+lpips_lam0.001_res_val.png
- eps_k_mse+lpips_lam0.01_res_val.png
- eps_k_mse+lpips_lam0.1_res_val.png
- eps_k_mse+lpips_lam1.0_res_val.png
- eps_k_mse_lam0.0_res_val.png
- error_by_band_mse+lpips_lam0.001_res_val.png
- error_by_band_mse+lpips_lam0.01_res_val.png
- error_by_band_mse+lpips_lam0.1_res_val.png
- error_by_band_mse+lpips_lam1.0_res_val.png
- error_by_band_mse_lam0.0_res_val.png
- lambda_sweep_val.png
- rho_k_mse+lpips_lam0.001_res_val.png
- rho_k_mse+lpips_lam0.01_res_val.png
- rho_k_mse+lpips_lam0.1_res_val.png
- rho_k_mse+lpips_lam1.0_res_val.png
- rho_k_mse_lam0.0_res_val.png
- spectra_mse+lpips_lam0.001_res_val.png
- spectra_mse+lpips_lam0.01_res_val.png
- spectra_mse+lpips_lam0.1_res_val.png
- spectra_mse+lpips_lam1.0_res_val.png
- spectra_mse_lam0.0_res_val.png
- train_curves_val.png
- transfer_k_mse+lpips_lam0.001_res_val.png
- transfer_k_mse+lpips_lam0.01_res_val.png
- transfer_k_mse+lpips_lam0.1_res_val.png
- transfer_k_mse+lpips_lam1.0_res_val.png
- transfer_k_mse_lam0.0_res_val.png
