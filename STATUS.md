# Campaign status

Last updated: 2026-10-01. Nothing of the campaign has been launched; no tag created. Pilot and short measurements are in `results_pilot/` only (`results/` holds only the baselines). Open points: `OPEN_QUESTIONS.md`.

## Pipeline stages

| Stage | State | Commit |
|---|---|---|
| 4.1 Reproducibility, full checkpoints, resume, CSV/JSON logs, `--max-hours` | done | 3580c63 |
| 4.2 Relative early stopping / scheduler / best.pt on validation MSE | done | 6c5e0ff |
| 4.3 Global residual option (same skip for all 5 architectures) | done | 56049e3 |
| 4.4 Loss MSE + lambda * LPIPS (fixed LPIPS scale) | done | ec5e162 |
| 4.5 Test metrics (fixed-R PSNR/SSIM, KE/WKE, bands, T(k), eps(k), k_c, slope), baselines | done | c790ce9, cf0ba61, b3dc0ad |
| Noisy regime B: `NOISE_STD` | done | d3ea418, 8b39cc5, 028de96 |
| Determinism: fast default, `DETERMINISTIC: true` per run | done | 08c2bc5 |
| `collect_results.py` | done | e98373b, cf0ba61 |
| User decisions recorded (sigma, lambda grid, slope range, SRNO deterministic) | done | 550f2e1 |
| Block B configs, STATUS.md | done | 4f26583 |
| Noise-sweep figure saved without dates | done | 6e6a3d6 |
| Pilot timings and GPU-hours (first version) | done | 3a557f8 |
| 4.6 Statistics (`stats.py`, block bootstrap 7 / 30 days, 2000 resamples, mean +- sd over seeds) | done | f649285 |
| `make_block_c.py --sweep-arch` with several architectures | done | ebefca8 |
| 4.7 Multistage stages and continuous-training control in `train.py` | done | 6f5c512 |
| Block D configs (`configs/D`, 5) and `make_block_e.py` | done | 9c73aab |
| `stats.py` multistage tables | done | 6e87591 |
| `.gitignore` for `results_pilot/` | done | a904059 |
| Pilot (all 10 runs), validation timing, GPU-hours with regime B = 13 runs, this file, `OPEN_QUESTIONS.md` | done | last commit |

## Tests (all pass, run on 2026-10-01 after the last code commit; log `results_pilot/tests.log`)

| Test | Checks | |
|---|---|---|
| `tests/test_residual.py` | 20/20 | old |
| `tests/test_loss.py` | 8/8 | old |
| `tests/test_metrics.py` | 16/16 | old |
| `tests/test_noise.py` | 11/11 | old |
| `tests/test_resume.py` (GPU) | 2/2 | old |
| `tests/test_stats.py` | 11/11 | new (4.6; 1 multistage check added with 4.7) |
| `tests/test_multistage.py` | 12/12 | new (4.7) |

## Decisions taken

- Stopping: relative improvement < 0.1% of validation MSE for 15 epochs, 50-300 epochs.
- R = 5.326794 m/s, SSIM offset -2.457645 m/s, LPIPS scale 2.869149 m/s (training HR).
- Determinism: fast mode (cudnn.benchmark) by default; SRNO runs with `DETERMINISTIC: true`.
- Regime B: sigma = 1e-2 m/s (3e-2 optional later); only after the advisor's approval.
- Lambda grid: {1e-3, 1e-2, 1e-1, 1} on SRNO and SRCNN (block B).
- Spectral-slope range: k = 11..46.
- Error bands: k < 5, 5-10, 10-20, 20-40, >= 40, plus the total for k >= D0.
- KE/WKE secondary in the noisy regime, always next to eps(k), k_c and T(k).
- Multistage (4.7): stage n = stage n-1 (frozen) + eps_n f_n(x) on the original input, eps_n = RMS of the previous residual on the training set, f_n trained with MSE on the normalized residual, `RESIDUAL: false`, `LAMBDA: 0`; clipping 0.5 and weight decay 1e-4 for every run (already the `train.py` setting); control = single stage for the measured epochs of stage 1 + stage 2. Details in `DECISIONS.md`.

## Blocks

| Block | Configs | State |
|---|---|---|
| A: 5 archs, MSE, seeds 0-2, residual | `configs/A/` (15) | generated |
| B: lambda sweep, SRNO + SRCNN | `configs/B/` (8) | generated |
| C: best lambda, 5 archs, seeds 0-2 | `make_block_c.py --lambda <v> --sweep-arch srno srcnn` (13) | waits for block B |
| D: residual=false ablation, MSE, seed 0 | `configs/D/` (5) | generated (cut to SRCNN + DSC/MS pending, see budget) |
| E: multistage (4 stages) + control | `make_block_e.py --arch <name>` (3 stages + control; stage 1 = an existing run) | waits for the best architecture |
| Regime B (sigma = 1e-2) | `make_block_n.py` (10); 13-run composition open | waits for the advisor and the best lambda |

## Pilot (`results_pilot/`, 2 epochs on the full data, real block-A configs with lambda 0 and 0.1)

Launched in tmux `pilot` (log `results_pilot/pilot.log`, commit f649285: DSC/MS x 2, PRUSR lambda 0.1) and `pilot2` (log `results_pilot/pilot2.log`, detached worktree at 6e87591: the other 7, refused in the first pass by the clean-tree guard, see `OPEN_QUESTIONS.md`). Validation timing in tmux `valtime` (`scripts/time_validation.py`, log `results_pilot/valtime.log`, `results_pilot/validation_time.csv`).

### Time per epoch, memory, GPU-hours per run

Seconds per epoch = epoch 2 of the pilot (after warm-up). "Train only" = epoch minus one timed validation pass (validation set, LPIPS always logged). Peak = nvidia-smi, whole V100 (16 384 MiB), sampled every 0.5 s during the run.

| Arch | Mode | lambda | s/epoch with val | val pass (s) | s/epoch train only | Peak (MiB) | GPU-h/run 50 / 150 / 300 ep |
|---|---|---|---|---|---|---|---|
| SRCNN | fast | 0 | 6.2 | 1.7 | 4.5 | 3 324 | 0.09 / 0.26 / 0.52 |
| SRCNN | fast | 0.1 | 31.9 | 1.7 | 30.1 | 4 758 | 0.44 / 1.33 / 2.65 |
| DSC/MS | fast | 0 | 20.5 | 2.4 | 18.1 | 3 558 | 0.29 / 0.86 / 1.71 |
| DSC/MS | fast | 0.1 | 46.1 | 2.4 | 43.7 | 6 278 | 0.64 / 1.92 / 3.85 |
| PRUSR | fast | 0 | 102.4 | 6.2 | 96.2 | 9 688 | 1.42 / 4.27 / 8.53 |
| PRUSR | fast | 0.1 | 128.1 | 6.2 | 121.9 | 9 692 | 1.78 / 5.34 / 10.67 |
| SRNO | deterministic | 0 | 179.4 | 7.2 | 172.2 | 12 294 | 2.49 / 7.47 / 14.95 |
| SRNO | deterministic | 0.1 | 206.6 | 7.2 | 199.4 | 13 214 | 2.87 / 8.61 / 17.21 |
| SoFoNO | fast | 0 | 242.9 | 10.5 | 232.5 | 13 204 | 3.37 / 10.12 / 20.24 |
| SoFoNO | fast | 0.1 | 276.4 | 10.5 | 265.9 | 13 268 | 3.84 / 11.52 / 23.03 |

No run above 81% of the card; no batch size was reduced. With lambda > 0 the LPIPS forward/backward dominates the cheap CNNs (SRCNN 5x slower per epoch).

### GPU-hours per block (`python scripts/gpu_hours.py --pilot results_pilot`, output in `results_pilot/gpu_hours.txt`)

Epochs per run: 50 (minimum), 150 (central), 300 (maximum). Multistage stage n costs 1 + 0.38 (n - 1) epochs of stage 1 (old PRUSR measurement).

| Block | Runs | 50 ep | 150 ep | 300 ep |
|---|---|---|---|---|
| A (5 archs x 3 seeds, MSE) | 15 | 23.0 | 68.9 | 137.9 |
| B (lambda sweep, 4 x SRNO + 4 x SRCNN) | 8 | 13.2 | 39.7 | 79.5 |
| C (best lambda, 5 archs x 3 seeds, minus the 2 seed-0 runs from B) | 13 | 25.4 | 76.2 | 152.4 |
| D (residual=false, 5 archs) | 5 | 7.7 | 23.0 | 46.0 |
| Regime B, sigma = 1e-2, 13 runs (5 MSE + sweep 4 x SRNO + 4 x SRCNN) | 13 | 20.9 | 62.7 | 125.4 |
| **A+B+C+D+regime B** | 54 | **90.2** | **270.5** | **541.1** |
| E (stages 2-4 + control; stage 1 is the block-A run), best arch SRCNN .. SoFoNO | 4 | 0.6-24.6 | 1.9-73.7 | 3.8-147.4 |
| **Total with E** | 58 | 91-115 | 272-344 | 545-688 |
| (reference) regime B as 10 runs (5 MSE + 5 best lambda) | 10 | 17.2 | 51.7 | 103.4 |

E per architecture (150 ep): SRCNN 1.9, DSC/MS 6.2, PRUSR 31.1, SRNO 54.4, SoFoNO 73.7 h. (Corrected in this commit: the earlier version also counted stage 1, which is the block-A run, and gave the control the epochs of all 4 stages instead of stage 1 + stage 2.)

**Budget rule:** the central total is above ~250 h. Proposed cuts, **not applied** (waiting for you):

| Cut | Runs | 50 ep | 150 ep | 300 ep |
|---|---|---|---|---|
| A: PRUSR with 2 seeds | -1 | -1.4 | -4.3 | -8.5 |
| C: PRUSR with 2 seeds | -1 | -1.8 | -5.3 | -10.7 |
| D: ablation only on SRCNN and DSC/MS | -3 | -7.3 | -21.9 | -43.7 |
| **A+B+C+D+regime B after the cuts** | 49 | **79.7** | **239.1** | **478.2** |
| **Total with E after the cuts** | 53 | 80-104 | 241-313 | 482-626 |

After the cuts the central total stays under 250 h only if E runs on SRCNN (241.0 h) or DSC/MS (245.3 h); with PRUSR 270.2 h, SRNO 293.5 h, SoFoNO 312.8 h.

### Pilot results (2 epochs: a smoke test of the whole chain, not a comparison)

Figures from `collect_results.py --results-dir results_pilot --out results_pilot/report` (PDF + PNG):
- Training and validation curves: `results_pilot/report/train_curves.pdf`. All 10 runs decrease in train loss and validation MSE from epoch 1 to 2; best.pt = epoch 2 everywhere.
- Output spectrum against the target: `results_pilot/report/spectra_mse_lam0.0_res.pdf` and `spectra_mse+lpips_lam0.1_res.pdf`; also `eps_k_*`, `transfer_k_*`, `rho_k_*`, `error_by_band_*`.

Test set after 2 epochs:

| Arch | lambda | MSE | PSNR | SSIM | k_c | eps (k >= D0) | T (k >= D0) | KE |
|---|---|---|---|---|---|---|---|---|
| SRCNN | 0 | 2.23e-3 | 41.18 | 0.973 | 21 | 0.288 | 0.285 | 1.71 |
| SRCNN | 0.1 | 2.38e-3 | 40.88 | 0.974 | 23 | 0.282 | 0.396 | 1.07 |
| DSC/MS | 0 | 2.95e-3 | 39.97 | 0.968 | 18 | 0.393 | 0.168 | 3.58 |
| DSC/MS | 0.1 | 3.02e-3 | 39.85 | 0.968 | 19 | 0.384 | 0.294 | 2.21 |
| PRUSR | 0 | 2.00e-3 | 41.67 | 0.972 | 23 | 0.260 | 0.408 | 0.91 |
| PRUSR | 0.1 | 1.64e-3 | 42.49 | 0.975 | 27 | 0.208 | 0.600 | 0.33 |
| SRNO | 0 | 1.24e-3 | 43.75 | 0.981 | 63 | 0.169 | 0.537 | 0.48 |
| SRNO | 0.1 | 7.65e-4 | 45.84 | 0.988 | 63 | 0.098 | 0.870 | 0.04 |
| SoFoNO | 0 | 1.14e-3 | 44.15 | 0.982 | 63 | 0.153 | 0.751 | 0.21 |
| SoFoNO | 0.1 | 6.33e-4 | 46.66 | 0.989 | 63 | 0.081 | 0.883 | 0.03 |

Reading: every model is below the input (identity, MSE 4.55e-3) after 2 epochs and far from Wiener (3.24e-4). Below k ~ 10 all outputs follow the target. Above it the spectra fall under the target (T < 1, energy lost at small scales), least for SoFoNO and SRNO and most for DSC/MS. With lambda = 0.1, T(k >= D0) rises for every architecture (more small-scale energy), and KE drops. For the operators, eps(k) never reaches 1/2, so k_c saturates at 63 (`OPEN_QUESTIONS.md`, item 6).

## Baselines (test set, sigma = 0)

| | MSE | PSNR | SSIM | k_c | k_c_fase |
|---|---|---|---|---|---|
| identity | 4.55e-3 | 38.07 | 0.961 | 15 | 52.5 |
| Wiener (gamma 1e-12) | 3.24e-4 | 49.55 | 0.997 | 51 | 52.7 |

With sigma = 1e-2: Wiener MSE 2.28e-3, PSNR 41.05, k_c 22 (`tables/wiener_noise_sweep.csv`).

## Next steps (in order, after your answers in `OPEN_QUESTIONS.md`)

1. Approve or adjust the budget cuts; then launch block A (`python train.py --config configs/A/*.yaml --max-hours 70`), then B.
2. Choose lambda from B; `make_block_c.py --lambda <v> --sweep-arch srno srcnn`; then D.
3. Choose the architecture for E; `make_block_e.py --arch <name>`; queue stages 2, 3, 4, then the control.
4. Regime B after the advisor's approval and the 13-run composition.
