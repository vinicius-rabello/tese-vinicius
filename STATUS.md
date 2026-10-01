# Campaign status

Last updated: 2026-10-01 (after the user's answers to OPEN_QUESTIONS). Nothing of the campaign has been launched; no tag created. How to run it: `RUNNING.md`. Pilot and short measurements are in `results_pilot/` only (`results/` holds only the baselines). Open points: `OPEN_QUESTIONS.md`.

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
| Pilot (all 10 runs), validation timing, GPU-hours with regime B = 13 runs, STATUS.md, `OPEN_QUESTIONS.md` | done | 69b6f62 |
| k_c with two thresholds (eps <= 0.5, <= 0.1) and censoring markers | done | 290861f |
| Block D reduced (SRCNN, DSC/MS, SRNO); PRUSR/SoFoNO in `configs/D_optional/` | done | eebd84b |
| `make_block_n.py --sweep-arch srcnn` (regime B, 13 configs) | done | 6dfb2a1 |
| Legacy multistage archived in `archive/legacy_multistage/` | done | 133e3fa |
| `gpu_hours.py --measured`, decided composition, hours per queue | done | bc0b5bc |
| Queues `queues/fase1-7.txt` + `scripts/run_queue.py` | done | 4db8079 |
| `RUNNING.md` (worktree at `campanha-v1`), this update | done | last commit |

## Tests (all pass, run on 2026-10-01 after the last code commit; log `results_pilot/tests.log`)

`scripts/run_queue.py` was checked by hand: refusal outside a worktree at a `campanha-*` tag (tag simulated, no tag created), a tiny queue trained and evaluated end to end, a relaunch with nothing left to do, and every queue resolved against generated C/E/N configs in a temporary folder.

| Test | Checks | |
|---|---|---|
| `tests/test_residual.py` | 20/20 | old |
| `tests/test_loss.py` | 8/8 | old |
| `tests/test_metrics.py` | 19/19 | old (16) + 3 new for k_c_01 / censoring |
| `tests/test_noise.py` | 11/11 | old |
| `tests/test_resume.py` (GPU) | 2/2 | old |
| `tests/test_stats.py` | 13/13 | new (4.6; + multistage with 4.7; + k_c_01 / censoring) |
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
- k_c reported with two thresholds (eps <= 0.5: `k_c`; eps <= 0.1: `k_c_01`) and a censoring marker when it reaches the last shell (shown as `63+`).
- Budget: no cuts in A and C; block D only on SRCNN, DSC/MS, SRNO.
- Regime B: 5 MSE + 5 best lambda + lambda sweep on SRCNN = 13 runs, seed 0, after the advisor's approval.
- Block E: stage 1 = block A run (MSE, seed 0) of the best architecture of A; stages 2-4 with min 50 epochs; control with the measured epochs of stages 1 + 2.
- Multistage (4.7): stage n = stage n-1 (frozen) + eps_n f_n(x) on the original input, eps_n = RMS of the previous residual on the training set, f_n trained with MSE on the normalized residual, `RESIDUAL: false`, `LAMBDA: 0`; clipping 0.5 and weight decay 1e-4 for every run (already the `train.py` setting); control = single stage for the measured epochs of stage 1 + stage 2. Details in `DECISIONS.md`.

## Blocks and queues

| Block | Configs | State |
|---|---|---|
| A: 5 archs, MSE, seeds 0-2, residual | `configs/A/` (15) | generated |
| B: lambda sweep, SRNO + SRCNN | `configs/B/` (8) | generated |
| C: best lambda, 5 archs, seeds 0-2 | `make_block_c.py --lambda <v> --sweep-arch srno srcnn` (13) | waits for block B |
| D: residual=false ablation, MSE, seed 0 | `configs/D/` (3: SRCNN, DSC/MS, SRNO); `configs/D_optional/` (PRUSR, SoFoNO, in no queue) | generated |
| E: multistage (stages 2-4) + control | `make_block_e.py --arch <best of A>` (4) | waits for block A |
| Regime B (sigma = 1e-2) | `make_block_n.py --noise 0.01 --lambda <v> --sweep-arch srcnn` (13) | waits for the advisor and lambda |

Queues (`queues/`, each run with `python scripts/run_queue.py queues/<fase>.txt --max-hours 70` from the campaign worktree, see `RUNNING.md`):

| Queue | Content | Runs | GPU-h (150 ep) | 70 h blocks |
|---|---|---|---|---|
| fase1 | A seed 0 (SRCNN, DSC/MS, PRUSR, SRNO, SoFoNO), then B | 13 | 62.7 | 1 (2 if runs go to ~300 epochs) |
| fase2 | C seed 0 | 3 | 18.8 | 1 |
| fase3 | D reduced | 3 | 8.6 | 1 |
| fase4 | A seeds 1-2 | 10 | 46.0 | 1 |
| fase5 | C seeds 1-2 | 10 | 57.4 | 1 |
| fase6 | E (stages 2-4 + control) | 4 | 1.9-73.7 | 1-2 (depends on the architecture) |
| fase7 | regime B | 13 | 55.7 | 1 |

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

### GPU-hours per block (`python scripts/gpu_hours.py`, output in `results_pilot/gpu_hours.txt`)

Composition as decided on 2026-10-01. Epochs per run: 50 (minimum), 150 (central), 300 (maximum). After block A seed 0, `python scripts/gpu_hours.py --measured results` replaces the scenarios with the measured epochs per architecture (and the measured s/epoch). Multistage stage n costs 1 + 0.38 (n - 1) epochs of stage 1 (old PRUSR measurement).

| Block | Runs | 50 ep | 150 ep | 300 ep |
|---|---|---|---|---|
| A (5 archs x 3 seeds, MSE) | 15 | 23.0 | 68.9 | 137.9 |
| B (lambda sweep, 4 x SRNO + 4 x SRCNN) | 8 | 13.2 | 39.7 | 79.5 |
| C (best lambda, 5 archs x 3 seeds, minus the 2 seed-0 runs from B) | 13 | 25.4 | 76.2 | 152.4 |
| D (residual=false: SRCNN, DSC/MS, SRNO) | 3 | 2.9 | 8.6 | 17.2 |
| Regime B, sigma = 1e-2 (5 MSE + 5 best lambda + 3 SRCNN sweep) | 13 | 18.6 | 55.7 | 111.3 |
| **A+B+C+D+regime B** | 52 | **83.0** | **249.1** | **498.2** |
| E (stages 2-4 + control; stage 1 is the block-A run), best arch SRCNN .. SoFoNO | 4 | 0.6-24.6 | 1.9-73.7 | 3.8-147.4 |
| **Total with E** | 56 | 84-108 | 251-323 | 502-646 |
| (not counted) D_optional: PRUSR, SoFoNO | 2 | 4.8 | 14.4 | 28.8 |

E per architecture (150 ep): SRCNN 1.9, DSC/MS 6.2, PRUSR 31.1, SRNO 54.4, SoFoNO 73.7 h.

### Pilot results (2 epochs: a smoke test of the whole chain, not a comparison)

Figures from `collect_results.py --results-dir results_pilot --out results_pilot/report` (PDF + PNG):
- Training and validation curves: `results_pilot/report/train_curves.pdf`. All 10 runs decrease in train loss and validation MSE from epoch 1 to 2; best.pt = epoch 2 everywhere.
- Output spectrum against the target: `results_pilot/report/spectra_mse_lam0.0_res.pdf` and `spectra_mse+lpips_lam0.1_res.pdf`; also `eps_k_*`, `transfer_k_*`, `rho_k_*`, `error_by_band_*`.

Test set after 2 epochs:

| Arch | lambda | MSE | PSNR | SSIM | k_c | k_c_01 | eps (k >= D0) | T (k >= D0) | KE |
|---|---|---|---|---|---|---|---|---|---|
| SRCNN | 0 | 2.23e-3 | 41.18 | 0.973 | 21 | 13 | 0.288 | 0.285 | 1.71 |
| SRCNN | 0.1 | 2.38e-3 | 40.88 | 0.974 | 23 | 11 | 0.282 | 0.396 | 1.07 |
| DSC/MS | 0 | 2.95e-3 | 39.97 | 0.968 | 18 | 11 | 0.393 | 0.168 | 3.58 |
| DSC/MS | 0.1 | 3.02e-3 | 39.85 | 0.968 | 19 | 11 | 0.384 | 0.294 | 2.21 |
| PRUSR | 0 | 2.00e-3 | 41.67 | 0.972 | 23 | 13 | 0.260 | 0.408 | 0.91 |
| PRUSR | 0.1 | 1.64e-3 | 42.49 | 0.975 | 27 | 15 | 0.208 | 0.600 | 0.33 |
| SRNO | 0 | 1.24e-3 | 43.75 | 0.981 | 63+ | 17 | 0.169 | 0.537 | 0.48 |
| SRNO | 0.1 | 7.65e-4 | 45.84 | 0.988 | 63+ | 23 | 0.098 | 0.870 | 0.04 |
| SoFoNO | 0 | 1.14e-3 | 44.15 | 0.982 | 63+ | 18 | 0.153 | 0.751 | 0.21 |
| SoFoNO | 0.1 | 6.33e-4 | 46.66 | 0.989 | 63+ | 25 | 0.081 | 0.883 | 0.03 |

Reading: every model is below the input (identity, MSE 4.55e-3) after 2 epochs and far from Wiener (3.24e-4). Below k ~ 10 all outputs follow the target. Above it the spectra fall under the target (T < 1, energy lost at small scales), least for SoFoNO and SRNO and most for DSC/MS. With lambda = 0.1, T(k >= D0) rises for every architecture (more small-scale energy), and KE drops. For the operators, eps(k) never reaches 1/2, so k_c is censored at 63 (`63+`). The stricter k_c_01 (eps <= 0.1) still separates them: SRNO 17/23, SoFoNO 18/25, against 11-15 for the CNNs.

## Baselines (test set, sigma = 0)

| | MSE | PSNR | SSIM | k_c | k_c_fase |
|---|---|---|---|---|---|
| identity | 4.55e-3 | 38.07 | 0.961 | 15 | 52.5 |
| Wiener (gamma 1e-12) | 3.24e-4 | 49.55 | 0.997 | 51 | 52.7 |

With sigma = 1e-2: Wiener MSE 2.28e-3, PSNR 41.05, k_c 22 (`tables/wiener_noise_sweep.csv`).

## Next steps (nothing launched until you confirm)

1. Your answers to `OPEN_QUESTIONS.md` items 1-2 (selection criterion on validation; versioning of `results/`).
2. On your go-ahead: tag `campanha-v1`, create the worktree and launch `fase1` (`RUNNING.md`, sections 2-3).
3. After A seed 0: `python scripts/gpu_hours.py --measured results`; choose the architecture for E.
4. After B: choose lambda; generate C; `fase2`, `fase3`, `fase4`, `fase5`; generate E, `fase6`.
5. Regime B (`fase7`) after the advisor's approval.
