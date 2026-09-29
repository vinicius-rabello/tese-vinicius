# Campaign status

Last updated: 2026-09-29. Nothing of the campaign has been launched; no tag created.

## Pipeline stages

| Stage | State | Commit |
|---|---|---|
| 4.1 Reproducibility, full checkpoints, resume, CSV/JSON logs, `--max-hours` | done | 3580c63 |
| 4.2 Relative early stopping / scheduler / best.pt on validation MSE | done | 6c5e0ff |
| 4.3 Global residual option (same skip for all 5 architectures) | done | 56049e3 |
| 4.4 Loss MSE + lambda * LPIPS (fixed LPIPS scale) | done | ec5e162 |
| 4.5 Test metrics (fixed-R PSNR/SSIM, KE/WKE, bands, T(k), eps(k), k_c, slope), baselines | done | c790ce9, cf0ba61 |
| Noisy regime B: `NOISE_STD` | done | d3ea418 |
| Determinism: fast default, `DETERMINISTIC: true` per run | done | 08c2bc5 |
| `collect_results.py` | done | e98373b, cf0ba61 |
| 4.6 Statistics (`stats.py`, block bootstrap) | not started | |
| 4.7 Multistage (4 stages + continuous-training control) | not started | |
| Block D and E generators | not started | |

Tests: `tests/test_residual.py` 20/20, `test_loss.py` 8/8, `test_metrics.py` 16/16, `test_noise.py` 11/11, `test_resume.py` 2/2.

## Decisions taken

- Stopping: relative improvement < 0.1% of validation MSE for 15 epochs, 50-300 epochs.
- R = 5.326794 m/s, SSIM offset -2.457645 m/s, LPIPS scale 2.869149 m/s (training HR).
- Determinism: fast mode (cudnn.benchmark) by default; SRNO runs with `DETERMINISTIC: true`.
- Regime B: sigma = 1e-2 m/s (3e-2 optional later).
- Lambda grid: {1e-3, 1e-2, 1e-1, 1} on SRNO and SRCNN (block B).
- Spectral-slope range: k = 11..46.
- Error bands: k < 5, 5-10, 10-20, 20-40, >= 40, plus the total for k >= D0.

## Blocks

| Block | Configs | State |
|---|---|---|
| A: 5 archs, MSE, seeds 0-2, residual | `configs/A/` (15) | generated |
| B: lambda sweep, SRNO + SRCNN | `configs/B/` (8) | generated |
| C: best lambda, 5 archs, seeds 0-2 | `make_block_c.py` | waits for block B |
| D: residual=false ablation | not written | |
| E: multistage + control | not written | |
| Regime B (sigma = 1e-2) | `make_block_n.py` | waits for the best lambda |

## Baselines (test set, sigma = 0)

| | MSE | PSNR | SSIM | k_c | k_c_fase |
|---|---|---|---|---|---|
| identity | 4.55e-3 | 38.07 | 0.961 | 15 | 52.5 |
| Wiener (gamma 1e-12) | 3.24e-4 | 49.55 | 0.997 | 51 | 52.7 |

With sigma = 1e-2: Wiener MSE 2.28e-3, PSNR 41.05, k_c 22 (`tables/wiener_noise_sweep.csv`).

## Measurements

| Arch | Mode | Lambda | s/epoch (train + val) | nvidia-smi peak |
|---|---|---|---|---|
| PRUSR | fast | 0 | 112.6 | (allocated 9.8 GB) |
| SRNO | fast | 0 | 180.4 | 14 616 MiB (89%) |
| SRNO | deterministic | 0.1 | 207.4 | 13 212 MiB (81%) |
| SoFoNO | fast | 0 | 245.6 | 13 204 MiB (81%) |
| SoFoNO | fast | 0.1 | 280.7 | 13 266 MiB (81%) |

Short pilot (real block configs, 2 epochs, full data; seconds per epoch = second epoch, training + validation with LPIPS logging):

| Arch | Mode | lambda = 0 | lambda > 0 | Peak lambda = 0 / > 0 (MiB) |
|---|---|---|---|---|
| SRCNN | fast | 6.2 | 31.9 | 3324 / 4726 |
| DSC/MS | fast | 20.6 | 46.2 | 3470 / 5640 |
| PRUSR | fast | 104.1 | 128.8 | 9688 / 9688 |
| SRNO | deterministic | 179.5 | 207.4 (memory check) | 12294 / 13212 |
| SoFoNO | fast | 243.0 | 280.7 (memory check) | 13204 / 13266 |

Memory check done: no run above ~15 GB, so no batch size was reduced.

## GPU-hours per block (`scripts/gpu_hours.py`)

Epochs per run: 50 (minimum), 150 (central; the new rule replayed on the old LPIPS logs gives 139-202 for the CNNs, >= 86-107 for the operators, and every old MSE run was cut short, `scripts/estimate_epochs.py`), 300 (maximum).

| Block | Runs | 50 ep | 150 ep | 300 ep |
|---|---|---|---|---|
| A (5 archs x 3 seeds, MSE) | 15 | 23.1 | 69.2 | 138.3 |
| B (lambda sweep, SRNO + SRCNN) | 8 | 13.3 | 39.9 | 79.8 |
| C (best lambda, 3 seeds, minus the 2 seed-0 runs from B) | 13 | 25.6 | 76.9 | 153.8 |
| D (residual=false, 5 archs) | 5 | 7.7 | 23.1 | 46.1 |
| Regime B, sigma = 1e-2 (5 MSE + 5 best lambda) | 10 | 17.3 | 52.0 | 104.0 |
| **A+B+C+D+regime B** | 51 | **87.0** | **261.0** | **522.0** |
| E (4 stages + control), best arch SRCNN .. SoFoNO | 5 | 0.9-34.7 | 2.7-104.1 | 5.3-208.1 |
| **Total with E** | 56 | 88-122 | 264-365 | 527-730 |

The central estimate is above ~250 h; nothing has been cut (waiting for the user). Regime B is counted as 10 runs (what make_block_n.py generates); the user mentioned 13.

## Open items

- Decide on the budget (the central estimate is above ~250 h).
- Composition of regime B (10 runs as generated, or 13).
- make_block_c.py takes one sweep architecture; with the sweep on SRNO and SRCNN it must skip both seed-0 runs (to be adapted before block C).
