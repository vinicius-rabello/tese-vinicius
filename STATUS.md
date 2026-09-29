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

Memory check done: no run above ~15 GB, so no batch size was reduced.

## Open items

- Short pilot and GPU-hours estimate per block.
