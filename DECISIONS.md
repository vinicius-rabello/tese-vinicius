# Implementation decisions

One line per decision not covered by the instructions, with the reason.

## 4.1 Reproducibility and pipeline infrastructure

- Run outputs go to `results/<run_id>/`, `run_id = <RUN_NAME>_seed<SEED>` (`RUN_NAME` defaults to the YAML file name); the old timestamp suffix on `ROOT_FOLDER` is gone because automatic resume needs a folder that is the same on every launch. `ROOT_FOLDER` is no longer read by `train.py`.
- `last.pt` (full state) is overwritten every epoch and `best.pt` is written on improvement; the old every-50-epochs snapshots were dropped since `last.pt` supersedes them.
- Checkpoints and JSON files are written to a `.tmp` file and then `os.replace`d, so a kill in the middle of a save can never leave a truncated file.
- Checkpoints are loaded with `map_location='cpu'`: RNG states must stay CPU byte tensors, and `load_state_dict` moves weights/optimizer state to the model's device anyway.
- On resume, rows of `train_log.csv` for epochs after the checkpoint's epoch are dropped, so an epoch interrupted between the CSV append and the checkpoint save is not logged twice.
- The DataLoader generator state is saved in the checkpoint, so the shuffle order after a resume is identical to an uninterrupted run (verified: bit-identical losses at every epoch).
- `--max-hours` is a budget for the whole queue, not per run; a run stops *before* starting an epoch that would cross it (using the last epoch's duration as the estimate), writes `summary.json` with status `interrupted_max_hours`, and the rest of the queue stays pending. Relaunching the same command continues it.
- Relaunching skips runs whose `summary.json` status is `completed_early_stop` or `completed_max_epochs`; raising `NUM_EPOCHS` later does not reopen a finished run (delete its `summary.json` to do that). A run that crashed has no summary and resumes from `last.pt`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set when `utils/reproducibility.py` is imported: cuBLAS requires it to be deterministic and reads it when its handle is created.
- The validation DataLoader also gets a seeded generator and `worker_init_fn` (it doesn't shuffle, but this keeps any future randomness in workers seeded).
- The git `dirty` flag in `config.json` only looks at tracked files; untracked outputs in the repo don't change what code ran.
- PSNR/SSIM range R = max − min of the training HR fields (u and v together) = **5.326794 m/s**; LPIPS scale = max(|u|, |v|) over the training HR fields = **2.869149 m/s**. Both are recomputed from the full training set at the start of every run (even with `SUBSET_TRAIN`) and stored in `config.json`; they are used from 4.4/4.5 on.
- `SUBSET_TRAIN` / `SUBSET_VAL` config keys (first N samples) exist only for quick sanity runs.
- `EarlyStopping` and `save_checkpoint` keep their old signatures because `train_multistage.py` imports them; multistage moves to the new infrastructure in 4.7.
- Real (non-subset) runs refuse to start if tracked files have uncommitted changes (override: `ALLOW_DIRTY: true`), so every campaign `config.json` points at an exact commit; subset sanity runs are allowed on a dirty tree.
- Launch 72 h lab blocks with `--max-hours 70`: the run stops before an epoch that would cross the budget, and the 2 h margin covers lab interruptions and the final save.

## 4.2 Early stopping, scheduler and checkpoint

- An epoch counts as an improvement if `val_mse < es_best * (1 - 1e-3)`, where `es_best` is the early stopper's own reference (updated only on such improvements), not the previous epoch.
- The early-stopping counter runs from epoch 1, but stopping can only happen at epoch >= `min_epochs` (50); a plateau that began earlier therefore stops the run exactly at epoch 50.
- `best.pt` is saved on any strict decrease of `val_mse`, so it can come from an epoch the stopping rule counted as "no significant improvement" (seen in the tests: new best at epoch 52 while the counter went to 1).
- `ReduceLROnPlateau` keeps its previous settings (patience 5, factor 0.5, PyTorch's default relative threshold 1e-4) and is now fed the validation MSE.
- `NUM_EPOCHS` is ignored by `train.py`; the limits are `MIN_EPOCHS`=50 and `MAX_EPOCHS`=300 by default. The keys `MIN_EPOCHS`, `MAX_EPOCHS`, `ES_PATIENCE`, `ES_REL_DELTA` exist only to shrink quick tests; the resolved values are saved under `stopping` in `config.json`.
- Thesis wording for the rule: "each run trains at least 50 epochs and stops when the relative improvement of the validation MSE stays below 0.1% for 15 consecutive epochs (at most 300 epochs)".
- `EarlyStopping` gained `relative` and `min_epochs` arguments; with the defaults (`relative=False`, `min_epochs=0`) it behaves exactly like the old class, which `train_multistage.py` still uses (checked on 500 random sequences: identical counters, bests and stop flags).

## 4.3 Global residual

- `create_model(name, residual=True|False)` wraps every architecture in `GlobalResidual`: `y = x + f(x)` when true, `y = f(x)` when false. SRNO/SoFoNO's built-in bilinear input skip is switched off (`input_skip = False`) in both cases, so `residual=false` means no global skip for all five architectures (for SRNO/SoFoNO this removes a skip the original models had).
- `create_model(name)` with `residual=None` still returns the untouched original models, because the old checkpoints, `train_multistage.py`, the report scripts and the notebooks load them that way.
- Checkpoints trained with the new option have their keys under `net.` (the wrapper); load them with `create_model(name, residual=<value in config.json>)`.
- `RESIDUAL` has no default: a config without an explicit `true`/`false` is rejected, and the run name gets `_res` or `_nores`, so block D (ablation) can never land under a block A name.
- SoFoNO builds its query grid with `ranges=(-1, 1)`, the same as SRNO, so its bilinear skip is also exactly the identity at s = 1 (max abs diff 0.0); both operators are therefore numerically unchanged with `residual=true`.
- Thesis wording (ch. 4): SRNO and SoFoNO had a global skip and the CNNs didn't, but at s = 1 that skip is exactly x, so the residual vs non-residual comparison is clean.
- `tests/test_residual.py` loads the model code of commit 6c5e0ff (before the change) from git and checks bit-identical outputs with the same weights: the 3 CNNs with `residual=false`, SRNO/SoFoNO with `residual=true`, and all 5 with `residual=None`; plus test (e) and a zeroed-output check that `residual=false` leaves no hidden skip.

## 4.4 Loss MSE + lambda * LPIPS

- The loss is `MseLpipsLoss` (`losses/losses.py`); `LOSS_MODE`/`LOSS_WEIGHTS` are no longer read by `train.py`, and `FluidCompositeLoss` stays only for old code. With lambda = 0 the LPIPS network is never built during training, and loss and gradients are bit-identical to the old MSE loss (tested).
- LPIPS preprocessing lives in one function, `losses/lpips_utils.py:to_lpips_input` (component / 2.869149 m/s, repeated to 3 channels); `lpips_distance` averages LPIPS(u) and LPIPS(v). The training loss uses it now and the evaluation metric (4.5) must import the same function. Predictions are not clamped, so they may slightly exceed [-1, 1].
- LPIPS backbone for training: VGG (`lpips` 0.1.4, v0.1 linear weights).
- Validation LPIPS is computed and logged every epoch for every lambda (so lambda = 0 runs log it too); training LPIPS is only computed when lambda > 0 and is `nan` in the CSV otherwise.
- Epoch means (train and validation) are now weighted by the number of samples; before, they were a plain mean over batches, which gave the smaller last batch extra weight.
- `LAMBDA` has no default: a config without an explicit number >= 0 is rejected. It is part of the run name (`_lam<value>`, e.g. `_lam0`, `_lam0.01`) and is stored in `config.json` and `summary.json`.
- The one-epoch magnitude check (`scripts/lpips_magnitude_check.py`) uses the batch sizes and learning rates used so far: 32 and 1e-4 for SRCNN/DSC-MS/PRUSR, 16 and 4e-5 for SRNO, 8 and 4e-5 for SoFoNO.

## 4.5 Evaluation metrics

- The new evaluation (`evaluate_run.py`, `evaluation/spectral_metrics.py`) does not use `metrics/image_metrics.py` (per-sample-rescaled PSNR/SSIM, FSIM, EPI); that file stays only for the old notebooks.
- MSE and PSNR are per test sample over u and v together; PSNR = 10 log10(R^2 / MSE) with R = 5.326794 m/s.
- SSIM is implemented directly instead of calling `piq.ssim`, because piq rejects inputs outside [0, data_range] and predictions can fall slightly outside the training range. Same formula and constants as piq (11x11 Gaussian window, sigma 1.5, K1 = 0.01, K2 = 0.03, no padding); matches piq to 3e-6 where piq accepts the input. Fields are shifted by one fixed offset, the training minimum (-2.457645 m/s, `hr_train_min` in `config.json`), with `data_range = R`; no clipping, no per-sample rescaling; mean of u and v.
- Radial shells follow the existing KE convention: k = |(kx, ky)| in cycles per 128-pixel domain, shell [k, k + 1), k = 1..63.
- Per-band error: MSE split by Parseval into bands with edges k = 0, 5, 10, 20, 40 and the corners (k up to 64 sqrt 2), so the bands add up exactly to the MSE; `band_k_ge_D0` = bands with k >= 10.
- rho(k) sums u and v together within each shell (absolute value of the complex cross sum). k_c follows the FRC "first crossing" convention: the last k before rho first drops below 1/2, so noise re-crossing 1/2 at high k can't inflate it; 0 if rho(1) < 1/2, 63 if it never drops.
- The spectral-slope function is in place but its fitting range is not fixed yet (awaiting confirmation). Every evaluation stores the per-sample radial spectra of prediction and target in `spectra.npz`, so the slope can be computed later without re-running any model.
- Secondary LPIPS (appendix): VGG and AlexNet backbones, through the same `lpips_utils.lpips_distance` the training loss uses (a test checks it is the same function object).
- Test dates come from the netCDF `MT` variable (`scripts/make_test_dates.py` -> `evaluation/test_dates.csv`): 974 days, 2022-01-01 to 2024-08-31, in the .npy order. The patch (22.9-27.5 N, 90.1-85.0 W) is all ocean, and one HR pixel is 0.04 deg (about 4 km), so k = 1 is a wavelength of about 5.1 deg.
- Wiener/Tikhonov: the same H(k) as the dataset; gamma is chosen on a log grid 1e-8..1 (33 values) by the lowest validation MSE; the chosen gamma and the whole curve go into `results/baseline_wiener/config.json`.
- Evaluation reads architecture, residual flag and the three constants from the run's `config.json` and refuses the run (clear error, run listed as not evaluated) if the file or any key is missing, instead of building a model with guessed settings.

## Experiment blocks

- Block configs are generated, never written by hand: `scripts/blocks.py` holds the per-architecture batch size and learning rate (32 and 1e-4 for SRCNN/DSC-MS/PRUSR; 16 and 4e-5 for SRNO; 8 and 4e-5 for SoFoNO, as used so far) and the YAML format, and `make_block_a.py`, `make_block_b.py`, `make_block_c.py` go through it. Every config also sets `NUM_WORKERS: 4` and `IMAGE_SAVE_INTERVAL: 25`.
- Config files are `configs/<block>/<block>_<arch>_<res|nores>_lam<lambda>_seed<seed>.yaml`, and `RUN_NAME = <block>_<arch>`, so the run id names the block, architecture, residual flag, lambda and seed.
- `make_block_b.py` has no defaults: the sweep architecture and the 4 lambda values are chosen after the magnitude check, and lambda = 0 is refused (it is block A).
- `make_block_c.py --lambda <value> --sweep-arch <arch>` skips seed 0 of the sweep architecture, which is the block B run `B_<arch>_res_lam<value>_seed0`; `collect_results.py` must take that run for C's seed 0.
- Wiener gamma grid is 1e-16..1 (33 values, half-decade steps) and `choose_gamma` fails if the best value lands on the grid edge. The first grid (1e-8..1) was wrong: the degradation is noiseless, so validation MSE keeps falling down to gamma = 1e-12 (val MSE 2.8e-4), where only float32 round-off of the LR input limits the inverse; the filter then recovers k up to ~53 of 64.

## collect_results.py

- Outputs follow PROMPT_collect_results.md; the report folder also gets `runs/<run_id>/` with each run's `config.json`, `summary.json`, `train_log.csv` and `eval_summary.json` (all small), so the analysis doesn't need access to `results/`. No `metrics_per_sample.csv`, `spectra.npz` or `*.pt` go into the report.
- `status` is an extra column (`completa`/`incompleta`): a run is complete only with status `completed_early_stop` or `completed_max_epochs`. `motivo_parada` maps those to `early_stopping`/`max_epochs`, `interrupted_max_hours` to `max_hours`, and is empty for a run still in progress.
- Band columns: the spec asks for `e_banda_1..4` with band 4 = k > D0, but the 4.5 bands are five (k < 5, 5-10, 10-20, 20-40, >= 40), so the CSV has `e_banda_1..5` plus `e_banda_k_ge_D0`. PENDING the user's choice; any regrouping works from the stored per-sample bands without re-evaluating.
- `lpips_apendice` = LPIPS with the AlexNet backbone; `lpips_vgg` (the training backbone) is an extra column.
- `inclinacao_espectral` stays empty until the slope range is confirmed (`SLOPE_RANGE` in the script); it is then computed from the stored spectra.
- A finished run without `metrics_per_sample.csv` is evaluated once (test set, `no_grad`, `evaluate_run.evaluate`) and its test means are written to `summary.json` under `test_metrics`; `--no-eval` skips this. Later calls never re-evaluate.
- Figures are grouped by comparable runs: same loss, lambda, residual flag (and stage for multistage), all architectures together, one line/bar per architecture averaged over its seeds (n shown in the legend). The LR curve comes from `results/baseline_identity/spectra.npz`. Files are `spectra_<group>`, `rho_k_<group>`, `error_by_band_<group>`, plus `train_curves` (one panel per run) and `lambda_sweep` (only when an architecture has test metrics at more than one lambda; symlog x axis so lambda = 0 fits).
- Idempotent: figures are saved without creation dates or software stamps, so two calls on the same results give byte-identical folders (tested); only the .zip differs in file timestamps.
- The .zip is written next to the report folder as `report_<folder name>.zip`, and `*.pt` is excluded explicitly.
- Alert "PSNR inconsistent": per test sample, |PSNR - 10 log10(R^2/MSE)| > 1e-3 dB. Alert "LPIPS dominates": lambda * train LPIPS / train MSE > 10 in the last logged epoch.

## Noisy regime (regime B): NOISE_STD

- `SuperResDataset(noise_std=0.0, noise_mode='fixed'|'random', noise_seed_offset=0)`: white Gaussian noise (m/s) added to the LR input after the low-pass filter; the HR target stays clean. With `noise_std = 0` (the default) no noise code runs and no RNG is touched, so the dataset is bit-identical to before (tested against the code of commit e98373b).
- Fixed noise (validation, test, baselines): sample i draws its noise from its own CPU generator seeded with `offset + i`, so every model, the identity and the Wiener filter see exactly the same noise on the same sample. Offsets: 1,000,000 for validation, 2,000,000 for test (`NOISE_SEED_OFFSET`), so the two splits never share a realization.
- Training noise: a fresh draw on every access from torch's global RNG, so it changes every epoch and is controlled by the run seed through `set_seed`, the DataLoader generator and `worker_init_fn` (and restored on resume with the loader generator state).
- `NOISE_STD` (default 0) is validated as a number >= 0, stored in `config.json`, `summary.json` and as a new last column `noise_std` of `train_log.csv`; the run name gets `_noise<sigma>` before `_seed` only when it is > 0, so all existing names and YAMLs stay valid.
- Evaluation reads `noise_std` from the run's `config.json` (a missing key means 0: those runs were trained without noise); baselines take `--noise-std` and go to `baseline_<name>_noise<sigma>` when it is > 0, leaving `baseline_identity`/`baseline_wiener` for the noiseless regime.
- R, the SSIM offset and the LPIPS scale come from the clean training HR and do not depend on the noise.

## Amplitude-aware spectral metrics (k_c redefined)

- The old k_c (rho(k) first dropping below 1/2) did not distinguish blur: rho(k) is a phase coherence, and a Gaussian blur multiplies each shell by an almost constant real factor that cancels out, so the blurred LR input got k_c = 52.5, the same as Wiener (52.7). rho(k) is kept, now documented as "phase coherence" in code, docstrings and figures, and its k_c is kept as `k_c_fase`.
- Added per shell [k, k + 1): transfer ratio T(k) = E_yhat(k) / E_y(k) (amplitude) and relative spectral error eps(k) = E|Yhat - Y|^2 / E|Y|^2 (amplitude and phase), with E over the test set: the shell sums of |Yhat|^2, |Y|^2 and |Yhat - Y|^2 are stored per sample in `spectra.npz` and pooled over the test set.
- New k_c = the largest k such that eps(k') <= 0.5 for every k' <= k. The run-level `k_c` (in `eval_summary.json` under `pooled`, and in `collect_results.py`) is computed from the pooled eps(k); `metrics_per_sample.csv` has a per-sample `k_c` (per-sample eps) for the bootstrap, next to `k_c_fase`. With it the blurred input gets k_c = 15 (analytically 15.7, where (1 - H)^2 = 1/2) and Wiener 51.
- CHANGED, not only added (as instructed): the `k_c` column of `metrics_per_sample.csv` and of `collect_results.py` now has the new definition. For evaluations made before this change, `collect_results.py` reads their `k_c` column as `k_c_fase` and leaves the new `k_c`, `T_k` and `eps_k` empty (no mislabeled numbers); re-evaluating fills them.
- `collect_results.py` columns: `T_k` = mean of the pooled T(k) over k >= D0, `eps_k` = pooled eps over k >= D0 (sum of error power over sum of target power), `k_c`, `k_c_fase`, `ruido` (noise_std); the full curves are in each run's `eval_summary.json`. New figures `transfer_k_<group>` and `eps_k_<group>` with the input (LR) and Wiener curves of the same noise level for reference; groups now also split by noise level (`_noise<sigma>` only when > 0). `baselines.csv` lists every `results/baseline_*` folder with its noise level.
- Error bands: the 5 bands (k < 5, 5-10, 10-20, 20-40, >= 40) plus the total for k >= D0, as chosen by the user.
- Spectral-slope range: `scripts/propose_slope_range.py` proposes k = 11..46 from the test-set target spectrum (local slope within +-0.5 of its median -2.02 above D0; target slope over the range -2.11), figure in `figures/slope_range_proposal.png`. NOT fixed yet: `SLOPE_RANGE` stays None until the user confirms.

## Wiener against noise (for choosing sigma)

- `scripts/wiener_noise_sweep.py` runs the unchanged `evaluate_run.py --baseline identity wiener --noise-std sigma` for sigma in {0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} m/s (gamma chosen on validation per sigma, same fixed test noise for both baselines, no LPIPS), and writes `tables/wiener_noise_sweep.csv`, `tables/wiener_noise_eps_k.csv` (eps(k), T(k) per shell) and `figures/wiener_kc_vs_sigma.png/.pdf`. Sigma is not chosen here.
- Reference scales, over the training HR fields: std of the velocity field (u and v together) 0.4198 m/s, mean speed |V| = sqrt(u^2 + v^2) 0.4922 m/s, mean |u|, |v| 0.3137 m/s. The 0.33 m/s quoted earlier was the mean |component| of one test year, not the mean speed.
- Caveat found in the sweep: KE and WKE (log-spectral) reward any energy at high k, correct or not, so noise alone improves them for the identity (KE 42.7 at sigma = 0, 1.9 at sigma = 0.1, better than Wiener). In the noisy regime they must be read together with eps(k)/k_c, which don't have this problem (identity k_c stays 15).
- Block N (noisy regime B): `scripts/make_block_n.py --noise <sigma> --lambda <value>`, both required and > 0, writes `configs/N/`: the 5 architectures with lambda = 0 and the 5 with the chosen lambda, residual, seed 0, `NOISE_STD: sigma` (10 runs). `blocks.write_config` got an optional `noise` argument; with the default 0 it writes byte-identical files to before (block A regenerated and compared). `make_block_b.py`/`make_block_c.py` are unchanged; the SRCNN sweep next to SRNO's is `make_block_b.py --arch srcnn` with the same lambdas.
