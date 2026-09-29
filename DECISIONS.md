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
