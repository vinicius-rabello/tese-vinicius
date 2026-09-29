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
