# Legacy multistage (archive): NOT part of the campaign

Code, configs, logs and evaluation outputs of the multistage experiments done
before the campaign (September 2026), committed here for the record on
2026-10-01 by the user's decision. They had never been in git. Nothing here
is used by the campaign, and these results must not be reused in the thesis
(they were trained with the old, inconsistent early stopping and the
0.6 MSE + 0.4 LPIPS-type composite losses, which are not in the new grid).

The campaign's multistage is `train.py` with `STAGE1_CHECKPOINT` /
`FIXED_EPOCHS_FROM` (`models/stage_chain.py`, `scripts/make_block_e.py`,
DECISIONS.md section 4.7).

Layout (paths as they were in the repository root):

- `train_multistage.py`, `models/multistage.py`: the old stage-2 trainer and
  wrapper. They import `models.multistage` from the repository root, so they
  no longer run from here without copying them back.
- `configs/multistage*.yaml`: the old configs (50).
- `models/<ARCH>/test_composite_loss_*_multistage*/<date>/`: run folders
  with `epsilon.txt` and `output/logs/`. Checkpoints (`output/weights/`) and
  sample images stay on disk but are git-ignored.
- `notebooks/`: evaluation notebooks, summary CSVs, the PDF report and the
  per-run evaluation results; PNG plots are git-ignored.

Known difference from the campaign version: the old stage 2 of SRNO/SoFoNO
used the original models, which add a bilinear skip of x, so f_2 predicted
x + g(x); the campaign stages have no skip (RESIDUAL: false).
