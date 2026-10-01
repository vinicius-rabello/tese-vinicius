# Open questions

Points that need a decision, decisions already taken (kept for the record), and what I changed instead of only adding. The default in use is stated in each open item.

## Open

1. **Two readings of the lambda rule to confirm (before the tag).** As implemented (DECISIONS.md, "Selection rules, a priori"):
   - **(a) The 2% tie is measured against the lowest eps_hf.** Every candidate with eps_hf <= 1.02 x lowest ties, and the smallest tied lambda wins. The pairwise reading ("two lambdas within 2% of each other") can chain, e.g. 0.300 / 0.305 / 0.310, and give a different answer.
   - **(b) lambda = 0 is the reference, never a candidate.** Block C is "the best lambda > 0", and `make_block_c.py` refuses 0. If lambda = 0 has the lowest eps_hf, the rule still picks among the lambda > 0 within the MSE bound.
   - If no lambda > 0 meets the MSE bound, nothing is chosen and the decision comes back to you.
2. **Queue `fase1` at the maximum length.** At 150 epochs per run it takes 62.7 GPU-h; at 300 epochs, about 125 h. Today's 27 h slot covers roughly A seed 0 up to SRNO at 150 epochs. Relaunching the same command continues it. Recompute with `gpu_hours.py --measured results` after A seed 0.

## Resolved (user answers, 2026-10-01)

| # | Question | Decision | Done in |
|---|---|---|---|
| R1 | Budget cuts | No cuts in A/C; block D reduced to SRCNN, DSC/MS, SRNO; PRUSR/SoFoNO YAMLs moved to `configs/D_optional/` | eebd84b |
| R2 | Regime B composition | 5 MSE + 5 best lambda + lambda sweep on SRCNN (the best lambda is shared, so 13 configs); after the advisor | 6dfb2a1 (`make_block_n.py --sweep-arch srcnn`) |
| R3 | Stage 1 of E | Block A run (MSE, seed 0) of the best architecture of A | already the default of `make_block_e.py` |
| R4 | Minimum epochs of later stages | Same rule (min 50); the control uses the measured epochs | already implemented (4.7) |
| R5 | Legacy multistage files | Committed in `archive/legacy_multistage/` with a README (not in the campaign) | 133e3fa |
| R6 | k_c saturation | Two thresholds (eps <= 0.5 and <= 0.1) and a censoring marker; current columns kept | 290861f |
| R7 | Selection criterion | A priori rules in DECISIONS.md: architecture of E = lowest validation MSE (seed 0); lambda = lowest eps_hf among lambda > 0 within 10% of the validation MSE of lambda = 0, 2% ties -> smaller lambda; per family if SRCNN and SRNO differ; regime B reuses them; always on validation | this commit; `--split val`, `selecao_lambda_*.csv`, `--lambda-cnn/--lambda-op` (29250b9) |
| R8 | Versioning `results/` | Out of git; `scripts/backup_results.sh` (light zip + best.pt list) | 766219a |
| R9 | `--max-hours` | `run_queue.py --max-hours <h>` (e.g. 27), documented in RUNNING.md | 766219a |
| R10 | Block B order in `fase1` | SRCNN's 4 runs before SRNO's 4 (already so) | 4db8079 |

Also done on request: queues by priority (`queues/fase1-7.txt`, `scripts/run_queue.py`, 4db8079), `gpu_hours.py --measured` (bc0b5bc), worktree instructions (`RUNNING.md`). No tag was created and nothing was launched.

## Recorded, no decision needed unless you disagree

3. **Duplicate section in `DECISIONS.md`.** Commit b3dc0ad left two sections on the amplitude-aware metrics: "Amplitude-aware spectral metrics (k_c redefined)" and "Spectral metrics: amplitude, not only phase". The second still says the slope range is the proposed k in [12, 45]. The decided range is k = 11-46 (section "User decisions", `SLOPE_RANGE` in `collect_results.py`). By the add-only rule nothing was deleted; the later section is authoritative.
4. **Who re-ran the Wiener sweep at 14:09?** The tracked `figures/wiener_kc_vs_sigma.pdf` was regenerated, with only its embedded creation date changed, and that tripped the clean-tree guard of the first pilot. I restored the file and made the script save PDFs without dates (6e6a3d6), so a re-run no longer changes the tracked file. If it wasn't you, something else is running scripts in this checkout.
5. **Lost raw logs.** The machine rebooted on 2026-10-01 at 07:32, and the scratch space under /tmp was wiped along with the raw logs of the earlier pilot and memory checks. Their numbers are in `STATUS.md` and `DECISIONS.md`; the pilot was redone in `results_pilot/`.
6. **First pass of the new pilot partly refused.** While the pilot ran (tmux `pilot`), my uncommitted edits made the clean-tree guard refuse 7 of the 10 runs. The 3 that finished were DSC/MS with lambda 0 and 0.1, and PRUSR with 0.1. The 7 were re-run (tmux `pilot2`) from a detached git worktree at commit 6e87591, out of reach of edits in the main tree. The guard did what it should. I note it only because the pilot runs come from two commits, f649285 and 6e87591, which differ only by the 4.7 additions, unused by single-stage runs.
7. **A test tag was created and deleted at once.** While testing the worktree check of `run_queue.py`, I created a local tag `campanha-test` in a throwaway worktree. The test then failed because the script wasn't committed yet, and I deleted the tag immediately. The final test simulates the tag instead. `git tag -l` is empty.

## Changed instead of only added

8. `scripts/make_block_c.py`: `--sweep-arch` now takes one or more architectures and skips seed 0 of each. A single value works as before (14 configs); two give 13.
9. `train.py`: the optimizer receives only parameters with `requires_grad`. For single-stage runs, where every parameter is trainable, the list is identical; in multistage it excludes the frozen earlier stages. `evaluate_run.load_model` gained a branch for multistage runs; other runs take the old path.
10. `scripts/blocks.py:write_config` gained optional `run_name` and `extra` arguments. With the defaults it writes exactly the same files: blocks A and B regenerated byte-identical.
11. `scripts/gpu_hours.py`:
    - It follows the decided composition (D = 3 runs; regime B = 13 runs as in R2).
    - It no longer prints the proposed cuts, which were rejected.
    - It adds `--measured` and the per-queue table.
    - Block E counts stages 2-4 plus a control of stage 1 + stage 2 epochs. The earlier version counted stage 1 again and gave the control the length of all 4 stages, overestimating E by ~40%.
12. Multistage stage 2 of SRNO/SoFoNO: the old `train_multistage.py` built f_2 with the original models, which add a bilinear skip of x, so the old stage 2 predicted `x + g(x)` as the normalized residual. The new stages have no skip (`RESIDUAL: false`). Old and new stage-2 results are therefore not comparable for the operators.
13. `configs/D/`: the PRUSR and SoFoNO YAMLs were moved, not deleted, to `configs/D_optional/`.
14. Legacy multistage files moved, not deleted, from the repository root, `models/`, `configs/` and `notebooks/` to `archive/legacy_multistage/`, keeping their relative paths.
15. Figure titles in test mode: "Test error by radial band", "test set mean", "Test MSE"/"Test KE" became split-aware ("..., test set"), so the validation figures say "validation set". The data is unchanged; `report.md` and the CSV columns are unchanged, with `eps_hf` added at the end.
16. `results/baseline_wiener{,_noise0.01}/config.json` were rewritten by the first `evaluate_run.py --split val --baseline` call, with the same gamma and only the last digits of `val_mse_by_gamma` differing (GPU). The test files are unchanged (md5). Since then, `--split val` keeps an existing `config.json`.
17. `scripts/make_test_dates.py` gained `--split val` (writes `evaluation/val_dates.csv`); without it, the output is byte-identical to before.
