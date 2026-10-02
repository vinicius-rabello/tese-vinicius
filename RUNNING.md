# Running the campaign

The campaign runs from a **git worktree pinned at the tag `campanha-v1`**, while development continues in the main folder. Every run's `config.json` records the commit, so each result points at exact code. **None of these steps has been done yet: no tag, nothing launched.**

Paths below:

| Name | Path |
|---|---|
| Main folder (development) | `/home/lcadame/Vinicius_Thesis/Vinicius` |
| Campaign worktree (inside the main folder, git-ignored) | `/home/lcadame/Vinicius_Thesis/Vinicius/_run` |

## 1. Before tagging (main folder)

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius
git status --short | grep -v '^??'      # must print nothing (tracked files clean)
source venv/bin/activate
for t in tests/test_*.py; do python $t | tail -1; done   # all "N/N checks passed"
```

## 2. Tag and worktree (done on 2026-10-01: tag `campanha-v1` = b25f1f3, worktree `_run/`)

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius
git tag -a campanha-v1 -m "Campaign code, $(date +%F)"
git worktree add --detach _run campanha-v1
cd _run
ln -s ../hycom_numpy hycom_numpy   # data (read only)
ln -s ../venv venv                 # same Python environment
ln -s ../results results           # results land in the main folder's results/
mkdir -p logs                      # queue logs (git-ignored: /logs/)
```

`_run/` sits inside the main folder but is listed in the main folder's `.gitignore` (`/_run/`), so `git status` and editor searches skip it. Never edit files under `_run/`: it is the frozen campaign code; development happens in the main folder. The `RUNNING.md` inside `_run/` is the tagged copy and still names the first location `../Vinicius_campanha`; this file is the current one.

The symlinks are untracked, so the clean-tree guard of `train.py` still passes. `results/` already holds the baselines, and `collect_results.py` / `stats.py` run from the main folder on it. Development in the main folder must not write to `results/`: use `results_pilot/` or `RESULTS_DIR`.

`scripts/run_queue.py` refuses to start anywhere other than a linked worktree at a `campanha-*` tag, so a queue launched by mistake from the main folder does nothing.

## 3. Launching a queue (GNU screen)

Run one queue at a time, inside a `screen` session named `campanha`. Set `--max-hours` to the time available: 70 for a 72 h lab block (2 h of margin), 27 for a 27 h window.

| Action | Command |
|---|---|
| create the session | `screen -S campanha` |
| leave it running (detach) | Ctrl-a d |
| come back | `screen -r campanha` |
| list sessions | `screen -ls` |

Inside the session:

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius/_run
source venv/bin/activate
TQDM_DISABLE=1 python scripts/run_queue.py queues/fase1.txt --max-hours 27 2>&1 | tee -a logs/fase1.log
```

- **Budget:** `--max-hours <h>` is passed to `train.py`, which stops cleanly before an epoch that would cross it.
  - The check happens after each epoch, so a run that starts right before the deadline can overrun by up to one epoch plus data loading (at most ~5 min, SoFoNO).
  - The test-set evaluation of the finished runs comes after the budget (~20 s per run in the pilot).
  - Leave ~15 min of margin. Omitting the option means 70.
  - To stop at a fixed clock time, compute the hours left: `python3 -c "from datetime import datetime as d; print(round((d.fromisoformat('2026-10-02T16:30') - d.now()).total_seconds() / 3600, 2))"`.
- **`TQDM_DISABLE=1`** turns off the per-batch progress bars, which are useless in a file: they grew `fase1.log` to 23 MB in one day. The log then keeps one summary line per epoch (losses, validation MSE, lr, early-stopping counter) plus the run and queue messages. It needs no code change and no new tag. Leave it out when you are watching interactively and want the bar.
- **Log:** `/home/lcadame/Vinicius_Thesis/Vinicius/_run/logs/fase1.log` (`tee -a` appends on every relaunch). Follow it with `tail -f` from any terminal.
- **Next block or after a reboot:** open a new `screen -S campanha` and run the same three commands. Finished runs are skipped, an interrupted run resumes from its `last.pt` (saved atomically after every epoch, so at most the epoch in progress is lost), and finished runs that have not been evaluated get their test-set evaluation at the end.
- **Stopping early:** `screen -r campanha`, then Ctrl-c once. The run in progress keeps the `last.pt` of its last finished epoch; relaunching resumes from there.
- One queue at a time on the GPU. SRNO and SoFoNO use ~13 GB of the 16 GB, so no other GPU job during the campaign, including development tests in the main folder.

## 4. Order and decisions between queues

| Queue | Content | Before launching |
|---|---|---|
| `fase1` | A seed 0 (SRCNN, DSC/MS, PRUSR, SRNO, SoFoNO), then B | none |
| (after A seed 0) | `python scripts/gpu_hours.py --measured results` (main folder) recomputes the table with the measured epochs | |
| `fase2` | C seed 0 (DSC/MS, PRUSR, SoFoNO) | choose lambda from B with the a priori rule (section 4.1); in the worktree: the `make_block_c.py` command it prints (`--lambda <v>` or `--lambda-cnn <v> --lambda-op <v>`, plus `--sweep-arch srno srcnn`) |
| `fase3` | D reduced (SRCNN, DSC/MS, SRNO) | none |
| `fase4` | A seeds 1-2 | none |
| `fase5` | C seeds 1-2 | (configs from `fase2`) |
| `fase6` | E: stages 2-4 + control | best architecture of A = lowest validation MSE of seed 0 (section 4.1); in the worktree: `python scripts/make_block_e.py --arch <arch>` (stage 1 = `A_<arch>_res_lam0_seed0`) |
| `fase7` | regime B, sigma = 1e-2 (13 runs) | the advisor's approval; in the worktree: `python scripts/make_block_n.py --noise 0.01 --lambda <best> --sweep-arch srcnn` |

Generate configs **in the worktree**, because the queue reads them there. Then run the same command in the main folder and commit, so the configs are also versioned. The generators are deterministic, so the files are identical. A queue whose configs don't exist yet refuses to start and prints the command to run. `python scripts/run_queue.py queues/<fase>.txt --dry-run` lists what a queue will train, and works anywhere.

### 4.1 Selections (validation set only; rules fixed in `DECISIONS.md`, "Selection rules, a priori")

Run these in the main folder, with no queue running, because they evaluate on the GPU:

```bash
python collect_results.py --split val --runs "A_*_seed0" "B_*"
```

- **lambda:** the output prints the rule for the SRCNN and SRNO sweeps and the resulting `make_block_c.py` arguments. The full tables are `report/<date>_val/selecao_lambda_srcnn.csv` and `selecao_lambda_srno.csv`: validation MSE, the ratio to lambda = 0, eps_hf, candidates, ties and the choice. Check them before generating C.
- **Architecture for E:** in `report/<date>_val/all_runs_val.csv`, take the `A_*_seed0` row with the lowest `mse`.
- **Regime B:** reuses the same lambda(s) (`make_block_n.py --lambda ...` or `--lambda-cnn ... --lambda-op ...`).

## 5. Results (main folder)

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius && source venv/bin/activate
python collect_results.py              # report/<date>/: all_runs.csv, figures, report.md (evaluates missing runs)
python stats.py                        # tables/: main_results, comparisons, bands, lambda_sweep, ablation_skip, multistage
python scripts/gpu_hours.py --measured results
```

`collect_results.py` evaluates on the GPU if a finished run has no metrics. While a queue is running, use `--no-eval`.

**Backup** (`results/` stays out of git): `scripts/backup_results.sh` writes `/home/lcadame/Vinicius_Thesis/Vinicius/backups/results_light_<date>.zip`. The zip holds, per run, config, summary, train_log, eval summaries and per-day metrics, plus `tables/`, the newest test and validation reports, and DECISIONS/STATUS/OPEN_QUESTIONS; no `.pt`, no `.npz`. Next to the zip, `best_pt_<date>.txt` lists the `best.pt` of the main configurations (A, B, C, E) with size and sha256, then the others. Copy those `best.pt` files separately.

## 6. If the campaign code needs a fix

Never edit the worktree. Fix and commit in the main folder, tag `campanha-v1.1`, and move the worktree **between queues** (never while a run is going):

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius/_run && git checkout --detach campanha-v1.1
```

Runs already finished keep `campanha-v1` in their `config.json`. A run interrupted mid-way records each resume's commit under `resumes`. Record the fix and which runs it affects in `DECISIONS.md`.
