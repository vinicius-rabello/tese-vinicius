# Running the campaign

The campaign runs from a **git worktree pinned at the tag `campanha-v1`**, while development continues in the main folder. Every run's `config.json` records the commit, so each result points at exact code. **None of these steps has been done yet: no tag, nothing launched.**

Paths below:

| Name | Path |
|---|---|
| Main folder (development) | `/home/lcadame/Vinicius_Thesis/Vinicius` |
| Campaign worktree | `/home/lcadame/Vinicius_Thesis/Vinicius_campanha` |

## 1. Before tagging (main folder)

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius
git status --short | grep -v '^??'      # must print nothing (tracked files clean)
source venv/bin/activate
for t in tests/test_*.py; do python $t | tail -1; done   # all "N/N checks passed"
```

## 2. Tag and worktree (only after your go-ahead)

```bash
git tag -a campanha-v1 -m "Campaign code, $(date +%F)"
git worktree add --detach ../Vinicius_campanha campanha-v1
cd ../Vinicius_campanha
ln -s ../Vinicius/hycom_numpy hycom_numpy   # data (read only)
ln -s ../Vinicius/venv venv                 # same Python environment
ln -s ../Vinicius/results results           # results land in the main folder's results/
mkdir -p logs                               # queue logs (git-ignored: /logs/)
```

The symlinks are untracked, so the clean-tree guard of `train.py` still passes. `results/` already holds the baselines, and `collect_results.py` / `stats.py` run from the main folder on it. Development in the main folder must not write to `results/`: use `results_pilot/` or `RESULTS_DIR`.

`scripts/run_queue.py` refuses to start anywhere other than a linked worktree at a `campanha-*` tag, so a queue launched by mistake from the main folder does nothing.

## 3. Launching a queue

One tmux session per queue, `--max-hours 70` (a 72 h lab block with 2 h of margin):

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius_campanha
tmux new -d -s fase1 "source venv/bin/activate && python scripts/run_queue.py queues/fase1.txt --max-hours 70 2>&1 | tee -a logs/fase1.log"
```

- Session: `fase1`. Log: `/home/lcadame/Vinicius_Thesis/Vinicius_campanha/logs/fase1.log`. Follow with `tail -f`, or `tmux attach -t fase1` (detach: Ctrl-b d).
- **Next 72 h block:** relaunch the same command. Finished runs are skipped, an interrupted run resumes from its `last.pt`, and finished runs that have not been evaluated get their test-set evaluation at the end.
- One queue at a time on the GPU. SRNO and SoFoNO use ~13 GB of the 16 GB, so no other GPU job during the campaign, including development tests in the main folder.

## 4. Order and decisions between queues

| Queue | Content | Before launching |
|---|---|---|
| `fase1` | A seed 0 (SRCNN, DSC/MS, PRUSR, SRNO, SoFoNO), then B | none |
| (after A seed 0) | `python scripts/gpu_hours.py --measured results` (main folder) recomputes the table with the measured epochs | |
| `fase2` | C seed 0 (DSC/MS, PRUSR, SoFoNO) | choose lambda from B; in the worktree: `python scripts/make_block_c.py --lambda <best> --sweep-arch srno srcnn` |
| `fase3` | D reduced (SRCNN, DSC/MS, SRNO) | none |
| `fase4` | A seeds 1-2 | none |
| `fase5` | C seeds 1-2 | (configs from `fase2`) |
| `fase6` | E: stages 2-4 + control | choose the best architecture of A; in the worktree: `python scripts/make_block_e.py --arch <arch>` (stage 1 = `A_<arch>_res_lam0_seed0`) |
| `fase7` | regime B, sigma = 1e-2 (13 runs) | the advisor's approval; in the worktree: `python scripts/make_block_n.py --noise 0.01 --lambda <best> --sweep-arch srcnn` |

Generate configs **in the worktree**, because the queue reads them there. Then run the same command in the main folder and commit, so the configs are also versioned. The generators are deterministic, so the files are identical. A queue whose configs don't exist yet refuses to start and prints the command to run. `python scripts/run_queue.py queues/<fase>.txt --dry-run` lists what a queue will train, and works anywhere.

## 5. Results (main folder)

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius && source venv/bin/activate
python collect_results.py              # report/<date>/: all_runs.csv, figures, report.md (evaluates missing runs)
python stats.py                        # tables/: main_results, comparisons, bands, lambda_sweep, ablation_skip, multistage
python scripts/gpu_hours.py --measured results
```

`collect_results.py` evaluates on the GPU if a finished run has no metrics. While a queue is running, use `--no-eval`.

## 6. If the campaign code needs a fix

Never edit the worktree. Fix and commit in the main folder, tag `campanha-v1.1`, and move the worktree **between queues** (never while a run is going):

```bash
cd /home/lcadame/Vinicius_Thesis/Vinicius_campanha && git checkout --detach campanha-v1.1
```

Runs already finished keep `campanha-v1` in their `config.json`. A run interrupted mid-way records each resume's commit under `resumes`. Record the fix and which runs it affects in `DECISIONS.md`.
