"""Runs one campaign queue (queues/<name>.txt) through train.py, then
evaluates its finished runs on the test set.

    python scripts/run_queue.py queues/fase1.txt [--max-hours 70] [--dry-run] [--no-eval]

A queue file lists configs in priority order, one per line: a path or a glob
pattern (for configs generated later: C, E, regime B); blank lines and
#-comments are ignored, and a config already listed is not repeated. Every
line must match at least one file, otherwise the queue refuses to start and
names the generator to run first.

Training is one `train.py --config <all configs> --max-hours <h>` call, so
finished runs are skipped, an interrupted run resumes from last.pt, and the
budget stops the queue cleanly; relaunch the same command to continue. After
training, runs that finished and have no metrics_per_sample.csv are
evaluated (evaluate_run.py), so a relaunch also evaluates what is left.

The campaign runs from a linked git worktree at a tag campanha-* (RUNNING.md);
anywhere else (including the main folder) the queue refuses to start, except with --dry-run or
QUEUE_ALLOW_UNTAGGED=1 (tests only).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GENERATORS = {'configs/C/': 'scripts/make_block_c.py --lambda <best> --sweep-arch srno srcnn',
              'configs/E/': 'scripts/make_block_e.py --arch <best arch of block A>',
              'configs/N/': 'scripts/make_block_n.py --noise 0.01 --lambda <best> --sweep-arch srcnn'}


def resolve(queue_file):
    """Config paths in queue order, without repeats; error on an empty match."""
    paths, missing = [], []
    with open(queue_file) as f:
        for line in f:
            entry = line.split('#', 1)[0].strip()
            if not entry:
                continue
            matches = sorted(glob.glob(os.path.join(ROOT, entry)))
            if not matches:
                hint = next((g for prefix, g in GENERATORS.items() if entry.startswith(prefix)), None)
                missing.append(entry + (f"  (generate with: python {hint})" if hint else ""))
            for m in matches:
                rel = os.path.relpath(m, ROOT)
                if rel not in paths:
                    paths.append(rel)
    return paths, missing


def campaign_tag():
    """The campanha-* tag of HEAD, only inside a linked git worktree (right
    after tagging, the main folder is also at the tag, but development goes on
    there)."""
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    linked = os.path.realpath(os.path.join(ROOT, git('rev-parse', '--git-dir'))) != \
        os.path.realpath(os.path.join(ROOT, git('rev-parse', '--git-common-dir')))
    tag = git('describe', '--exact-match', '--tags', 'HEAD')
    return tag if linked and tag.startswith('campanha-') else None


def run_dir_of(config_path):
    from train import run_id_for
    with open(os.path.join(ROOT, config_path)) as f:
        config = yaml.safe_load(f)
    return os.path.join(config.get('RESULTS_DIR', 'results'), run_id_for(config, config_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('queue')
    parser.add_argument('--max-hours', type=float, default=70)
    parser.add_argument('--dry-run', action='store_true', help="only resolve and print the queue")
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    paths, missing = resolve(args.queue)
    print(f"Queue {args.queue}: {len(paths)} config(s)")
    for p in paths:
        print(f"  {p}  ->  {run_dir_of(p)}")
    if missing:
        print("Lines without any config:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    tag = campaign_tag()
    print(f"Checkout: {tag or 'not a worktree at a campanha-* tag'}")
    if args.dry_run:
        return
    if tag is None and os.environ.get('QUEUE_ALLOW_UNTAGGED') != '1':
        sys.exit("Refusing to run: the campaign runs from the campanha-* worktree (see RUNNING.md).")

    train = subprocess.run([sys.executable, 'train.py', '--config', *paths, '--max-hours', str(args.max_hours)],
                           cwd=ROOT)
    if train.returncode != 0:
        sys.exit(train.returncode)
    if args.no_eval:
        return

    to_eval = []
    for p in paths:
        run = run_dir_of(p)
        summary = os.path.join(ROOT, run, 'summary.json')
        if not os.path.exists(summary):
            continue
        with open(summary) as f:
            status = json.load(f).get('status')
        if status in ('completed_early_stop', 'completed_max_epochs') and \
                not os.path.exists(os.path.join(ROOT, run, 'metrics_per_sample.csv')):
            to_eval.append(run)
    if to_eval:
        print(f"Evaluating {len(to_eval)} finished run(s) on the test set ...")
        sys.exit(subprocess.run([sys.executable, 'evaluate_run.py', *to_eval], cwd=ROOT).returncode)
    print("Nothing to evaluate.")


if __name__ == "__main__":
    main()
