"""Resume test through train.py (tiny subset, GPU if available). Run from the repo root:

    python tests/test_resume.py

For each determinism mode, one run trains straight through and another is
stopped by --max-hours after every epoch and relaunched until it finishes.
DETERMINISTIC: true must give bit-identical logs; the default
(cudnn.benchmark, not bit-reproducible) must agree within a relative
difference < 1e-3 in validation MSE at every epoch.
"""

import csv
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL_TOL = 1e-3
COLS = ('epoch', 'lr', 'train_mse', 'val_mse', 'es_best', 'es_counter', 'best_val_mse')


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def write_config(folder, name, deterministic):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = os.path.join(folder, f"{name}.yaml")
    with open(path, 'w') as f:
        f.write(f"""MODEL_NAME: "srcnn"
RUN_NAME: "{name}"
RESULTS_DIR: "{folder}/results"
DEVICE: "{device}"
NUM_WORKERS: 2
BATCH_SIZE: 16
LEARNING_RATE: 0.0001
LAMBDA: 0
RESIDUAL: true
SEED: 0
IMAGE_SAVE_INTERVAL: 1000
SUBSET_TRAIN: 64
SUBSET_VAL: 32
MIN_EPOCHS: 3
ES_PATIENCE: 2
ES_REL_DELTA: 0.3
MAX_EPOCHS: 8
DETERMINISTIC: {'true' if deterministic else 'false'}
""")
    return path


def train(config, *extra):
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'train.py'), '--config', config, *extra],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return out.stdout


def log(folder, name):
    with open(os.path.join(folder, 'results', f"{name}_res_lam0_seed0", 'train_log.csv')) as f:
        return [{c: r[c] for c in COLS} for r in csv.DictReader(f)]


def main():
    results = []
    with tempfile.TemporaryDirectory() as folder:
        for deterministic in (True, False):
            mode = 'deterministic' if deterministic else 'default'
            straight = write_config(folder, f"straight_{mode}", deterministic)
            resumed = write_config(folder, f"resumed_{mode}", deterministic)
            train(straight)
            launches = 0
            while launches < 20:
                launches += 1
                if 'completed_' in train(resumed, '--max-hours', '0.00025').split('Queue stopped')[0]:
                    break
            a, b = log(folder, f"straight_{mode}"), log(folder, f"resumed_{mode}")
            if deterministic:
                results.append(check(f"{mode}: resumed after every epoch ({launches} launches) is bit-identical",
                                     a == b, f"({len(a)} vs {len(b)} epochs)"))
            else:
                n = min(len(a), len(b))
                rel = max(abs(float(x['val_mse']) - float(y['val_mse'])) / float(x['val_mse'])
                          for x, y in zip(a[:n], b[:n]))
                results.append(check(f"{mode}: resumed after every epoch ({launches} launches) matches within "
                                     f"relative {REL_TOL:g} in val MSE", len(a) == len(b) and rel < REL_TOL,
                                     f"(max relative diff {rel:.2e}; {len(a)} vs {len(b)} epochs)"))
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
