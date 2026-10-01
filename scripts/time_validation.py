"""Time of one validation pass per pilot run, so the epoch time can be split
into training and validation (train_log.csv's epoch_sec includes both).

    python scripts/time_validation.py <run dir> [<run dir> ...] [--out results_pilot/validation_time.csv]

Runs train.validate exactly as train.py does (validation set, the run's batch
size, NUM_WORKERS and pin_memory, LPIPS always computed), once to warm up and
once timed, with the run's determinism setting. Train-only seconds per epoch
= epoch_sec of the last logged epoch - the timed validation pass.
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.super_res_dataset import NOISE_SEED_OFFSET, SuperResDataset  # noqa: E402
from evaluate_run import load_model  # noqa: E402
from losses.losses import MseLpipsLoss  # noqa: E402
from train import validate  # noqa: E402
from utils import reproducibility as repro  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', default='results_pilot/validation_time.csv')
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        with open(os.path.join(run, 'config.json')) as f:
            meta = json.load(f)
        cfg = meta['config']
        repro.set_seed(0, meta.get('deterministic', False), meta.get('deterministic_algorithms', True))
        ds = SuperResDataset(hr_files=["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"],
                             downsample_factor=1, freq_threshold=10, noise_std=meta.get('noise_std', 0.0),
                             noise_mode='fixed', noise_seed_offset=NOISE_SEED_OFFSET['val'])
        loader = DataLoader(ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=cfg['NUM_WORKERS'],
                            pin_memory=True)
        model, _ = load_model(run, 'cuda')
        criterion = MseLpipsLoss(meta['lambda'], meta['lpips_scale'], device='cuda', net='vgg')
        validate(model, loader, criterion, 'cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        validate(model, loader, criterion, 'cuda')
        torch.cuda.synchronize()
        val_sec = time.perf_counter() - start
        with open(os.path.join(run, 'train_log.csv')) as f:
            epoch_sec = float(list(csv.DictReader(f))[-1]['epoch_sec'])
        rows.append({'run': os.path.basename(run), 'model': meta['model'], 'lambda': meta['lambda'],
                     'deterministic': meta.get('deterministic', False), 'epoch_sec_with_val': epoch_sec,
                     'val_sec': round(val_sec, 2), 'epoch_sec_train_only': round(epoch_sec - val_sec, 2)})
        print(rows[-1], flush=True)
        del model, criterion
        torch.cuda.empty_cache()

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
