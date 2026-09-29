"""Cost of the determinism settings on one architecture (section 6, item 3).

Runs one full training epoch plus validation under one cuDNN/determinism
setting and appends a row to tables/determinism_benchmark.csv. Each setting
must run in its own process (flags must not leak between them):

    for s in 1 2 3 4; do python scripts/determinism_benchmark.py --arch prusr --setting $s; done

Settings (cudnn.deterministic, cudnn.benchmark, use_deterministic_algorithms):
    1  False  True   off   fast reference
    2  True   False  off   cudnn.deterministic only
    3  True   False  on    current decision (warn_only)
    4  False  False  off   PyTorch defaults (what the old runs used)
"""

import argparse
import csv
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from losses.losses import MseLpipsLoss  # noqa: E402
from models.model_factory import create_model  # noqa: E402
from train import dataset_constants, train_one_epoch, validate  # noqa: E402
from utils import reproducibility as repro  # noqa: E402

SETTINGS = {
    1: (False, True, False),
    2: (True, False, False),
    3: (True, False, True),
    4: (False, False, False),
}
ARCHS = {"srcnn": (32, 1e-4), "dscms": (32, 1e-4), "prusr": (32, 1e-4), "srno": (16, 4e-5), "sofono": (8, 4e-5)}
SEED, DEVICE, OUT = 0, "cuda", "tables/determinism_benchmark.csv"


def validate_mse_only(model, loader, criterion):
    """Validation as the old MSE runs did it: no LPIPS."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            _, m = criterion(model(lr), hr)
            total += m['mse'] * lr.shape[0]
            n += lr.shape[0]
    return total / n


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='prusr', choices=ARCHS)
    parser.add_argument('--setting', type=int, required=True, choices=SETTINGS)
    args = parser.parse_args()
    deterministic, benchmark, algorithms = SETTINGS[args.setting]
    batch_size, lr = ARCHS[args.arch]

    # utils.reproducibility sets CUBLAS_WORKSPACE_CONFIG on import; it only
    # matters for deterministic algorithms and restricts cuBLAS's workspace, so
    # drop it for the other settings (cuBLAS reads it at its first use, below).
    if not algorithms:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(algorithms, warn_only=True)

    train_ds = SuperResDataset([f"./hycom_numpy/hycom_{y}.npy" for y in range(2001, 2020)], 1, 10)
    val_ds = SuperResDataset(["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"], 1, 10)
    scale = dataset_constants(train_ds.hr_data)['lpips_scale']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              worker_init_fn=repro.seed_worker, generator=repro.make_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model(args.arch, residual=True).to(DEVICE)
    criterion = MseLpipsLoss(0.0, scale, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    config = {'RUN_DIR': tempfile.mkdtemp(), 'IMAGE_SAVE_INTERVAL': 10**9, 'LAMBDA': 0.0}

    _, train_sec = timed(lambda: train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, 1, config,
                                                 args.arch.upper()))
    val_mse, val_sec = timed(lambda: validate_mse_only(model, val_loader, criterion))
    criterion.lpips_model  # build VGG outside the timed region
    val_lpips, val_lpips_sec = timed(lambda: validate(model, val_loader, criterion, DEVICE))

    row = {
        'arch': args.arch, 'setting': args.setting, 'cudnn_deterministic': deterministic,
        'cudnn_benchmark': benchmark, 'deterministic_algorithms': algorithms,
        'train_epoch_sec': round(train_sec, 1), 'val_sec': round(val_sec, 1),
        'val_with_lpips_sec': round(val_lpips_sec, 1),
        'epoch_sec_with_val': round(train_sec + val_sec, 1),
        'epoch_sec_with_val_lpips': round(train_sec + val_lpips_sec, 1),
        'val_mse_after_epoch': val_mse, 'val_lpips_after_epoch': val_lpips['lpips'],
        'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new_file = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
