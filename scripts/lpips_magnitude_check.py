"""One-epoch magnitude check of the two loss terms, to set the lambda grid.

For each architecture (residual=true, lambda=0, full training set) it measures
the validation MSE and LPIPS before and after one training epoch, and reports
LPIPS/MSE and lambda* = MSE/LPIPS, the lambda at which both terms of
MSE + lambda*LPIPS are equal. Run from the repo root:

    python scripts/lpips_magnitude_check.py
"""

import csv
import os
import sys
import tempfile
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from losses.losses import MseLpipsLoss  # noqa: E402
from models.model_factory import create_model  # noqa: E402
from train import dataset_constants, train_one_epoch, validate  # noqa: E402
from utils import reproducibility as repro  # noqa: E402

# Batch size and learning rate as used so far for each architecture.
ARCHS = {
    "srcnn": (32, 1e-4),
    "dscms": (32, 1e-4),
    "prusr": (32, 1e-4),
    "srno": (16, 4e-5),
    "sofono": (8, 4e-5),
}
SEED, DEVICE, OUT = 0, "cuda", "tables/lpips_magnitude.csv"


def main():
    train_ds = SuperResDataset([f"./hycom_numpy/hycom_{y}.npy" for y in range(2001, 2020)], 1, 10)
    val_ds = SuperResDataset(["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"], 1, 10)
    scale = dataset_constants(train_ds.hr_data)['lpips_scale']
    rows = []

    for name, (batch_size, lr) in ARCHS.items():
        repro.set_seed(SEED)
        gen = repro.make_generator(SEED)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  worker_init_fn=repro.seed_worker, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = create_model(name, residual=True).to(DEVICE)
        criterion = MseLpipsLoss(0.0, scale, DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        config = {'RUN_DIR': tempfile.mkdtemp(), 'IMAGE_SAVE_INTERVAL': 10**9, 'LAMBDA': 0.0}

        start = validate(model, val_loader, criterion, DEVICE)
        t0 = time.perf_counter()
        train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, 1, config, name.upper())
        epoch_sec = time.perf_counter() - t0
        end = validate(model, val_loader, criterion, DEVICE)

        for when, m in (("start", start), ("end", end)):
            rows.append({
                'arch': name, 'when': when, 'val_mse': m['mse'], 'val_lpips': m['lpips'],
                'lpips_over_mse': m['lpips'] / m['mse'], 'lambda_star': m['mse'] / m['lpips'],
                'epoch_sec': round(epoch_sec, 1), 'batch_size': batch_size, 'lr': lr,
                'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
            })
        print(f"{name}: start mse {start['mse']:.4g} lpips {start['lpips']:.4g} | "
              f"end mse {end['mse']:.4g} lpips {end['lpips']:.4g} | epoch {epoch_sec:.0f}s", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'arch':8s} {'when':6s} {'val MSE':>10s} {'val LPIPS':>10s} {'LPIPS/MSE':>10s} {'lambda*':>10s}")
    for r in rows:
        print(f"{r['arch']:8s} {r['when']:6s} {r['val_mse']:10.4g} {r['val_lpips']:10.4g} "
              f"{r['lpips_over_mse']:10.4g} {r['lambda_star']:10.4g}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
