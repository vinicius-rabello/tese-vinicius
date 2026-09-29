"""Checks for the MSE + lambda * LPIPS loss (stage 4.4). Run from the repo root:

    python tests/test_loss.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import lpips_utils  # noqa: E402
from losses.losses import FluidCompositeLoss, MseLpipsLoss  # noqa: E402
from models.model_factory import create_model  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LPIPS_SCALE = 2.869149  # max(|u|, |v|) over the 2001-2019 HR fields (config.json: lpips_scale)


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def loss_and_grads(criterion, weights, x, y):
    model = create_model("srcnn", residual=True).to(DEVICE)
    model.load_state_dict(weights)
    loss, _ = criterion(model(x), y)
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters()]


def main():
    set_seed(0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 2, 128, 128, generator=g).to(DEVICE)
    y = torch.randn(4, 2, 128, 128, generator=g).to(DEVICE)
    weights = create_model("srcnn", residual=True).state_dict()
    results = []

    # 1. lambda = 0 is exactly the old MSE loss: same value, same gradients, no VGG
    new = MseLpipsLoss(0.0, LPIPS_SCALE, DEVICE)
    old = FluidCompositeLoss(mode=1, weights={'mse': 1.0, 'lpips': 0.0, 'kinetic': 0.0}, device=DEVICE)
    new_loss, new_grads = loss_and_grads(new, weights, x, y)
    old_loss, old_grads = loss_and_grads(old, weights, x, y)
    results.append(check("lambda=0: loss bit-identical to the old MSE loss", torch.equal(new_loss, old_loss)))
    results.append(check("lambda=0: every parameter gradient bit-identical",
                         all(torch.equal(a, b) for a, b in zip(new_grads, old_grads))))
    results.append(check("lambda=0: the LPIPS network is never built while training", new._lpips_model is None))

    # 2. lambda > 0: loss = MSE + lambda * shared lpips_distance, through the shared preprocessing
    lam = 0.1
    criterion = MseLpipsLoss(lam, LPIPS_SCALE, DEVICE)
    pred = x * 0.5
    calls = {'n': 0}
    original = lpips_utils.to_lpips_input

    def counting(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    lpips_utils.to_lpips_input = counting
    try:
        total, metrics = criterion(pred, y)
    finally:
        lpips_utils.to_lpips_input = original
    results.append(check("lambda>0: the loss goes through lpips_utils.to_lpips_input (u and v, pred and target)",
                         calls['n'] == 4, f"(calls: {calls['n']})"))

    shared = lpips_utils.lpips_distance(criterion.lpips_model, pred, y, LPIPS_SCALE)
    mse = torch.nn.functional.mse_loss(pred, y)
    results.append(check("lambda>0: LPIPS term equals the shared lpips_distance (the evaluation path)",
                         metrics['lpips'] == shared.item()))
    results.append(check("lambda>0: loss == MSE + lambda * LPIPS", torch.equal(total, mse + lam * shared)))

    # validation logs LPIPS even with lambda = 0, without changing the loss
    total0, metrics0 = MseLpipsLoss(0.0, LPIPS_SCALE, DEVICE)(pred, y, compute_lpips=True)
    results.append(check("lambda=0 with compute_lpips: LPIPS logged, loss still plain MSE",
                         torch.equal(total0, mse) and metrics0['lpips'] == shared.item()))

    # 3. the fixed scale maps the training fields into [-1, 1]
    train_max = max(np.abs(np.load(f"hycom_numpy/hycom_{year}.npy")).max() for year in range(2001, 2020))
    extreme = torch.full((1, 2, 4, 4), float(train_max))
    scaled = lpips_utils.to_lpips_input(extreme, 0, LPIPS_SCALE)
    results.append(check("scale: training extremes map into [-1, 1], channel repeated 3x",
                         scaled.shape[1] == 3 and scaled.abs().max().item() <= 1.0 + 1e-6,
                         f"(max |u|,|v| = {train_max:.6f}, scaled max = {scaled.abs().max().item():.6f})"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
