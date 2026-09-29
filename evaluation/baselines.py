"""Reference baselines: identity (the LR input itself) and Wiener/Tikhonov
deconvolution of the known Gaussian degradation, Yhat = H / (H^2 + gamma) X."""

import numpy as np
import torch

from evaluation.spectral_metrics import D0, gaussian_transfer, mse


def identity(lr):
    return lr


def wiener(lr, gamma: float, d0: float = D0):
    h = gaussian_transfer(lr.shape[-1], d0, lr.device)
    return torch.fft.ifft2(torch.fft.fft2(lr) * (h / (h ** 2 + gamma))).real


def choose_gamma(loader, gammas=None, device='cpu'):
    """Picks gamma by the lowest validation MSE over a log-spaced grid.
    Returns (best_gamma, {gamma: val_mse})."""
    # The degradation is noiseless, so the optimum sits where only float32
    # round-off of the LR input limits the inverse (about 1e-12); the grid
    # must reach far below that for the minimum to be interior.
    gammas = np.logspace(-16, 0, 33) if gammas is None else gammas
    sums = {float(g): 0.0 for g in gammas}
    n = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            for g in sums:
                sums[g] += mse(wiener(lr, g), hr).sum().item()
            n += lr.shape[0]
    val = {g: s / n for g, s in sums.items()}
    best = min(val, key=val.get)
    ordered = sorted(val)
    if best in (ordered[0], ordered[-1]):
        raise RuntimeError(f"Best gamma {best:g} is at the edge of the grid; extend the grid.")
    return best, val
