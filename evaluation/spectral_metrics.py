"""Evaluation metrics on velocity fields in m/s (stage 4.5).

Every function takes (B, 2, N, N) tensors (u, v) and returns one value per
sample, so results can be stored per test day and bootstrapped. Radial
wavenumbers follow the existing KE convention: k = |(kx, ky)| in cycles per
domain (integer FFT frequencies), shell k = [k, k + 1).
"""

import math

import torch
import torch.nn.functional as F

from metrics.kinetic_energy_spectrum import KineticEnergySpectrumLoss, WeightedKineticEnergySpectrumLoss

D0 = 10.0
# Radial bands for the per-band error, edges in k (cycles per domain); the
# last band runs to the corners (k up to 64*sqrt(2)), so the bands cover every
# frequency and add up to the total MSE. Bands 3-5 are k >= D0.
BAND_EDGES = (0, 5, 10, 20, 40, math.inf)
BAND_NAMES = ('k0_5', 'k5_10', 'k10_20', 'k20_40', 'k40_max')


def radius_grid(n: int, device=None) -> torch.Tensor:
    k = torch.fft.fftfreq(n, device=device) * n
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    return torch.sqrt(kx ** 2 + ky ** 2)


def gaussian_transfer(n: int, d0: float = D0, device=None) -> torch.Tensor:
    """H(k) = exp(-k^2 / (2 d0^2)), the degradation applied to make the LR input."""
    return torch.exp(-radius_grid(n, device) ** 2 / (2 * d0 ** 2))


# ---------------------------------------------------------------- pointwise --
def mse(pred, target):
    return ((pred - target) ** 2).flatten(1).mean(1)


def psnr(pred, target, data_range: float):
    """10 log10(R^2 / MSE) with one fixed R for every model, sample and run."""
    return 10 * torch.log10(data_range ** 2 / mse(pred, target))


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=torch.float32):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred, target, data_range: float, offset: float):
    """SSIM per sample, mean over u and v, standard constants (K1 = 0.01,
    K2 = 0.03, 11x11 Gaussian window, sigma 1.5, no padding -- the same as
    piq.ssim). Fields are shifted by one fixed offset (the training minimum)
    so the luminance term sees the same mapping for every model and sample;
    no per-sample rescaling, and values outside [0, R] are not clipped."""
    x, y = pred - offset, target - offset
    b, c, h, w = x.shape
    window = _gaussian_window(device=x.device, dtype=x.dtype).expand(c, 1, -1, -1)
    conv = lambda t: F.conv2d(t, window, groups=c)  # noqa: E731
    mu_x, mu_y = conv(x), conv(y)
    sxx = conv(x * x) - mu_x ** 2
    syy = conv(y * y) - mu_y ** 2
    sxy = conv(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return ssim_map.flatten(2).mean(2).mean(1)


# ----------------------------------------------------------------- spectral --
_KE, _WKE = {}, {}


def ke(pred, target):
    """Existing KineticEnergySpectrumLoss, one value per sample."""
    n = pred.shape[-1]
    if n not in _KE:
        _KE[n] = KineticEnergySpectrumLoss(size=n)
    fn = _KE[n].to(pred.device)
    return torch.stack([fn(pred[i:i + 1], target[i:i + 1]) for i in range(pred.shape[0])])


def wke(pred, target):
    """Existing WeightedKineticEnergySpectrumLoss, one value per sample."""
    n = pred.shape[-1]
    if n not in _WKE:
        _WKE[n] = WeightedKineticEnergySpectrumLoss(size=n)
    fn = _WKE[n].to(pred.device)
    return torch.stack([fn(pred[i:i + 1], target[i:i + 1]) for i in range(pred.shape[0])])


def radial_spectrum(field):
    """Radial kinetic-energy spectrum E(k), k = 1..N/2-1, mean removed -- the
    same spectrum KineticEnergySpectrumLoss compares. Shape (B, N/2-1)."""
    n = field.shape[-1]
    if n not in _KE:
        _KE[n] = KineticEnergySpectrumLoss(size=n)
    return _KE[n].to(field.device).compute_spectrum(field)


def band_errors(pred, target):
    """Per-band MSE by Parseval: sum_b band_errors[:, b] == mse(pred, target)
    (up to float rounding). Shape (B, len(BAND_NAMES))."""
    n = pred.shape[-1]
    d = torch.fft.fft2(pred - target)
    power = (d.abs() ** 2).mean(1) / n ** 4  # mean over u, v -> (B, N, N)
    r = radius_grid(n, pred.device)
    return torch.stack([power[:, (r >= lo) & (r < hi)].sum(1)
                        for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])], dim=1)


def spectral_correlation(pred, target):
    """Phase coherence rho(k) = |sum Yhat conj(Y)| / sqrt(sum|Yhat|^2 sum|Y|^2)
    over each shell [k, k + 1), k = 1..N/2-1, summing u and v together
    (FRC-style). Shape (B, N/2-1).

    It measures phase agreement only: a Gaussian blur multiplies each shell by
    an almost constant real factor, which cancels out, so a blurred field
    scores rho ~ 1. Use transfer_ratio / relative_spectral_error for amplitude."""
    n = pred.shape[-1]
    p, t = torch.fft.fft2(pred), torch.fft.fft2(target)
    cross = (p * t.conj()).sum(1)
    pp, tt = (p.abs() ** 2).sum(1), (t.abs() ** 2).sum(1)
    r = radius_grid(n, pred.device)
    rho = []
    for k in range(1, n // 2):
        mask = (r >= k) & (r < k + 1)
        num = cross[:, mask].sum(1).abs()
        den = torch.sqrt(pp[:, mask].sum(1) * tt[:, mask].sum(1))
        rho.append(num / den)
    return torch.stack(rho, dim=1)


def effective_resolution(rho, threshold: float = 0.5):
    """k_c_fase (phase coherence): the last k before rho first drops below the threshold (the
    'first crossing' convention of FRC), so noise re-crossing at high k can't
    inflate it. rho[:, 0] is k = 1; returns 0 if already below at k = 1."""
    below = rho < threshold
    first_below = torch.where(below.any(1), below.float().argmax(1),
                              torch.full((rho.shape[0],), rho.shape[1], device=rho.device))
    return first_below.float()  # index of first k below = number of k's above, i.e. last k above


def shell_powers(pred, target):
    """Per-sample sums over each shell [k, k + 1), k = 1..N/2-1, of |Yhat|^2,
    |Y|^2 and |Yhat - Y|^2 (u and v together). Summing these over samples
    gives the test-set T(k) and eps(k). Each has shape (B, N/2-1)."""
    n = pred.shape[-1]
    p, t = torch.fft.fft2(pred), torch.fft.fft2(target)
    pp, tt, ee = (p.abs() ** 2).sum(1), (t.abs() ** 2).sum(1), ((p - t).abs() ** 2).sum(1)
    r = radius_grid(n, pred.device)
    masks = [(r >= k) & (r < k + 1) for k in range(1, n // 2)]
    return {name: torch.stack([x[:, m].sum(1) for m in masks], dim=1)
            for name, x in (('pred', pp), ('target', tt), ('error', ee))}


def transfer_ratio(pred_power, target_power):
    """T(k) = E_yhat(k) / E_y(k): amplitude kept per shell (1 = all energy, H(k)^2
    for the blurred input). Pass sums over the test set for the test-set T(k)."""
    return pred_power / target_power


def relative_spectral_error(error_power, target_power):
    """eps(k) = E|Yhat - Y|^2 / E|Y|^2 per shell: amplitude and phase error
    together (0 = perfect, (1 - H)^2 for the blurred input)."""
    return error_power / target_power


def effective_resolution_eps(eps, threshold: float = 0.5):
    """k_c: the largest k such that eps(k') <= threshold for every k' <= k
    (0 if eps(1) > threshold, N/2-1 if it never exceeds it). eps[..., 0] is k = 1."""
    above = eps > threshold
    first_above = torch.where(above.any(-1), above.float().argmax(-1),
                              torch.full(eps.shape[:-1], eps.shape[-1], device=eps.device))
    return first_above.float()


# k_c thresholds on eps(k): 1/2 (the main k_c) and 0.1 (k_c_01, stricter;
# separates models whose eps never reaches 1/2).
K_C_THRESHOLDS = {'k_c': 0.5, 'k_c_01': 0.1}


def k_c_censored(k_c, n_shells: int):
    """True where k_c hit the last shell: eps never exceeded the threshold up
    to the end of the spectrum, so the true k_c is only known to be >= n_shells."""
    return k_c >= n_shells


def spectral_slope(spectrum, k_min: float, k_max: float):
    """Least-squares slope of log E vs log k over k_min <= k <= k_max, per
    sample. The fitting range is set by the user (DECISIONS.md)."""
    k = torch.arange(1, spectrum.shape[1] + 1, dtype=spectrum.dtype, device=spectrum.device)
    sel = (k >= k_min) & (k <= k_max)
    x = torch.log(k[sel])
    y = torch.log(spectrum[:, sel])
    xc = x - x.mean()
    return ((y - y.mean(1, keepdim=True)) * xc).sum(1) / (xc ** 2).sum()
