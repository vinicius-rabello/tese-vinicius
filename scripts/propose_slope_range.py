"""Proposes the k range for the spectral-slope fit from the target (HR)
spectrum of the test set, and plots it for the user to confirm.

    python scripts/propose_slope_range.py

Looks at the local slope d log E / d log k of the mean test-set HR spectrum
in the restored region (k > D0) and picks the widest range where it stays
within +-TOL of its median there. Writes figures/slope_range_proposal.png/.pdf.
Nothing is fixed until the range is confirmed (collect_results.SLOPE_RANGE).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from evaluation.spectral_metrics import D0, radial_spectrum  # noqa: E402

TOL = 0.5  # allowed deviation of the local slope from its median (in log-log units)
OUT = "figures/slope_range_proposal"


def main():
    ds = SuperResDataset([f"./hycom_numpy/hycom_{y}.npy" for y in (2022, 2023, 2024)], 1, 10)
    with torch.no_grad():
        spectra = torch.cat([radial_spectrum(hr) for _, hr in DataLoader(ds, batch_size=64)])
    e = spectra.mean(0).double().numpy()
    k = np.arange(1, len(e) + 1)
    logk, loge = np.log(k), np.log(e)
    local = np.gradient(loge, logk)
    # 5-point running mean: the raw local slope of a mean spectrum is still noisy shell to shell
    smooth = np.convolve(local, np.ones(5) / 5, mode='same')

    region = k > D0
    med = np.median(smooth[region & (k < len(k) - 2)])
    ok = region & (np.abs(smooth - med) <= TOL)
    # widest contiguous run of ok shells
    best, start = (0, 0), None
    for i, flag in enumerate(np.append(ok, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    k_min, k_max = int(k[best[0]]), int(k[best[1] - 1])
    sel = (k >= k_min) & (k <= k_max)
    slope, intercept = np.polyfit(logk[sel], loge[sel], 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.loglog(k, e, color='black', lw=2, label='Target (HR), test-set mean')
    ax1.loglog(k[sel], np.exp(intercept) * k[sel] ** slope, color='red', lw=2, ls='--',
               label=f'fit on k = {k_min}..{k_max}: slope {slope:.2f}')
    ax1.axvspan(k_min, k_max, color='red', alpha=0.08)
    ax1.axvline(D0, color='gray', ls=':', lw=1, label=f'D0 = {D0:g}')
    ax1.set_ylabel('Kinetic energy E(k) (m²/s²)')
    ax1.set_title('Proposed spectral-slope range (to be confirmed)')
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.semilogx(k, local, color='gray', lw=1, label='local slope d log E / d log k')
    ax2.semilogx(k, smooth, color='black', lw=1.5, label='5-shell running mean')
    ax2.axhline(med, color='red', ls=':', lw=1, label=f'median for k > D0: {med:.2f} (±{TOL})')
    ax2.axvspan(k_min, k_max, color='red', alpha=0.08)
    ax2.axvline(D0, color='gray', ls=':', lw=1)
    ax2.set_xlabel('Wavenumber k (cycles per 128-px domain; 1 px = 0.04°)')
    ax2.set_ylabel('Local slope (dimensionless)')
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    ax2.legend(fontsize=8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches='tight')
    print(f"Proposed range: k = {k_min}..{k_max} (target slope {slope:.2f}); local-slope median for k > D0: {med:.2f}")
    print(f"Wrote {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
