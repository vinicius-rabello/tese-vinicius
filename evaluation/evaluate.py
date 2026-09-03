# evaluation/evaluate.py

import numpy as np
import torch
import torch.nn.functional as F

from metrics.image_metrics import (
    mseLoss,
    ssimLoss,
    psnrLoss,
    lpipsLoss,
    fsimLoss,
    epiLoss,
)

from metrics.physical_metrics import LossTKE
from metrics.kinetic_energy_spectrum import KineticEnergySpectrumLoss, WeightedKineticEnergySpectrumLoss


def evaluate_model(model, test_loader, device, img_size=128):
    """Runs a model over test_loader, returning per-channel mean/std for every
    per-channel metric, plus channel-combined mean/std for ke/wke (the
    kinetic-energy-spectrum losses inherently mix u and v, so they aren't
    split by channel like the others).

    test_loader must yield (lr, hr) tuples, as produced by SuperResDataset.
    """
    model.eval()

    ke_loss_fn = KineticEnergySpectrumLoss(size=img_size).to(device)
    wke_loss_fn = WeightedKineticEnergySpectrumLoss(size=img_size).to(device)

    metrics = {
        'mse': {'u': [], 'v': []},
        'ssim': {'u': [], 'v': []},
        'psnr': {'u': [], 'v': []},
        'lpips': {'u': [], 'v': []},
        'fsim': {'u': [], 'v': []},
        'epi': {'u': [], 'v': []},
        'tke': {'u': [], 'v': []},
    }
    ke_values, wke_values = [], []

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            for channel in range(2):  # u and v components
                channel_label = 'u' if channel == 0 else 'v'
                metrics['mse'][channel_label].append(mseLoss(output, hr, channel, idx))
                metrics['ssim'][channel_label].append(ssimLoss(output, hr, channel, idx))
                metrics['psnr'][channel_label].append(psnrLoss(output, hr, channel, idx))
                metrics['lpips'][channel_label].append(lpipsLoss(output, hr, channel, idx))
                metrics['fsim'][channel_label].append(fsimLoss(output, hr, channel, idx))
                metrics['epi'][channel_label].append(epiLoss(output, hr, channel, idx))
                metrics['tke'][channel_label].append(LossTKE(output, hr, channel, idx))

            ke_values.append(ke_loss_fn(output, hr).item())
            wke_values.append(wke_loss_fn(output, hr).item())

    result = {k: {'u': {'mean': np.mean(v['u']), 'std': np.std(v['u'])},
                   'v': {'mean': np.mean(v['v']), 'std': np.std(v['v'])}}
              for k, v in metrics.items()}
    result['ke'] = {'mean': np.mean(ke_values), 'std': np.std(ke_values)}
    result['wke'] = {'mean': np.mean(wke_values), 'std': np.std(wke_values)}
    return result


def interpolate_baseline(lr, mode, downsample_factor=4):
    """Baseline output by interpolating the coarse field back to HR size.

    lr has shape [B, C, H, W] where H, W are already the upsampled (nearest)
    dimensions. Recovers the coarse grid via adaptive avg pool, then
    interpolates with the chosen mode.
    """
    coarse_h = lr.shape[2] // downsample_factor
    coarse_w = lr.shape[3] // downsample_factor
    coarse = F.adaptive_avg_pool2d(lr, (coarse_h, coarse_w))
    return F.interpolate(coarse, size=(lr.shape[2], lr.shape[3]), mode=mode,
                          align_corners=False if mode != 'nearest' else None)