import torch
import torch.nn as nn
import lpips
# Adjust this import based on your exact folder structure
from metrics.kinetic_energy_spectrum import KineticEnergySpectrumLoss, WeightedKineticEnergySpectrumLoss
from losses.lpips_utils import build_lpips, lpips_distance

class FluidCompositeLoss(nn.Module):
    """
    mode 1: MSE
    mode 2: MSE + LPIPS
    mode 3: MSE + LPIPS + Kinetic energy spectrum
    mode 4: MSE + Weighted kinetic energy spectrum (no LPIPS)
    """
    def __init__(self, mode, weights, device, img_size=128):
        super().__init__()
        self.mode = mode
        self.weights = weights

        self.mse = nn.MSELoss()

        if self.mode in (2, 3):
            self.lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False

        if self.mode == 3:
            self.kinetic_model = KineticEnergySpectrumLoss(size=img_size).to(device)

        if self.mode == 4:
            self.weighted_kinetic_model = WeightedKineticEnergySpectrumLoss(size=img_size).to(device)

    def forward(self, pred, target):
        # Dictionary to store raw (unweighted) values for logging
        metrics = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0, 'kinetic': 0.0, 'weighted_kinetic': 0.0}

        # 1. MSE
        l_mse = self.mse(pred, target)
        metrics['mse'] = l_mse.item()
        total_loss = self.weights['mse'] * l_mse

        # 2. LPIPS
        if self.mode in (2, 3):
            l_lpips_u = self.lpips_model(pred[:, 0:1, :, :], target[:, 0:1, :, :]).mean()
            l_lpips_v = self.lpips_model(pred[:, 1:2, :, :], target[:, 1:2, :, :]).mean()
            l_lpips = (l_lpips_u + l_lpips_v) / 2.0

            metrics['lpips'] = l_lpips.item()
            total_loss += self.weights['lpips'] * l_lpips

        # 3. Kinetic
        if self.mode == 3:
            l_kin = self.kinetic_model(pred, target)
            metrics['kinetic'] = l_kin.item()
            total_loss += self.weights['kinetic'] * l_kin

        # 4. Weighted kinetic
        if self.mode == 4:
            l_wkin = self.weighted_kinetic_model(pred, target)
            metrics['weighted_kinetic'] = l_wkin.item()
            total_loss += self.weights['weighted_kinetic'] * l_wkin

        metrics['total'] = total_loss.item()
        return total_loss, metrics

class MseLpipsLoss(nn.Module):
    """loss = MSE + lam * LPIPS, with LPIPS on the fixed-scale fields
    (see losses/lpips_utils.py). With lam = 0 the LPIPS network is never
    built and the loss is exactly nn.MSELoss.

    forward() returns (loss, metrics). LPIPS is only computed when it enters
    the loss (lam > 0) or when compute_lpips=True, which validation uses to
    log it every epoch whatever lam is; otherwise metrics['lpips'] is NaN."""

    def __init__(self, lam: float, lpips_scale: float, device, net: str = 'vgg'):
        super().__init__()
        self.lam = float(lam)
        self.lpips_scale = float(lpips_scale)
        self.net = net
        self.device = device
        self.mse = nn.MSELoss()
        self._lpips_model = None

    @property
    def lpips_model(self):
        # Built on first use, so a lam = 0 training loop never loads VGG.
        if self._lpips_model is None:
            self._lpips_model = build_lpips(self.net, self.device)
        return self._lpips_model

    def forward(self, pred, target, compute_lpips: bool = False):
        l_mse = self.mse(pred, target)
        metrics = {'mse': l_mse.item(), 'lpips': float('nan')}
        total = l_mse
        if self.lam > 0 or compute_lpips:
            l_lpips = lpips_distance(self.lpips_model, pred, target, self.lpips_scale)
            metrics['lpips'] = l_lpips.item()
            if self.lam > 0:
                total = l_mse + self.lam * l_lpips
        metrics['total'] = total.item()
        return total, metrics
