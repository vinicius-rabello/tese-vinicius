"""LPIPS on velocity fields, shared by the training loss and the evaluation
metric so both always see exactly the same inputs.

LPIPS expects 3-channel images in [-1, 1]. Each velocity component (u, v) is
divided by a fixed scale -- max(|u|, |v|) over the training HR fields, stored
as `lpips_scale` in every run's config.json -- and repeated to 3 channels. The
reported value is the mean of LPIPS(u) and LPIPS(v). Predictions are not
clamped, so they may slightly exceed [-1, 1].
"""

import lpips
import torch


def build_lpips(net: str = 'vgg', device='cpu'):
    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def to_lpips_input(field: torch.Tensor, channel: int, scale: float) -> torch.Tensor:
    """(B, 2, H, W) velocity field -> (B, 3, H, W) LPIPS input for one component."""
    return (field[:, channel:channel + 1] / scale).repeat(1, 3, 1, 1)


def lpips_distance(lpips_model, pred: torch.Tensor, target: torch.Tensor, scale: float,
                   reduction: str = 'mean') -> torch.Tensor:
    """Mean of LPIPS(u) and LPIPS(v); reduction='none' keeps one value per sample."""
    per_channel = [
        lpips_model(to_lpips_input(pred, c, scale), to_lpips_input(target, c, scale)).flatten()
        for c in (0, 1)
    ]
    per_sample = (per_channel[0] + per_channel[1]) / 2
    return per_sample.mean() if reduction == 'mean' else per_sample
