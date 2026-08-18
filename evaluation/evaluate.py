# evaluation/evaluate.py

import numpy as np
import torch

from metrics.image_metrics import (
    mseLoss,
    ssimLoss,
    psnrLoss,
    lpipsLoss,
    fsimLoss,
    epiLoss,
)

from metrics.physical_metrics import LossTKE


CHANNELS = {
    "u": 0,
    "v": 1,
}

def compute_sample_metrics(
    prediction,
    target,
):
    results = {}

    for channel_name, channel_idx in CHANNELS.items():

        results[channel_name] = {
            "mse": mseLoss(prediction, target, channel_idx),
            "ssim": ssimLoss(prediction, target, channel_idx),
            "psnr": psnrLoss(prediction, target, channel_idx),
            "lpips": lpipsLoss(prediction, target, channel_idx),
            "fsim": fsimLoss(prediction, target, channel_idx),
            "epi": epiLoss(prediction, target, channel_idx),
            "tke": LossTKE(prediction, target, channel_idx),
        }

    return results

def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def evaluate_model(
    model,
    dataloader,
    device,
):
    model.eval()

    metrics = {
        metric: {
            "u": [],
            "v": [],
        }
        for metric in [
            "mse",
            "ssim",
            "psnr",
            "lpips",
            "fsim",
            "epi",
            "tke",
        ]
    }

    with torch.no_grad():

        for batch in dataloader:

            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            prediction = model(lr)

            sample = compute_sample_metrics(
                prediction,
                hr,
            )

            for channel in ["u", "v"]:
                for metric in metrics:
                    metrics[metric][channel].append(
                        sample[channel][metric]
                    )

    return {
        metric: {
            channel: summarize(values)
            for channel, values in metric_data.items()
        }
        for metric, metric_data in metrics.items()
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    
    metrics = {
        'mse': {'u': [], 'v': []},
        'ssim': {'u': [], 'v': []},
        'psnr': {'u': [], 'v': []},
        'lpips': {'u': [], 'v': []},
        'fsim': {'u': [], 'v': []},
        'epi':{'u': [], 'v': []},
        'tke': {'u': [], 'v': []}
    }
    
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            # Compute metrics for each channel
            for channel in range(2):  # u and v components
                channel_label = 'u' if channel == 0 else 'v'
                metrics['mse'][f'{channel_label}'].append(mseLoss(output, hr, channel, idx))
                metrics['ssim'][f'{channel_label}'].append(ssimLoss(output, hr, channel, idx))
                metrics['psnr'][f'{channel_label}'].append(psnrLoss(output, hr, channel, idx))
                metrics['lpips'][f'{channel_label}'].append(lpipsLoss(output, hr, channel, idx))
                metrics['fsim'][f'{channel_label}'].append(fsimLoss(output, hr, channel, idx))
                metrics['epi'][f'{channel_label}'].append(epiLoss(output, hr, channel, idx))
                metrics['tke'][f'{channel_label}'].append(LossTKE(output, hr, channel, idx))

    # Aggregate results
    results = {k: {'u': {'mean': np.mean(v['u']), 'std': np.std(v['u'])},
                    'v': {'mean': np.mean(v['v']), 'std': np.std(v['v'])}}
               for k, v in metrics.items()}
    return results