import numpy as np
import torch
import json
import os
from torch.utils.data import DataLoader
from models.DSCMS.model import DSCMS
from models.DSCMS import config
from datasets.sr_tiny_dataset import SRTinyDataset
from datasets.super_res_dataset import SuperResDataset
from metrics.image_metrics import mseLoss, ssimLoss, psnrLoss, lpipsLoss, fsimLoss, epiLoss
from metrics.physical_metrics import LossTKE
import argparse
import torch.nn.functional as F


def interpolate_baseline(lr, mode, downsample_factor=4):
    """Generate baseline output by interpolating the coarse field back to HR size.
    lr has shape [B, C, H, W] where H, W are already the upsampled (nearest) dimensions.
    We first downsample back to the coarse grid, then interpolate with the chosen mode.
    """
    coarse_h = lr.shape[2] // downsample_factor
    coarse_w = lr.shape[3] // downsample_factor
    # Recover the coarse field via adaptive avg pool (reverses nearest-neighbor replication)
    coarse = F.adaptive_avg_pool2d(lr, (coarse_h, coarse_w))
    # Upsample with the desired interpolation mode
    output = F.interpolate(coarse, size=(lr.shape[2], lr.shape[3]), mode=mode,
                           align_corners=False if mode != 'nearest' else None)
    return output


def evaluate_model(model, test_loader, device):
    model.eval()
    
    metrics = {
        'mse': {'u': [], 'v': []},
        'ssim': {'u': [], 'v': []},
        'psnr': {'u': [], 'v': []},
        'lpips': {'u': [], 'v': []},
        'fsim': {'u': [], 'v': []},
        'epi': {'u': [], 'v': []},
        'tke': {'u': [], 'v': []}
    }
    
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            
            # Compute metrics for each channel
            for channel in range(2):  # psi1 and psi2 components
                channel_label = 'u' if channel == 0 else 'v'
                metrics['mse'][channel_label].append(mseLoss(output, hr, channel, idx))
                metrics['ssim'][channel_label].append(ssimLoss(output, hr, channel, idx))
                metrics['psnr'][channel_label].append(psnrLoss(output, hr, channel, idx))
                metrics['lpips'][channel_label].append(lpipsLoss(output, hr, channel, idx))
                metrics['fsim'][channel_label].append(fsimLoss(output, hr, channel, idx))
                metrics['epi'][channel_label].append(epiLoss(output, hr, channel, idx))
                metrics['tke'][channel_label].append(LossTKE(output, hr, channel, idx))

    # Aggregate results
    results = {
        k: {
            'u': {'mean': round(float(np.mean(v['u'])), 5), 'std': round(float(np.std(v['u'])), 5)},
            'v': {'mean': round(float(np.mean(v['v'])), 5), 'std': round(float(np.std(v['v'])), 5)}
        }
        for k, v in metrics.items()
    }
    return results

def evaluate_baseline(mode, test_loader, device, downsample_factor=4):
    """Evaluate a classical interpolation baseline (bilinear or bicubic)."""
    metrics = {
        'mse': {'u': [], 'v': []},
        'ssim': {'u': [], 'v': []},
        'psnr': {'u': [], 'v': []},
        'lpips': {'u': [], 'v': []},
        'fsim': {'u': [], 'v': []},
        'epi': {'u': [], 'v': []},
        'tke': {'u': [], 'v': []}
    }

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            output = interpolate_baseline(lr, mode=mode, downsample_factor=downsample_factor)

            for channel in range(2):
                channel_label = 'u' if channel == 0 else 'v'
                metrics['mse'][channel_label].append(mseLoss(output, hr, channel, idx))
                metrics['ssim'][channel_label].append(ssimLoss(output, hr, channel, idx))
                metrics['psnr'][channel_label].append(psnrLoss(output, hr, channel, idx))
                metrics['lpips'][channel_label].append(lpipsLoss(output, hr, channel, idx))
                metrics['fsim'][channel_label].append(fsimLoss(output, hr, channel, idx))
                metrics['epi'][channel_label].append(epiLoss(output, hr, channel, idx))
                metrics['tke'][channel_label].append(LossTKE(output, hr, channel, idx))

    results = {
        k: {
            'u': {'mean': round(float(np.mean(v['u'])), 5), 'std': round(float(np.std(v['u'])), 5)},
            'v': {'mean': round(float(np.mean(v['v'])), 5), 'std': round(float(np.std(v['v'])), 5)}
        }
        for k, v in metrics.items()
    }
    return results


def main(model_name):
    # Setup test dataset
    test_dataset = SRTinyDataset(
        hr_files=['data/100/window_2023.npy'],
        downsample_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Interpolation baselines
    BASELINE_MODES = {'bilinear': 'bilinear', 'bicubic': 'bicubic'}

    if model_name in BASELINE_MODES:
        mode = BASELINE_MODES[model_name]
        print(f"Evaluating {model_name} interpolation baseline...")
        results = evaluate_baseline(mode, test_loader, device, downsample_factor=4)
    else:
        # Load learned model based on model_name
        if model_name == 'DSCMS':
            from models.DSCMS.model import DSCMS
            model = DSCMS(2, 2)
            checkpoint_path = './models/DSCMS/output/weights/DSCMS_best.pth'
        elif model_name == 'PRUSR':
            from models.PRUSR.model import PRUSR
            model = PRUSR(2, 2)
            checkpoint_path = './models/PRUSR/output/weights/PRUSR_best.pth'
        # Add more models here as needed
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load checkpoint
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Evaluating {model_name}...")
        results = evaluate_model(model, test_loader, device)

    # Save results
    output_dir = f'./models/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Super Resolution Model')
    parser.add_argument('--model', type=str, required=True, 
                    help='Model name to evaluate (e.g., DSCMS, PRUSR, bilinear, bicubic)')
    args = parser.parse_args()
    main(args.model)