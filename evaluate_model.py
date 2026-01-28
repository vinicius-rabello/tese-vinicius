import numpy as np
import torch
import json
import os
from torch.utils.data import DataLoader
from models.DSCMS.model import DSCMS
from models.DSCMS import config
from datasets.sr_tiny_dataset import SRTinyDataset
from metrics.image_metrics import mseLoss, ssimLoss, psnrLoss, lpipsLoss, fsimLoss, epiLoss
from metrics.physical_metrics import LossTKE
import argparse

def evaluate_model(model, test_loader, device):
    model.eval()
    
    metrics = {
        'mse': {'psi1': [], 'psi2': []},
        'ssim': {'psi1': [], 'psi2': []},
        'psnr': {'psi1': [], 'psi2': []},
        'lpips': {'psi1': [], 'psi2': []},
        'fsim': {'psi1': [], 'psi2': []},
        'epi': {'psi1': [], 'psi2': []},
        'tke': {'psi1': [], 'psi2': []}
    }
    
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            
            # Compute metrics for each channel
            for channel in range(2):  # psi1 and psi2 components
                metrics['mse'][f'psi{channel+1}'].append(mseLoss(output, hr, channel, idx))
                metrics['ssim'][f'psi{channel+1}'].append(ssimLoss(output, hr, channel, idx))
                metrics['psnr'][f'psi{channel+1}'].append(psnrLoss(output, hr, channel, idx))
                metrics['lpips'][f'psi{channel+1}'].append(lpipsLoss(output, hr, channel, idx))
                metrics['fsim'][f'psi{channel+1}'].append(fsimLoss(output, hr, channel, idx))
                metrics['epi'][f'psi{channel+1}'].append(epiLoss(output, hr, channel, idx))
                metrics['tke'][f'psi{channel+1}'].append(LossTKE(output, hr, channel, idx))

    # Aggregate results
    results = {
        k: {
            'psi1': {'mean': float(np.mean(v['psi1'])), 'std': float(np.std(v['psi1']))},
            'psi2': {'mean': float(np.mean(v['psi2'])), 'std': float(np.std(v['psi2']))}
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

    # Load model based on model_name
    if model_name == 'DSCMS':
        from models.DSCMS.model import DSCMS
        model = DSCMS(2, 2)
        checkpoint_path = './models/DSCMS/output/weights/DSCMS_best.pth'
    elif model_name == 'PRUSR':
        from models.PRUSR.model import PRUSR  # Adjust import as needed
        model = PRUSR(2, 2)  # Adjust parameters as needed
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
                        help='Model name to evaluate (e.g., DSCMS, PRUSR)')
    args = parser.parse_args()
    main(args.model)