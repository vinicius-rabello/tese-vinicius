import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.DSCMS.model import DSCMS
from models.DSCMS import config
from datasets.super_res_dataset import SuperResDataset
from datasets.sr_tiny_dataset import SRTinyDataset
from metrics.image_metrics import ssimLoss, psnrLoss, lpipsLoss, fsimLoss, epiLoss
from metrics.physical_metrics import LossTKE
from typing import Tuple

def compute_metrics_for_dataset(model, dataloader, device):
    # Store metrics for u and v
    metrics = {
        "MSE_u": [], "MSE_v": [],
        "SSIM_u": [], "SSIM_v": [],
        "PSNR_u": [], "PSNR_v": [],
        "LPIPS_u": [], "LPIPS_v": [],
        "FSIM_u": [], "FSIM_v": [],
        "EPI_u": [], "EPI_v": [],
        "TKE_u": [], "TKE_v": [],
    }

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
            output = model(lr_upsampled)

            # Convert to numpy for MSE
            high = hr.cpu().squeeze().numpy()
            pred = output.cpu().squeeze().numpy()

            metrics["MSE_u"].append(np.mean((high[0] - pred[0]) ** 2))
            metrics["MSE_v"].append(np.mean((high[1] - pred[1]) ** 2))

            metrics["SSIM_u"].append(ssimLoss(output, hr, 0, 0))
            metrics["SSIM_v"].append(ssimLoss(output, hr, 1, 0))

            metrics["PSNR_u"].append(psnrLoss(output, hr, 0, 0))
            metrics["PSNR_v"].append(psnrLoss(output, hr, 1, 0))

            metrics["LPIPS_u"].append(lpipsLoss(output, hr, 0, 0))
            metrics["LPIPS_v"].append(lpipsLoss(output, hr, 1, 0))

            metrics["FSIM_u"].append(fsimLoss(output, hr, 0, 0))
            metrics["FSIM_v"].append(fsimLoss(output, hr, 1, 0))

            metrics["EPI_u"].append(epiLoss(output, hr, 0, 0))
            metrics["EPI_v"].append(epiLoss(output, hr, 1, 0))

            metrics["TKE_u"].append(LossTKE(output, hr, 0, 0))
            metrics["TKE_v"].append(LossTKE(output, hr, 1, 0))

    return metrics

def print_and_save_averages(metrics, save_path="metrics_averages.txt"):
    with open(save_path, "w") as f:
        for key, values in metrics.items():
            arr = np.array(values)
            mean, std = arr.mean(), arr.std()
            print(f"{key}: mean={mean:.6f}, std={std:.6f}")
            f.write(f"{key}: mean={mean:.6f}, std={std:.6f}\n")

def get_data_loaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets and returns their DataLoaders.
    """
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model and data
    model = DSCMS(2, 2)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load('./models/DSCMS/output/weights/DSCMS_best.pth', map_location=torch.device(device)))
    model.eval()
    mean = np.array([-0.00561308, 0.07556629])
    std = np.array([0.32576539, 0.38299691])
    # Load dataset
    dataset = SRTinyDataset(
        hr_files=['data/100/window_2023.npy'], downsample_factor=4)
    dataloader, _ = get_data_loaders(dataset, batch_size=config.BATCH_SIZE)

    metrics = compute_metrics_for_dataset(model, dataloader, device)
    print_and_save_averages(metrics)