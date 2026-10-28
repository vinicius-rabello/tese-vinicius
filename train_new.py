import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


# --- Import your models and datasets ---
from models.PRUSR.model import PRUSR
from datasets.sr_tiny_dataset import SRTinyDataset

# --- Import your custom physics losses ---
# (Assuming you saved the losses from our previous steps into a file named 'losses.py')
from metrics.kinetic_energy_spectrum import KineticEnergySpectrumLoss, WeightedKineticEnergySpectrumLoss
from pytorch_msssim import ssim

# ==========================================
# Dynamic Composite Loss Module
# ==========================================
class FluidCompositeLoss(nn.Module):
    def __init__(self, loss_config, img_size=32):
        super().__init__()
        self.weights = loss_config
        
        # Initialize only the loss modules we actually need (weights > 0)
        if self.weights.get('mse', 0) > 0:
            self.mse = nn.MSELoss()
            
        if self.weights.get('epi', 0) > 0:
            # We are using MSSSIM as a proxy for structural/edge preservation here based on your original code.
            # Replace with a custom EPI loss class if you have one.
            self.use_ssim = True 
            
        if self.weights.get('spectral_unweighted', 0) > 0:
            self.spec_unweighted = KineticEnergySpectrumLoss(size=img_size)
            
        if self.weights.get('spectral_weighted', 0) > 0:
            self.spec_weighted = WeightedKineticEnergySpectrumLoss(size=img_size)

    def forward(self, pred, target):
        total_loss = 0.0
        loss_dict = {} # Keep track of individual components for the progress bar
        
        if hasattr(self, 'mse'):
            l_mse = self.mse(pred, target)
            total_loss += self.weights['mse'] * l_mse
            loss_dict['mse'] = l_mse.item()
            
        if getattr(self, 'use_ssim', False):
            # Using 1 - SSIM as the edge/structure preservation loss
            l_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True)
            total_loss += self.weights['epi'] * l_ssim
            loss_dict['epi'] = l_ssim.item()
            
        if hasattr(self, 'spec_unweighted'):
            l_spec = self.spec_unweighted(pred, target)
            total_loss += self.weights['spectral_unweighted'] * l_spec
            loss_dict['spec_unweighted'] = l_spec.item()
            
        if hasattr(self, 'spec_weighted'):
            l_spec_w = self.spec_weighted(pred, target)
            total_loss += self.weights['spectral_weighted'] * l_spec_w
            loss_dict['spec_weighted'] = l_spec_w.item()
            
        return total_loss, loss_dict


# ==========================================
# Training Utilities
# ==========================================
def get_data_loaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)


# ==========================================
# Core Epoch Loops
# ==========================================
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, out_dir):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")
    
    for batch_idx, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(device), hr.to(device)
        
        # Forward pass
        outputs = model(lr)
        
        # Dynamic Loss Calculation
        loss, loss_dict = criterion(outputs, hr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar with specific loss breakdown
        display_dict = {'Total': f"{loss.item():.4f}"}
        display_dict.update({k: f"{v:.4f}" for k, v in loss_dict.items()})
        loop.set_postfix(**display_dict)
        
        # Save sample images every 200 batches to the specific experiment folder
        if batch_idx % 200 == 0:
            img_dir = os.path.join(out_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            plt.imsave(os.path.join(img_dir, f"hr_epoch{epoch}_batch{batch_idx}.png"), hr.cpu().numpy()[0][0])
            plt.imsave(os.path.join(img_dir, f"lr_epoch{epoch}_batch{batch_idx}.png"), lr.cpu().numpy()[0][0])
            plt.imsave(os.path.join(img_dir, f"output_epoch{epoch}_batch{batch_idx}.png"), outputs.detach().cpu().numpy()[0][0])
    
    return epoch_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            outputs = model(lr)
            loss, _ = criterion(outputs, hr) # We only need the total loss for validation
            val_loss += loss.item()
    return val_loss / len(val_loader)


# ==========================================
# Main Execution
# ==========================================
def main(config_path):
    # 1. Load YAML Configuration
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # 2. Setup Environment and Paths
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment-specific directories
    exp_dir = os.path.join(cfg['experiment']['root_folder'], cfg['experiment']['name'])
    log_dir = os.path.join(exp_dir, "logs")
    weight_dir = os.path.join(exp_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    
    print(f"--- Starting Experiment: {cfg['experiment']['name']} ---")
    print(f"Logging to: {exp_dir}")

    # 3. Load Dataset
    dataset = SRTinyDataset(hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=cfg['training']['batch_size'])
    
    # 4. Initialize Model
    if cfg['model']['architecture'] == "PRUSR":
        model = PRUSR(in_channels=cfg['model']['in_channels'], out_channels=cfg['model']['out_channels'])
    else:
        raise ValueError(f"Unknown architecture: {cfg['model']['architecture']}")
        
    model = torch.nn.DataParallel(model).to(device)
    
    # 5. Initialize Dynamic Loss, Optimizer, and Scheduler
    criterion = FluidCompositeLoss(cfg['loss_weights'], img_size=32)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=cfg['training']['patience'], factor=0.5)
    
    best_val_loss = float('inf')
    num_epochs = cfg['training']['num_epochs']
    
    # 6. Training Loop
    for epoch in range(num_epochs):
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1, exp_dir)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update LR
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Console Logging
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {train_loss:.6f}, Loss Val.: {val_loss:.6f}, LR: {current_lr}")
        
        # File Logging
        with open(os.path.join(log_dir, "loss_log.txt"), "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {train_loss:.4f}, Loss Val.: {val_loss:.4f}, LR: {current_lr}\n")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, os.path.join(weight_dir, 'best_model.pth'))

        # Save Periodic Checkpoint (e.g., every 50 epochs)
        if (epoch+1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, os.path.join(weight_dir, f'model_epoch{epoch+1}.pth'))


if __name__ == "__main__":
    # Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Train Fluid Super-Resolution Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    
    main(args.config)