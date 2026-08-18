import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model_factory import create_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from datasets.super_res_dataset import SuperResDataset
from losses.losses import FluidCompositeLoss

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: dict,
    model_name_display: str
) -> dict:
    model.train()
    # Initialize dictionary to accumulate losses
    epoch_losses = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0, 'kinetic': 0.0}
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")
    
    for batch_idx, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(device), hr.to(device)
        
        # Forward pass now returns the tensor loss AND the metrics dictionary
        outputs = model(lr)
        loss, batch_metrics = criterion(outputs, hr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Accumulate metrics
        for key in epoch_losses:
            epoch_losses[key] += batch_metrics[key]
        
        # Update progress bar with total loss
        loop.set_postfix(loss=loss.item())
        
        # Save subplot grid every 200 batches
        if batch_idx == 0:
            os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/images"), exist_ok=True)
            
            # Extract first image in the batch, first channel
            lr_img = lr.detach().cpu().numpy()[0][0]
            hr_img = hr.detach().cpu().numpy()[0][0]
            pred_img = outputs.detach().cpu().numpy()[0][0]
            
            # Create 1x3 subplot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Dynamically format loss string based on chosen LOSS_MODE
            loss_str = f"Total: {batch_metrics['total']:.4f} | MSE: {batch_metrics['mse']:.4f}"
            if config['LOSS_MODE'] >= 2:
                loss_str += f" | LPIPS: {batch_metrics['lpips']:.4f}"
            if config['LOSS_MODE'] >= 3:
                loss_str += f" | Kin: {batch_metrics['kinetic']:.4f}"
                
            fig.suptitle(f"Epoch {epoch} - Batch {batch_idx}\n{loss_str}", fontsize=14, fontweight='bold')
            
            # Plot LR
            im0 = axes[0].imshow(lr_img, cmap='viridis')
            axes[0].set_title("Low Resolution")
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot Prediction (Now uses the dynamic name)
            im1 = axes[1].imshow(pred_img, cmap='viridis')
            axes[1].set_title(f"Prediction ({model_name_display})")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot HR
            im2 = axes[2].imshow(hr_img, cmap='viridis')
            axes[2].set_title("High Resolution (GT)")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(config['ROOT_FOLDER'], f"output/images/sample_epoch{epoch}_batch{batch_idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
    # Average the metrics over the epoch
    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader)
        
    return epoch_losses


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    model.eval()
    val_losses = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0, 'kinetic': 0.0}
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            outputs = model(lr)
            _, batch_metrics = criterion(outputs, hr)
            
            for key in val_losses:
                val_losses[key] += batch_metrics[key]
                
    for key in val_losses:
        val_losses[key] /= len(val_loader)
        
    return val_losses


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, 
                   train_loss: float, val_loss: float, filepath: str):
    """
    Saves model checkpoint with optimizer state for resuming training.
    """
    # Safely extract state dict whether it is wrapped in DataParallel or not
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, filepath: str):
    """
    Loads model checkpoint and returns the epoch and losses.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def main(config):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 1. EXTRACT MODEL NAME FROM CONFIG (Defaults to 'dscms' if missing)
    current_model = config.get('MODEL_NAME', 'dscms').lower()
    upper_model = current_model.upper()
    
    # Create output directories
    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/logs"), exist_ok=True)
    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/weights"), exist_ok=True)
    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/images"), exist_ok=True)
    
    # Load datasets and dataloaders
    train_dataset = SuperResDataset(
        hr_files=[
            f"./hycom_numpy/hycom_{year}.npy"
            for year in range(2001, 2020)
        ],
        downsample_factor=1,
        freq_threshold=10
    )

    val_dataset = SuperResDataset(
        hr_files=[
            "./hycom_numpy/hycom_2020.npy",
            "./hycom_numpy/hycom_2021.npy",
        ],
        downsample_factor=1,
        freq_threshold=10
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
    )
    
    # 2. INITIALIZE MODEL VIA FACTORY
    print(f"Initializing model: {upper_model}")
    model = create_model(current_model, in_channels=2, out_channels=2)
    
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(config['DEVICE'])
    
    # Loss and optimizer
    criterion = FluidCompositeLoss(
        mode=config['LOSS_MODE'], 
        weights=config['LOSS_WEIGHTS'], 
        device=config['DEVICE'],
        img_size=128 
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    # Track best model based on TOTAL validation loss
    best_val_loss = float('inf')
    
    # Initialize early stopping
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
    
    # Checkpoint resuming
    start_epoch = 0
    resume_checkpoint_path = config.get('RESUME_CHECKPOINT', None)
    
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from {resume_checkpoint_path}")
        start_epoch, _, _ = load_checkpoint(model, optimizer, resume_checkpoint_path)
    else:
        print("Starting training from scratch.")
        
    cumulative_time_sec = 0.0
    
    # 3. SET DYNAMIC FILE PATHS BASED ON MODEL NAME
    log_file_path = os.path.join(config['ROOT_FOLDER'], f"output/logs/{upper_model}_loss.txt")
    best_weight_path = os.path.join(config['ROOT_FOLDER'], f"output/weights/{upper_model}_best.pth")
    
    # Training loop
    for epoch in range(start_epoch, config['NUM_EPOCHS']):
        epoch_start_time = time.perf_counter()
        
        # Train for one epoch (Pass upper_model down for the plot title)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, config['DEVICE'], epoch+1, config, upper_model)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config['DEVICE'])
        
        # Scheduler steps based on total validation loss
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate time taken for this epoch
        epoch_duration = time.perf_counter() - epoch_start_time
        cumulative_time_sec += epoch_duration
        cumulative_time_hrs = cumulative_time_sec / 3600

        # Format the log string
        log_str = (
            f"[Duration: {epoch_duration:.1f}s | Total Duration: {cumulative_time_hrs:.2f}h] Epoch [{epoch+1}/{config['NUM_EPOCHS']}] | LR: {current_lr:.6f}\n"
            f"  Train -> Total: {train_metrics['total']:.6f} | MSE: {train_metrics['mse']:.6f} | "
            f"LPIPS: {train_metrics['lpips']:.6f} | Kin: {train_metrics['kinetic']:.6f}\n"
            f"  Val   -> Total: {val_metrics['total']:.6f} | MSE: {val_metrics['mse']:.6f} | "
            f"LPIPS: {val_metrics['lpips']:.6f} | Kin: {val_metrics['kinetic']:.6f}\n"
            f"{'-'*70}\n"
        )
        
        # Print to console
        print(log_str.strip())

        # Append to log file dynamically
        with open(log_file_path, "a") as f:
            f.write(log_str)

        # Save best model dynamically
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            save_checkpoint(model, optimizer, epoch+1, train_metrics['total'], val_metrics['total'],
                          best_weight_path)
        
        # Save checkpoint every 50 epochs dynamically
        if (epoch+1) % 50 == 0:
            epoch_weight_path = os.path.join(config['ROOT_FOLDER'], f"output/weights/{upper_model}_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch+1, train_metrics['total'], val_metrics['total'],
                          epoch_weight_path)

        # Check early stopping
        early_stopper(val_metrics['total'])
        if early_stopper.early_stop:
            print("Early stopping triggered! Validation loss has not improved sufficiently.")
            break


# 4. NEW ENTRY POINT - READS THE YAML CONFIG
if __name__ == "__main__":
    import argparse
    import yaml
    
    # Set up argument parsing to grab the YAML file from the terminal
    parser = argparse.ArgumentParser(description="Train Super Resolution Models")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    
    # Read the YAML file into a Python dictionary
    with open(args.config, 'r') as file:
        experiment_config = yaml.safe_load(file)
        
    # Pass the dictionary directly into the main function
    main(experiment_config)