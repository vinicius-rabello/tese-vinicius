import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.DSCMS.model import DSCMS
from models.DSCMS import config
from typing import Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datasets.super_res_dataset import SuperResDataset


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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")
    
    for batch_idx, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(device), hr.to(device)
        
        # Forward pass
        outputs = model(lr)
        loss = criterion(outputs, hr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
        
        # Save sample images every 200 batches
        if batch_idx % 200 == 0:
            os.makedirs(config.ROOT_FOLDER + "output/images", exist_ok=True)
            plt.imsave(config.ROOT_FOLDER + f"output/images/hr_epoch{epoch}_batch{batch_idx}.png", 
                      hr.cpu().numpy()[0][0])
            plt.imsave(config.ROOT_FOLDER + f"output/images/lr_epoch{epoch}_batch{batch_idx}.png", 
                      lr.cpu().numpy()[0][0])
            plt.imsave(config.ROOT_FOLDER + f"output/images/output_epoch{epoch}_batch{batch_idx}.png", 
                      outputs.detach().cpu().numpy()[0][0])
    
    return epoch_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            outputs = model(lr)
            loss = criterion(outputs, hr)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, 
                   train_loss: float, val_loss: float, filepath: str):
    """
    Saves model checkpoint with optimizer state for resuming training.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create output directories
    os.makedirs(config.ROOT_FOLDER + "output/logs", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/weights", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/images", exist_ok=True)
    
    # Load dataset
    dataset = SuperResDataset(
        hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    train_loader, val_loader = get_data_loaders(
        dataset, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    model = DSCMS(in_channels=2, out_channels=2)
    model = torch.nn.DataParallel(model)
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, epoch+1)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
              f"Loss Train.: {train_loss:.6f}, Loss Val.: {val_loss:.6f}, "
              f"Learning rate: {current_lr}")
        
        with open(config.ROOT_FOLDER + "output/logs/DSCMS_loss.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
                   f"Loss Train.: {train_loss:.4f}, Loss Val.: {val_loss:.4f}, "
                   f"Learning rate: {current_lr}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss,
                          config.ROOT_FOLDER + 'output/weights/DSCMS_best.pth')
        
        # Save checkpoint every 2 epochs
        if (epoch+1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss,
                          config.ROOT_FOLDER + f'output/weights/DSCMS_epoch{epoch+1}.pth')


if __name__ == "__main__":
    main()