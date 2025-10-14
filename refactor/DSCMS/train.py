import torch
from dataset import SuperResDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import DSCMS
import config
from typing import Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    epoch_loss = 0.0
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)
        outputs = model(lr)
        loss = criterion(outputs, hr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        epoch_loss += loss.item()
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

def train_fn(model: DSCMS, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, scheduler: optim.lr_scheduler.ReduceLROnPlateau):
    loop = tqdm(train_loader, leave=True)
    for idx, (lr, hr) in enumerate(loop):
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)
        if idx % 200 == 0:
            avg_loss = train_one_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, device=config.DEVICE)
            val_loss = validate(model, val_loader, criterion=criterion, device=config.DEVICE)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            plt.imsave(f"saved_images/dscms_hr_{idx}.png", hr.numpy()[0][0])
            plt.imsave(f"saved_images/dscms_lr_{idx}.png", lr.numpy()[0][0])
            plt.imsave(f"saved_images/dscms_output_{idx}.png", model(lr).detach().numpy()[0][0])
    return avg_loss, val_loss, current_lr
    

def main():
    dataset = SuperResDataset(hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=config.BATCH_SIZE)
    model = DSCMS(in_channels=2, out_channels=2)
    model = torch.nn.DataParallel(model)
    model = model.to(config.DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    for epoch in range(config.NUM_EPOCHS):
        avg_loss, val_loss, current_lr = train_fn(model, train_loader, val_loader, optimizer, criterion, scheduler)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss Train.: {avg_loss:.6f}, Loss Val.: {val_loss:.6f}, Learning rate: {current_lr}")
        with open("output_PRUSR.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss Train.: {avg_loss:.4f}, Loss Val.: {val_loss:.4f}, Learning rate: {current_lr}\n")

        if (epoch+1) % 2 == 1:
            torch.save(model.state_dict(), f'test_{epoch+1}.pth')

if __name__ == "__main__":
    main()