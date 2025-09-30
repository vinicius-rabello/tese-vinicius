import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, List, Dict, Any, Type

from DSCMS_paper import DSCMS
from PRUSR import PRUSR
from loader import SuperResNpyDataset2

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
        lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        outputs = model(lr_upsampled)
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
            lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
            outputs = model(lr_upsampled)
            loss = criterion(outputs, hr)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main(
    model_class: Type[nn.Module] = PRUSR,
    in_channels: int = 2,
    out_channels: int = 2,
    data_folder: str = "./data",
    lr_files: List[str] = ["25/window_2003.npy"],
    hr_files: List[str] = ["100/window_2003.npy"],
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: Any = None
) -> None:
    """
    Main training loop for any compatible model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SuperResNpyDataset2(data_folder, lr_files, hr_files)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size)

    model = model_class(in_channels=in_channels, out_channels=out_channels)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print(f"Total training size: {len(train_loader)}, validation size: {len(val_loader)}")

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.6f}, Loss Val.: {val_loss:.6f}, Learning rate: {current_lr}")
        with open("output_PRUSR.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.4f}, Loss Val.: {val_loss:.4f}, Learning rate: {current_lr}\n")

        if (epoch+1) % 2 == 1:
            torch.save(model.state_dict(), f'test_{epoch+1}.pth')

if __name__ == "__main__":
    main()