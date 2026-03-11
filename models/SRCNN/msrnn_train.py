import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.SRCNN.model import SRCNN
from models.SRCNN import config
from typing import Tuple, List
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datasets.super_res_dataset import SuperResDataset
from datasets.sr_tiny_dataset import SRTinyDataset


NUM_STAGES = 4


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


# ---------------------------------------------------------------------------
# NEW: compute the normalised residual target for a batch.
# For stage 0: target = HR / eps  (eps = RMS of HR)
# For stage n: target = (HR - sum_i eps_i * u_i(LR)) / eps_n
# ---------------------------------------------------------------------------
def compute_target(
    lr: torch.Tensor,
    hr: torch.Tensor,
    previous_models: List[nn.Module],
    previous_epsilons: List[float],
    epsilon: float
) -> torch.Tensor:
    with torch.no_grad():
        residual = hr.clone()
        for prev_model, eps in zip(previous_models, previous_epsilons):
            residual = residual - eps * prev_model(lr)
    return residual / epsilon


# ---------------------------------------------------------------------------
# NEW: compute RMS of the residual over the full training loader.
# Called once before each stage to get epsilon_n.
# ---------------------------------------------------------------------------
def compute_rms(
    loader: DataLoader,
    device: torch.device,
    previous_models: List[nn.Module],
    previous_epsilons: List[float]
) -> float:
    sum_sq = 0.0
    count = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            residual = hr.clone()
            for prev_model, eps in zip(previous_models, previous_epsilons):
                residual = residual - eps * prev_model(lr)
            sum_sq += (residual ** 2).sum().item()
            count += residual.numel()
    rms = (sum_sq / count) ** 0.5
    return rms if rms > 1e-12 else 1.0


# ---------------------------------------------------------------------------
# train_one_epoch / validate: identical to train.py except they receive
# a pre-computed `target` tensor instead of using `hr` directly.
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    stage: int,
    previous_models: List[nn.Module],
    previous_epsilons: List[float],
    epsilon: float
) -> float:
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, leave=True, desc=f"Stage {stage+1} | Epoch {epoch}")

    for batch_idx, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(device), hr.to(device)

        # ---- only change from train.py: target is the normalised residual ----
        target = compute_target(lr, hr, previous_models, previous_epsilons, epsilon)

        # Forward pass
        outputs = model(lr)
        loss = criterion(outputs, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # Save sample images every 200 batches
        if batch_idx % 200 == 0:
            img_dir = config.ROOT_FOLDER + "output/images"
            os.makedirs(img_dir, exist_ok=True)
            tag = f"stage{stage+1}_epoch{epoch}_batch{batch_idx}"
            plt.imsave(f"{img_dir}/hr_{tag}.png",    hr.cpu().numpy()[0][0])
            plt.imsave(f"{img_dir}/lr_{tag}.png",    lr.cpu().numpy()[0][0])
            plt.imsave(f"{img_dir}/output_{tag}.png", outputs.detach().cpu().numpy()[0][0])

    return epoch_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage: int,
    previous_models: List[nn.Module],
    previous_epsilons: List[float],
    epsilon: float
) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)

            # ---- only change from train.py: same normalised residual target ----
            target = compute_target(lr, hr, previous_models, previous_epsilons, epsilon)

            outputs = model(lr)
            loss = criterion(outputs, target)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# ---------------------------------------------------------------------------
# Unchanged from train.py
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    train_loss: float, val_loss: float, filepath: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, filepath: str):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


# ---------------------------------------------------------------------------
# Inference: reconstruct HR from all stages
# HR_hat = sum_i( eps_i * u_i(LR) )
# ---------------------------------------------------------------------------
def multistage_predict(
    lr: torch.Tensor,
    models: List[nn.Module],
    epsilons: List[float]
) -> torch.Tensor:
    prediction = None
    for model, eps in zip(models, epsilons):
        model.eval()
        with torch.no_grad():
            out = model(lr)
        prediction = eps * out if prediction is None else prediction + eps * out
    return prediction


# ---------------------------------------------------------------------------
# Main: train.py's main() with an outer stage loop added
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(config.ROOT_FOLDER + "output/logs", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/weights", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/images", exist_ok=True)

    dataset = SRTinyDataset(
        hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    train_loader, val_loader = get_data_loaders(
        dataset, batch_size=config.BATCH_SIZE)

    trained_models: List[nn.Module] = []
    epsilons: List[float] = []

    log_path = config.ROOT_FOLDER + "output/logs/MSNN_loss.txt"

    # -----------------------------------------------------------------------
    # Outer stage loop — everything inside is identical to train.py
    # -----------------------------------------------------------------------
    for stage in range(NUM_STAGES):
        print(f"\n{'='*60}\n  STAGE {stage+1} / {NUM_STAGES}\n{'='*60}")

        # Compute epsilon_n = RMS of the residual this stage must learn
        print("Computing residual RMS …")
        epsilon = compute_rms(train_loader, config.DEVICE, trained_models, epsilons)
        epsilons.append(epsilon)
        print(f"  epsilon_{stage+1} = {epsilon:.6e}")

        # Fresh model for this stage (same as train.py)
        model = SRCNN(in_channels=2, out_channels=2)
        model = torch.nn.DataParallel(model)
        model = model.to(config.DEVICE)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float('inf')

        # Inner epoch loop — identical structure to train.py
        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, config.DEVICE,
                epoch + 1, stage, trained_models, epsilons[:-1], epsilon)

            val_loss = validate(
                model, val_loader, criterion, config.DEVICE,
                stage, trained_models, epsilons[:-1], epsilon)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Stage [{stage+1}/{NUM_STAGES}] | Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr}")

            with open(log_path, "a") as f:
                f.write(f"Stage [{stage+1}/{NUM_STAGES}] | Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"LR: {current_lr} | epsilon: {epsilon:.6e}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss,
                                config.ROOT_FOLDER + f'output/weights/MSNN_stage{stage+1}_best.pth')

            if (epoch + 1) % 50 == 0:
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss,
                                config.ROOT_FOLDER + f'output/weights/MSNN_stage{stage+1}_epoch{epoch+1}.pth')

        # Freeze and bank this stage
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        trained_models.append(model)

        print(f"\nStage {stage+1} done. Best val loss: {best_val_loss:.6f} | epsilon: {epsilon:.6e}")

    # Save epsilons for inference
    torch.save({'epsilons': epsilons},
               config.ROOT_FOLDER + 'output/weights/MSNN_epsilons.pth')
    print("\nMultistage training complete.")
    print(f"Epsilons: {[f'{e:.6e}' for e in epsilons]}")


if __name__ == "__main__":
    main()