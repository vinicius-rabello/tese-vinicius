import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from models.CycleGAN.discriminator_model import Discriminator
from models.CycleGAN.generator_model import Generator
from models.CycleGAN import config
import os
from typing import Tuple
from datasets.super_res_dataset import SuperResDataset


def get_data_loaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 1,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets and returns their DataLoaders.
    """
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    return train_loader, val_loader


def train_fn(disc_HR, disc_LR, gen_HR, gen_LR, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch):
    gen_HR.train()
    gen_LR.train()
    disc_HR.train()
    disc_LR.train()
    
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    epoch_cycle_loss = 0.0
    
    loop = tqdm(loader, leave=True, desc=f"Epoch {epoch}")
    
    for idx, (lr, hr) in enumerate(loop):
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)

        # Train Discriminators HR and LR
        with torch.cuda.amp.autocast():
            fake_hr = gen_HR(lr)
            D_HR_real = disc_HR(hr)
            D_HR_fake = disc_HR(fake_hr.detach())
            D_HR_real_loss = mse(D_HR_real, torch.ones_like(D_HR_real))
            D_HR_fake_loss = mse(D_HR_fake, torch.zeros_like(D_HR_fake))
            D_HR_loss = D_HR_real_loss + D_HR_fake_loss

            fake_lr = gen_LR(hr)
            D_LR_real = disc_LR(lr)
            D_LR_fake = disc_LR(fake_lr.detach())
            D_LR_real_loss = mse(D_LR_real, torch.ones_like(D_LR_real))
            D_LR_fake_loss = mse(D_LR_fake, torch.zeros_like(D_LR_fake))
            D_LR_loss = D_LR_real_loss + D_LR_fake_loss

            D_loss = (D_HR_loss + D_LR_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators HR and LR
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_HR_fake = disc_HR(fake_hr)
            D_LR_fake = disc_LR(fake_lr)
            loss_G_HR = mse(D_HR_fake, torch.ones_like(D_HR_fake))
            loss_G_LR = mse(D_LR_fake, torch.ones_like(D_LR_fake))

            # Cycle loss
            cycle_lr = gen_LR(fake_hr)
            cycle_hr = gen_HR(fake_lr)
            cycle_lr_loss = L1(lr, cycle_lr)
            cycle_hr_loss = L1(hr, cycle_hr)

            # Identity loss (optional, can be weighted differently)
            identity_hr = gen_HR(hr)
            identity_lr = gen_LR(lr)
            identity_hr_loss = L1(hr, identity_hr)
            identity_lr_loss = L1(lr, identity_lr)

            # Total generator loss
            G_loss = (
                loss_G_HR
                + loss_G_LR
                + cycle_hr_loss * config.LAMBDA_CYCLE
                + cycle_lr_loss * config.LAMBDA_CYCLE
                + identity_hr_loss * config.LAMBDA_IDENTITY
                + identity_lr_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Accumulate losses
        epoch_G_loss += G_loss.item()
        epoch_D_loss += D_loss.item()
        epoch_cycle_loss += (cycle_hr_loss.item() + cycle_lr_loss.item()) / 2
        
        # Update progress bar
        loop.set_postfix(G_loss=G_loss.item(), D_loss=D_loss.item())

        if idx % 200 == 0:
            os.makedirs(config.ROOT_FOLDER + "output/images", exist_ok=True)
            save_image(fake_hr, config.ROOT_FOLDER + f"output/images/fake_hr_epoch{epoch}_batch{idx}.png")
            save_image(fake_lr, config.ROOT_FOLDER + f"output/images/fake_lr_epoch{epoch}_batch{idx}.png")
            save_image(hr, config.ROOT_FOLDER + f"output/images/real_hr_epoch{epoch}_batch{idx}.png")
            save_image(lr, config.ROOT_FOLDER + f"output/images/real_lr_epoch{epoch}_batch{idx}.png")
    
    return epoch_G_loss / len(loader), epoch_D_loss / len(loader), epoch_cycle_loss / len(loader)


def validate(gen_HR, gen_LR, val_loader, L1, device):
    """
    Evaluates CycleGAN on validation set using cycle consistency loss.
    """
    gen_HR.eval()
    gen_LR.eval()
    val_cycle_loss = 0.0
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Cycle loss
            fake_hr = gen_HR(lr)
            cycle_lr = gen_LR(fake_hr)
            cycle_lr_loss = L1(lr, cycle_lr)
            
            fake_lr = gen_LR(hr)
            cycle_hr = gen_HR(fake_lr)
            cycle_hr_loss = L1(hr, cycle_hr)
            
            val_cycle_loss += (cycle_lr_loss.item() + cycle_hr_loss.item()) / 2
    
    return val_cycle_loss / len(val_loader)


def save_checkpoint(gen_HR, gen_LR, disc_HR, disc_LR, opt_gen, opt_disc, 
                   epoch, train_losses, val_loss, filepath):
    """
    Saves CycleGAN checkpoint with all models and optimizers.
    """
    checkpoint = {
        'epoch': epoch,
        'gen_HR_state_dict': gen_HR.state_dict(),
        'gen_LR_state_dict': gen_LR.state_dict(),
        'disc_HR_state_dict': disc_HR.state_dict(),
        'disc_LR_state_dict': disc_LR.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
        'G_loss': train_losses[0],
        'D_loss': train_losses[1],
        'cycle_loss': train_losses[2],
        'val_cycle_loss': val_loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(gen_HR, gen_LR, disc_HR, disc_LR, opt_gen, opt_disc, filepath):
    """
    Loads CycleGAN checkpoint.
    """
    checkpoint = torch.load(filepath)
    gen_HR.load_state_dict(checkpoint['gen_HR_state_dict'])
    gen_LR.load_state_dict(checkpoint['gen_LR_state_dict'])
    disc_HR.load_state_dict(checkpoint['disc_HR_state_dict'])
    disc_LR.load_state_dict(checkpoint['disc_LR_state_dict'])
    opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
    return checkpoint['epoch'], checkpoint['G_loss'], checkpoint['D_loss'], checkpoint['cycle_loss'], checkpoint['val_cycle_loss']


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create output directories
    os.makedirs(config.ROOT_FOLDER + "output/logs", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/weights", exist_ok=True)
    os.makedirs(config.ROOT_FOLDER + "output/images", exist_ok=True)
    
    # Initialize models
    disc_LR = Discriminator(in_channels=2).to(config.DEVICE)
    disc_HR = Discriminator(in_channels=2).to(config.DEVICE)
    gen_LR = Generator(in_channels=2).to(config.DEVICE)
    gen_HR = Generator(in_channels=2).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_LR.parameters()) + list(disc_HR.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_LR.parameters()) + list(gen_HR.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = SuperResDataset(hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=config.BATCH_SIZE)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    # Track best model
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        G_loss, D_loss, cycle_loss = train_fn(
            disc_HR, disc_LR, gen_HR, gen_LR, train_loader, 
            opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch+1
        )
        
        # Validate
        val_cycle_loss = validate(gen_HR, gen_LR, val_loader, L1, config.DEVICE)
        
        # Log results
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
              f"G_loss: {G_loss:.6f}, D_loss: {D_loss:.6f}, "
              f"Cycle: {cycle_loss:.6f}, Val_Cycle: {val_cycle_loss:.6f}")
        
        with open(config.ROOT_FOLDER + "output/logs/CycleGAN_loss.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
                   f"G_loss: {G_loss:.4f}, D_loss: {D_loss:.4f}, "
                   f"Cycle: {cycle_loss:.4f}, Val_Cycle: {val_cycle_loss:.4f}\n")
        
        # Save best model
        if val_cycle_loss < best_val_loss:
            best_val_loss = val_cycle_loss
            save_checkpoint(gen_HR, gen_LR, disc_HR, disc_LR, opt_gen, opt_disc,
                          epoch+1, (G_loss, D_loss, cycle_loss), val_cycle_loss,
                          config.ROOT_FOLDER + 'output/weights/CycleGAN_best.pth')
        
        # Save checkpoint every 2 epochs
        if (epoch+1) % 2 == 0:
            save_checkpoint(gen_HR, gen_LR, disc_HR, disc_LR, opt_gen, opt_disc,
                          epoch+1, (G_loss, D_loss, cycle_loss), val_cycle_loss,
                          config.ROOT_FOLDER + f'output/weights/CycleGAN_epoch{epoch+1}.pth')


if __name__ == "__main__":
    main()