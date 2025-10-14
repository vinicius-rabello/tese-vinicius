import torch
from dataset import cycleGANDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import config

def train_fn(disc_HR, disc_LR, gen_HR, gen_LR, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
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

        if idx % 200 == 0:
            save_image(fake_hr, f"saved_images/fake_hr_{idx}.png")
            save_image(fake_lr, f"saved_images/fake_lr_{idx}.png")
            save_image(hr, f"saved_images/real_hr_{idx}.png")
            save_image(lr, f"saved_images/real_lr_{idx}.png")
    

def main():
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

    dataset = cycleGANDataset(hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_HR, disc_LR, gen_HR, gen_LR, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

if __name__ == "__main__":
    main()