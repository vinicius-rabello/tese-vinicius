import os
import math
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.srno.model import EDSREncoder, SRONET
from models.srno import config

sys.path.append('./')
from datasets.sr_tiny_dataset_srno import SRTinyDataset


# # =============================================================================
# # CONFIG
# # =============================================================================

# CFG = {
#     # --- data ---
#     'hr_files':  ['data/100/window_2003.npy'],
#     'lr_size':   8,
#     'hr_size':   32,
#     'val_split': 0.2,

#     # --- normalisation — set mean=0 / std=1 to disable ---
#     'inp_mean': [0.0, 0.0],
#     'inp_std':  [1.0, 1.0],
#     'gt_mean':  [0.0, 0.0],
#     'gt_std':   [1.0, 1.0],

#     # --- model ---
#     'n_resblocks': 16,
#     'n_feats':     64,
#     'res_scale':   1.0,
#     'width':       256,
#     'blocks':      16,

#     # --- training ---
#     'batch_size':    1,
#     'lr':            3e-4,
#     'weight_decay':  1e-4,
#     'epoch_max':     500,
#     'warmup_epochs': 50,
#     'epoch_save':    50,   # numbered checkpoint every N epochs

#     # --- paths ---
#     'root_folder': './models/srno',
# }


# =============================================================================
# HELPERS
# =============================================================================

def normalise(t, mean, std, device):
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=device).view(1, -1, 1, 1)
    return (t - m) / s


class Averager:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, val):
        self.total += val
        self.count += 1

    def item(self):
        return self.total / self.count if self.count else 0.0


def calc_psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2).item()
    return 10 * math.log10(max_val ** 2 / (mse + 1e-8))


# =============================================================================
# SCHEDULER  (linear warmup → cosine decay)
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            factor = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# =============================================================================
# DATA
# =============================================================================

def get_data_loaders():
    dataset    = SRTinyDataset(config.HR_FILES)
    train_size = int((1 - config.VAL_SPLIT) * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Dataset: {len(dataset)} total  |  train {train_size}  |  val {val_size}')
    return train_loader, val_loader


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0.0
    img_dir    = os.path.join('./models/srno', 'output', 'images')
    loop       = tqdm(train_loader, leave=True, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(loop):
        inp   = normalise(batch['inp'].to(device),  config.INP_MEAN, config.INP_STD, device)
        gt    = normalise(batch['gt'].to(device),   config.GT_MEAN,  config.GT_STD,  device)
        coord = batch['coord'].to(device)
        cell  = batch['cell'].to(device)

        pred = model(inp, coord, cell)
        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # save u-channel images every 200 batches
        if batch_idx % 200 == 0:
            os.makedirs(img_dir, exist_ok=True)
            plt.imsave(os.path.join(img_dir, f'hr_epoch{epoch}_batch{batch_idx}.png'),
                       gt.cpu().numpy()[0][0])
            plt.imsave(os.path.join(img_dir, f'lr_epoch{epoch}_batch{batch_idx}.png'),
                       inp.cpu().numpy()[0][0])
            plt.imsave(os.path.join(img_dir, f'pred_epoch{epoch}_batch{batch_idx}.png'),
                       pred.detach().cpu().numpy()[0][0])

    return epoch_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    loss_avg = Averager()
    psnr_avg = Averager()

    for batch in val_loader:
        inp   = normalise(batch['inp'].to(device),  config.INP_MEAN, config.INP_STD, device)
        gt    = normalise(batch['gt'].to(device),   config.GT_MEAN,  config.GT_STD,  device)
        coord = batch['coord'].to(device)
        cell  = batch['cell'].to(device)

        pred = model(inp, coord, cell)
        loss_avg.add(criterion(pred, gt).item())
        psnr_avg.add(calc_psnr(pred, gt))

    model.train()
    return loss_avg.item(), psnr_avg.item()


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss':           train_loss,
        'val_loss':             val_loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['train_loss'], ckpt['val_loss']


# =============================================================================
# MAIN
# =============================================================================

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # output dirs
    for sub in ['output/logs', 'output/weights', 'output/images']:
        os.makedirs(os.path.join(config.ROOT_FOLDER, sub), exist_ok=True)

    # data
    train_loader, val_loader = get_data_loaders()

    # model
    encoder = EDSREncoder(n_resblocks=config.N_RESBLOCKS, n_feats=config.N_FEATS,
                          res_scale=config.RES_SCALE, n_colors=2)
    model   = SRONET(encoder=encoder, width=config.WIDTH, blocks=config.BLOCKS, n_out=2)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # optimiser + scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=config.WARMUP_EPOCHS,
                                      total_epochs=config.NUM_EPOCHS)
    criterion  = nn.L1Loss()
    log_path   = os.path.join(config.ROOT_FOLDER, 'output', 'logs', 'SRONET_loss.txt')
    w_dir      = os.path.join(config.ROOT_FOLDER, 'output', 'weights')
    best_val   = float('inf')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)

        val_loss, val_psnr = validate(
            model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch}/{config.NUM_EPOCHS}]  '
              f'train={train_loss:.6f}  val={val_loss:.6f}  '
              f'psnr={val_psnr:.2f}dB  lr={current_lr:.2e}')

        with open(log_path, 'a') as f:
            f.write(f'Epoch [{epoch}/{config.NUM_EPOCHS}]  '
                    f'train={train_loss:.4f}  val={val_loss:.4f}  '
                    f'psnr={val_psnr:.2f}dB  lr={current_lr:.2e}\n')

        # always save latest
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                        os.path.join(w_dir, 'SRONET_last.pth'))

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                            os.path.join(w_dir, 'SRONET_best.pth'))

        # periodic snapshot
        if epoch % config.EPOCH_SAVE == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                            os.path.join(w_dir, f'SRONET_epoch{epoch}.pth'))


if __name__ == '__main__':
    main()