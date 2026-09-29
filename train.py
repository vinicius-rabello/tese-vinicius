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
import csv
import json
import time
from datetime import datetime

import numpy as np
from torch.utils.data import Subset
from datasets.super_res_dataset import SuperResDataset
from losses.losses import FluidCompositeLoss
from utils import reproducibility as repro

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

    def state_dict(self):
        return {'counter': self.counter, 'best_loss': self.best_loss, 'early_stop': self.early_stop}

    def load_state_dict(self, state):
        self.counter = state['counter']
        self.best_loss = state['best_loss']
        self.early_stop = state['early_stop']


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
    epoch_losses = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0, 'kinetic': 0.0, 'weighted_kinetic': 0.0}
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
        
        # Save a sample subplot grid periodically (first batch of every Nth epoch)
        image_save_interval = config.get('IMAGE_SAVE_INTERVAL', 10)
        if batch_idx == 0 and epoch % image_save_interval == 0:
            os.makedirs(os.path.join(config['RUN_DIR'], "images"), exist_ok=True)
            
            # Extract first image in the batch, first channel
            lr_img = lr.detach().cpu().numpy()[0][0]
            hr_img = hr.detach().cpu().numpy()[0][0]
            pred_img = outputs.detach().cpu().numpy()[0][0]
            
            # Create 1x3 subplot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Dynamically format loss string based on chosen LOSS_MODE
            loss_str = f"Total: {batch_metrics['total']:.4f} | MSE: {batch_metrics['mse']:.4f}"
            if config['LOSS_MODE'] in (2, 3):
                loss_str += f" | LPIPS: {batch_metrics['lpips']:.4f}"
            if config['LOSS_MODE'] == 3:
                loss_str += f" | Kin: {batch_metrics['kinetic']:.4f}"
            if config['LOSS_MODE'] == 4:
                loss_str += f" | WKin: {batch_metrics['weighted_kinetic']:.4f}"
                
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
            save_path = os.path.join(config['RUN_DIR'], f"images/sample_epoch{epoch}_batch{batch_idx}.png")
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
    val_losses = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0, 'kinetic': 0.0, 'weighted_kinetic': 0.0}
    
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


# ---------------------------------------------------------------------------
# Run management: stable run directory, full checkpoints, auto-resume.
# ---------------------------------------------------------------------------
COMPLETED_STATUSES = ('completed_early_stop', 'completed_max_epochs')
CSV_FIELDS = ['epoch', 'lr', 'epoch_sec', 'train_total', 'train_mse', 'train_lpips',
              'val_total', 'val_mse', 'val_lpips', 'monitor', 'best_monitor', 'es_counter']


def run_id_for(config, config_path):
    name = config.get('RUN_NAME') or os.path.splitext(os.path.basename(config_path))[0]
    return f"{name}_seed{config.get('SEED', 0)}"


def dataset_constants(hr_data):
    """Fixed, data-derived constants shared by every run (computed on the full
    training HR set, u and v together): the PSNR/SSIM reference range R and
    the LPIPS input scale that maps the fields into [-1, 1]."""
    return {
        'psnr_ssim_R': float(hr_data.max() - hr_data.min()),
        'lpips_scale': float(np.abs(hr_data).max()),
    }


def atomic_torch_save(obj, path):
    # A kill mid-write must never leave a truncated checkpoint behind.
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def truncate_csv(path, max_epoch):
    """On resume, drop log rows for epochs the checkpoint doesn't cover (a run
    killed between the CSV append and the checkpoint save would otherwise log
    that epoch twice)."""
    if not os.path.exists(path):
        return
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if int(r['epoch']) <= max_epoch]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, row):
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(config, config_path, deadline=None):
    """Trains one run. Returns its final status string. `deadline` is an
    absolute time.time() value from --max-hours; when the next epoch would
    cross it, the run stops cleanly (its last.pt is already saved)."""
    seed = int(config.get('SEED', 0))
    deterministic = config.get('DETERMINISTIC', True)
    deterministic_algorithms = config.get('DETERMINISTIC_ALGORITHMS', True)
    repro.set_seed(seed, deterministic, deterministic_algorithms)

    current_model = config.get('MODEL_NAME', 'dscms').lower()
    upper_model = current_model.upper()

    run_id = run_id_for(config, config_path)
    run_dir = os.path.join(config.get('RESULTS_DIR', 'results'), run_id)
    config['RUN_DIR'] = run_dir
    os.makedirs(run_dir, exist_ok=True)
    paths = {name: os.path.join(run_dir, name) for name in
             ('config.json', 'summary.json', 'train_log.csv', 'last.pt', 'best.pt')}

    if os.path.exists(paths['summary.json']):
        status = read_json(paths['summary.json']).get('status')
        if status in COMPLETED_STATUSES:
            print(f"[{run_id}] already {status} -- skipping.")
            return status

    print(f"Run: {run_id} -> {run_dir}")

    train_dataset = SuperResDataset(
        hr_files=[f"./hycom_numpy/hycom_{year}.npy" for year in range(2001, 2020)],
        downsample_factor=1,
        freq_threshold=10,
    )
    val_dataset = SuperResDataset(
        hr_files=["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"],
        downsample_factor=1,
        freq_threshold=10,
    )
    constants = dataset_constants(train_dataset.hr_data)
    print(f"PSNR/SSIM R = {constants['psnr_ssim_R']:.6f} m/s | LPIPS scale = {constants['lpips_scale']:.6f} m/s")

    # Optional subsets, for quick sanity runs only (constants above always
    # come from the full training set).
    if config.get('SUBSET_TRAIN'):
        train_dataset = Subset(train_dataset, range(int(config['SUBSET_TRAIN'])))
    if config.get('SUBSET_VAL'):
        val_dataset = Subset(val_dataset, range(int(config['SUBSET_VAL'])))

    loader_generator = repro.make_generator(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
        num_workers=config['NUM_WORKERS'], pin_memory=True,
        worker_init_fn=repro.seed_worker, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
        num_workers=config['NUM_WORKERS'], pin_memory=True,
        worker_init_fn=repro.seed_worker, generator=repro.make_generator(seed),
    )

    print(f"Initializing model: {upper_model}")
    model = create_model(current_model, in_channels=2, out_channels=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(config['DEVICE'])

    criterion = FluidCompositeLoss(
        mode=config['LOSS_MODE'], weights=config['LOSS_WEIGHTS'], device=config['DEVICE'], img_size=128)
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

    git = repro.git_info()
    run_config = {k: v for k, v in config.items() if k != 'RUN_DIR'}
    if os.path.exists(paths['config.json']):
        meta = read_json(paths['config.json'])
        meta.setdefault('resumes', []).append({'time': datetime.now().isoformat(), 'git': git})
    else:
        meta = {
            'run_id': run_id, 'config_file': config_path, 'config': run_config, 'seed': seed,
            'n_params': n_params, 'git': git, 'libraries': repro.library_versions(),
            'deterministic': deterministic, 'deterministic_algorithms': deterministic_algorithms,
            'created': datetime.now().isoformat(), **constants,
        }
    write_json(meta, paths['config.json'])

    start_epoch, best_monitor, best_epoch, train_time_sec = 0, float('inf'), 0, 0.0
    if os.path.exists(paths['last.pt']):
        # Load on CPU: RNG states must stay CPU ByteTensors; load_state_dict moves
        # weights and optimizer state onto the model's device itself.
        ckpt = torch.load(paths['last.pt'], map_location='cpu', weights_only=False)
        unwrap(model).load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        early_stopper.load_state_dict(ckpt['early_stopper'])
        repro.set_rng_state(ckpt['rng_state'], loader_generator)
        start_epoch, best_monitor = ckpt['epoch'], ckpt['best_monitor']
        best_epoch, train_time_sec = ckpt['best_epoch'], ckpt['train_time_sec']
        truncate_csv(paths['train_log.csv'], start_epoch)
        print(f"Resumed from epoch {start_epoch} (best {best_monitor:.6g} @ epoch {best_epoch}).")
    else:
        print("Starting training from scratch.")

    def finish(status):
        summary = {
            'run_id': run_id, 'status': status, 'epochs_completed': epoch_done,
            'best_epoch': best_epoch, 'best_monitor': best_monitor, 'monitor_metric': 'val_total',
            'final_lr': optimizer.param_groups[0]['lr'], 'train_hours': train_time_sec / 3600,
            'n_params': n_params, 'finished': datetime.now().isoformat(),
        }
        write_json(summary, paths['summary.json'])
        print(f"[{run_id}] {status} after {epoch_done} epochs "
              f"(best {best_monitor:.6g} @ epoch {best_epoch}, {train_time_sec / 3600:.2f}h).")
        return status

    epoch_done = start_epoch
    if early_stopper.early_stop:
        return finish('completed_early_stop')

    for epoch in range(start_epoch, config['NUM_EPOCHS']):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, config['DEVICE'], epoch + 1, config, upper_model)
        val_metrics = validate(model, val_loader, criterion, config['DEVICE'])

        monitor = val_metrics['total']
        scheduler.step(monitor)
        epoch_sec = time.perf_counter() - epoch_start
        train_time_sec += epoch_sec
        epoch_done = epoch + 1

        if monitor < best_monitor:
            best_monitor, best_epoch = monitor, epoch_done
            atomic_torch_save({'model_state_dict': unwrap(model).state_dict(), 'epoch': epoch_done,
                               'val_metrics': val_metrics}, paths['best.pt'])
        early_stopper(monitor)

        atomic_torch_save({
            'epoch': epoch_done,
            'model_state_dict': unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopper': early_stopper.state_dict(),
            'best_monitor': best_monitor,
            'best_epoch': best_epoch,
            'train_time_sec': train_time_sec,
            'rng_state': repro.get_rng_state(loader_generator),
        }, paths['last.pt'])

        lr_now = optimizer.param_groups[0]['lr']
        append_csv(paths['train_log.csv'], {
            'epoch': epoch_done, 'lr': lr_now, 'epoch_sec': round(epoch_sec, 2),
            'train_total': train_metrics['total'], 'train_mse': train_metrics['mse'],
            'train_lpips': train_metrics['lpips'],
            'val_total': val_metrics['total'], 'val_mse': val_metrics['mse'], 'val_lpips': val_metrics['lpips'],
            'monitor': monitor, 'best_monitor': best_monitor, 'es_counter': early_stopper.counter,
        })
        print(f"[{run_id}] epoch {epoch_done}/{config['NUM_EPOCHS']} ({epoch_sec:.1f}s, total {train_time_sec / 3600:.2f}h) "
              f"lr {lr_now:.2e} | train mse {train_metrics['mse']:.6g} lpips {train_metrics['lpips']:.4g} | "
              f"val mse {val_metrics['mse']:.6g} lpips {val_metrics['lpips']:.4g} | "
              f"best {best_monitor:.6g} @ {best_epoch} | es {early_stopper.counter}/{early_stopper.patience}")

        if early_stopper.early_stop:
            return finish('completed_early_stop')
        # Stop before an epoch that would run past the budget, not after it.
        if deadline is not None and time.time() + epoch_sec > deadline:
            return finish('interrupted_max_hours')

    return finish('completed_max_epochs')


if __name__ == "__main__":
    import argparse
    import yaml
    import traceback

    # One or more YAML configs run as a queue. Relaunching the same queue
    # skips finished runs and resumes an interrupted one from its last.pt.
    parser = argparse.ArgumentParser(description="Train Super Resolution Models")
    parser.add_argument('--config', type=str, required=True, nargs='+',
                        help="Path(s) to one or more YAML config files")
    parser.add_argument('--max-hours', type=float, default=None,
                        help="Wall-clock budget for the whole queue; stops cleanly before exceeding it")
    args = parser.parse_args()

    queue_start = time.time()
    deadline = queue_start + args.max_hours * 3600 if args.max_hours else None

    print(f"Queued {len(args.config)} run(s):")
    for config_path in args.config:
        print(f"  - {config_path}")
    if deadline:
        print(f"Budget: {args.max_hours}h (until {datetime.fromtimestamp(deadline).isoformat(timespec='minutes')})")

    results = {}
    for i, config_path in enumerate(args.config, start=1):
        print(f"\n{'=' * 70}\nQueue [{i}/{len(args.config)}]: {config_path}\n{'=' * 70}")
        with open(config_path, 'r') as file:
            experiment_config = yaml.safe_load(file)
        try:
            results[config_path] = main(experiment_config, config_path, deadline)
        except Exception:
            print(f"Run failed for {config_path} -- skipping to the next queued config.")
            traceback.print_exc()
            results[config_path] = 'failed'
        finally:
            torch.cuda.empty_cache()
        if results[config_path] == 'interrupted_max_hours':
            print("Time budget reached -- the remaining runs stay pending; relaunch the same queue to continue.")
            break

    print(f"\nQueue stopped after {(time.time() - queue_start) / 3600:.2f}h.")
    for config_path in args.config:
        print(f"  {results.get(config_path, 'pending'):>24}  {config_path}")
