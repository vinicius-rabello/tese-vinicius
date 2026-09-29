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
from datasets.super_res_dataset import SuperResDataset, NOISE_SEED_OFFSET
from losses.losses import MseLpipsLoss
from utils import reproducibility as repro

class EarlyStopping:
    """Counts epochs without a significant improvement of the monitored metric.

    relative=False (default, used by train_multistage.py): an epoch improves if
    val < best - min_delta. relative=True: it improves if
    val < best * (1 - min_delta), i.e. the relative drop exceeds min_delta.
    Stopping is never triggered before epoch min_epochs; the counter still
    runs, so a plateau that started earlier stops right at min_epochs."""

    def __init__(self, patience=15, min_delta=1e-4, relative=False, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.relative = relative
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def improves(self, val_loss):
        if self.relative:
            return val_loss < self.best_loss * (1 - self.min_delta)
        return val_loss < (self.best_loss - self.min_delta)

    def __call__(self, val_loss, epoch=None):
        if self.improves(val_loss):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience and (epoch is None or epoch >= self.min_epochs):
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
    sums, n_samples = {}, 0
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
        
        # Sample-weighted sums, so the last (smaller) batch doesn't skew the epoch mean
        n_samples += lr.shape[0]
        for key, value in batch_metrics.items():
            sums[key] = sums.get(key, 0.0) + value * lr.shape[0]
        
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
            
            loss_str = f"Total: {batch_metrics['total']:.4g} | MSE: {batch_metrics['mse']:.4g}"
            if config['LAMBDA'] > 0:
                loss_str += f" | LPIPS: {batch_metrics['lpips']:.4g}"

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
            
    return {key: value / n_samples for key, value in sums.items()}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    model.eval()
    sums, n_samples = {}, 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            outputs = model(lr)
            # LPIPS is always logged on validation, whatever lambda is.
            _, batch_metrics = criterion(outputs, hr, compute_lpips=True)
            n_samples += lr.shape[0]
            for key, value in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + value * lr.shape[0]
    return {key: value / n_samples for key, value in sums.items()}


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
              'val_total', 'val_mse', 'val_lpips', 'best_val_mse', 'es_metric', 'es_best', 'es_counter',
              'noise_std']


def residual_flag(config):
    # Required and explicit: blocks A and D differ only in this flag, so a
    # silent default could put an ablation run under a main-configuration name.
    value = config.get('RESIDUAL')
    if not isinstance(value, bool):
        raise ValueError(f"RESIDUAL must be set to true or false in the config (got {value!r}).")
    return value


def lambda_value(config):
    # Required and explicit, like RESIDUAL: runs differing only in lambda must
    # never share a name. 0 means plain MSE.
    value = config.get('LAMBDA')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"LAMBDA must be set to a number >= 0 in the config (got {value!r}).")
    return float(value)


def noise_std_value(config):
    # Observation noise on the LR input (m/s); 0 (default) is the noiseless regime.
    value = config.get('NOISE_STD', 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"NOISE_STD must be a number >= 0 (got {value!r}).")
    return float(value)


def run_id_for(config, config_path):
    name = config.get('RUN_NAME') or os.path.splitext(os.path.basename(config_path))[0]
    skip = 'res' if residual_flag(config) else 'nores'
    noise = noise_std_value(config)
    noise_tag = f"_noise{noise:g}" if noise > 0 else ""
    return f"{name}_{skip}_lam{lambda_value(config):g}{noise_tag}_seed{config.get('SEED', 0)}"


def dataset_constants(hr_data):
    """Fixed, data-derived constants shared by every run (computed on the full
    training HR set, u and v together): the PSNR/SSIM reference range R and
    the LPIPS input scale that maps the fields into [-1, 1]."""
    return {
        'psnr_ssim_R': float(hr_data.max() - hr_data.min()),
        'hr_train_min': float(hr_data.min()),  # fixed SSIM offset (evaluation/spectral_metrics.ssim)
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

    noise_std = noise_std_value(config)
    train_dataset = SuperResDataset(
        hr_files=[f"./hycom_numpy/hycom_{year}.npy" for year in range(2001, 2020)],
        downsample_factor=1,
        freq_threshold=10,
        noise_std=noise_std, noise_mode='random',
    )
    val_dataset = SuperResDataset(
        hr_files=["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"],
        downsample_factor=1,
        freq_threshold=10,
        noise_std=noise_std, noise_mode='fixed', noise_seed_offset=NOISE_SEED_OFFSET['val'],
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
    residual = residual_flag(config)
    model = create_model(current_model, in_channels=2, out_channels=2, residual=residual)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(config['DEVICE'])

    lam = lambda_value(config)
    config['LAMBDA'] = lam
    criterion = MseLpipsLoss(lam, constants['lpips_scale'], device=config['DEVICE'], net='vgg')
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Same stopping rule for every loss: relative improvement of the
    # validation MSE. The config keys exist only so quick tests can shrink
    # them; the campaign uses the defaults.
    stopping = {
        'metric': 'val_mse',
        'relative_min_delta': float(config.get('ES_REL_DELTA', 1e-3)),
        'patience': int(config.get('ES_PATIENCE', 15)),
        'min_epochs': int(config.get('MIN_EPOCHS', 50)),
        'max_epochs': int(config.get('MAX_EPOCHS', 300)),
    }
    if 'NUM_EPOCHS' in config:
        print(f"Note: NUM_EPOCHS is ignored; training runs {stopping['min_epochs']}-{stopping['max_epochs']} epochs.")
    early_stopper = EarlyStopping(patience=stopping['patience'], min_delta=stopping['relative_min_delta'],
                                  relative=True, min_epochs=stopping['min_epochs'])

    git = repro.git_info()
    if git['dirty'] and not config.get('SUBSET_TRAIN') and not config.get('ALLOW_DIRTY'):
        raise RuntimeError("Tracked files have uncommitted changes; commit them before a real run "
                           "(or set ALLOW_DIRTY: true) so config.json records the exact code.")
    run_config = {k: v for k, v in config.items() if k != 'RUN_DIR'}
    if os.path.exists(paths['config.json']):
        meta = read_json(paths['config.json'])
        meta.setdefault('resumes', []).append({'time': datetime.now().isoformat(), 'git': git})
    else:
        meta = {
            'run_id': run_id, 'config_file': config_path, 'config': run_config, 'seed': seed,
            'model': current_model, 'residual': residual, 'lambda': lam, 'noise_std': noise_std,
            'loss': 'mse + lambda * lpips (lpips: vgg, fields / lpips_scale, channel repeated x3, mean of u and v)',
            'n_params': n_params, 'git': git, 'libraries': repro.library_versions(),
            'deterministic': deterministic, 'deterministic_algorithms': deterministic_algorithms,
            'created': datetime.now().isoformat(), 'stopping': stopping,
            'scheduler': {'type': 'ReduceLROnPlateau', 'metric': 'val_mse', 'patience': 5,
                          'factor': 0.5, 'threshold': 1e-4, 'threshold_mode': 'rel'},
            **constants,
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
            'best_epoch': best_epoch, 'best_val_mse': best_monitor, 'monitor_metric': 'val_mse',
            'final_lr': optimizer.param_groups[0]['lr'], 'train_hours': train_time_sec / 3600,
            'n_params': n_params, 'residual': residual, 'lambda': lam, 'noise_std': noise_std,
            'finished': datetime.now().isoformat(),
        }
        write_json(summary, paths['summary.json'])
        print(f"[{run_id}] {status} after {epoch_done} epochs "
              f"(best {best_monitor:.6g} @ epoch {best_epoch}, {train_time_sec / 3600:.2f}h).")
        return status

    epoch_done = start_epoch
    if early_stopper.early_stop:
        return finish('completed_early_stop')

    for epoch in range(start_epoch, stopping['max_epochs']):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, config['DEVICE'], epoch + 1, config, upper_model)
        val_metrics = validate(model, val_loader, criterion, config['DEVICE'])

        monitor = val_metrics['mse']
        scheduler.step(monitor)
        epoch_sec = time.perf_counter() - epoch_start
        train_time_sec += epoch_sec
        epoch_done = epoch + 1

        if monitor < best_monitor:
            best_monitor, best_epoch = monitor, epoch_done
            atomic_torch_save({'model_state_dict': unwrap(model).state_dict(), 'epoch': epoch_done,
                               'val_metrics': val_metrics}, paths['best.pt'])
        early_stopper(monitor, epoch_done)

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
            'best_val_mse': best_monitor, 'es_metric': stopping['metric'],
            'es_best': early_stopper.best_loss, 'es_counter': early_stopper.counter,
            'noise_std': noise_std,
        })
        print(f"[{run_id}] epoch {epoch_done}/{stopping['max_epochs']} ({epoch_sec:.1f}s, total {train_time_sec / 3600:.2f}h) "
              f"lr {lr_now:.2e} | train mse {train_metrics['mse']:.6g} lpips {train_metrics['lpips']:.4g} | "
              f"val mse {val_metrics['mse']:.6g} lpips {val_metrics['lpips']:.4g} | "
              f"best val_mse {best_monitor:.6g} @ {best_epoch} | es[val_mse] {early_stopper.counter}/{early_stopper.patience}")

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
