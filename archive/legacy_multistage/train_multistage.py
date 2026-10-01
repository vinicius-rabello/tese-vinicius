"""Stage-2 (multistage residual) training.

Loads a frozen, already-trained stage-1 checkpoint (any model_factory model,
any of its loss strategies), computes epsilon = RMS(HR - stage1(LR)) over the
training set, then trains a *fresh* copy of the same architecture (ordinary
init, plain MSE) to predict that normalized residual. At inference:

    final(LR) = stage1(LR) + epsilon * stage2(LR)

This intentionally does not touch train.py, model_factory.py, or losses.py --
it only imports generic, already-tested helpers from train.py (EarlyStopping,
save_checkpoint) and reuses model_factory/datasets exactly as train.py does.
See models/multistage.py for the MultistageModel wrapper used at eval time.
"""

import os
import time
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.model_factory import create_model
from models.multistage import MultistageModel
from datasets.super_res_dataset import SuperResDataset
from train import EarlyStopping, save_checkpoint


def load_stage1(model_name: str, checkpoint_path: str, device: str) -> nn.Module:
    """Instantiates model_name via the same factory train.py uses, loads the
    checkpoint (stripping a DataParallel "module." prefix if present), and
    freezes it.

    If checkpoint_path is itself a multistage checkpoint (has the
    "stage1_checkpoint"/"epsilon" keys save_stage2_checkpoint() adds), this
    recursively reconstructs that whole frozen MultistageModel first and
    returns it as the base instead -- this is what lets a stage-N config
    chain onto an existing stage-(N-1) checkpoint of arbitrary depth, just by
    pointing STAGE1_CHECKPOINT at it."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'stage1_checkpoint' in checkpoint:
        inner_stage1 = load_stage1(model_name, checkpoint['stage1_checkpoint'], device)

        inner_stage2 = create_model(model_name, in_channels=2, out_channels=2).to(device)
        cleaned = OrderedDict(
            (k[len('module.'):] if k.startswith('module.') else k, v)
            for k, v in checkpoint['model_state_dict'].items()
        )
        inner_stage2.load_state_dict(cleaned)

        model = MultistageModel(inner_stage1, inner_stage2, checkpoint['epsilon']).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    model = create_model(model_name, in_channels=2, out_channels=2).to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = OrderedDict(
        (k[len('module.'):] if k.startswith('module.') else k, v)
        for k, v in state_dict.items()
    )
    model.load_state_dict(cleaned)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def compute_epsilon(stage1: nn.Module, loader: DataLoader, device: str) -> float:
    """RMS of stage1's residual (HR - stage1(LR)) over every sample in loader."""
    stage1.eval()
    sq_sum, count = 0.0, 0
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Computing epsilon"):
            lr, hr = lr.to(device), hr.to(device)
            residual = hr - stage1(lr)
            sq_sum += (residual ** 2).sum().item()
            count += residual.numel()
    return (sq_sum / count) ** 0.5


def save_stage2_checkpoint(stage2, optimizer, epoch, train_loss, val_loss,
                            epsilon, stage1_checkpoint, filepath):
    """Reuses train.py's save_checkpoint, then adds the two multistage-only
    fields models/multistage.py's loader needs to reconstruct the full model."""
    save_checkpoint(stage2, optimizer, epoch, train_loss, val_loss, filepath)
    checkpoint = torch.load(filepath)
    checkpoint['epsilon'] = epsilon
    checkpoint['stage1_checkpoint'] = stage1_checkpoint
    torch.save(checkpoint, filepath)


def train_one_epoch_stage2(stage1, stage2, epsilon, train_loader, optimizer,
                            device, epoch, config, model_name_display, train_criterion):
    stage2.train()
    stage1.eval()
    mse_loss_fn = nn.MSELoss()  # always MSE, regardless of train_criterion -- this is
    # the fixed, cross-experiment-comparable "did it actually help" number used for
    # checkpoint selection/scheduling; train_criterion is the ablation variable.
    running_residual_loss = 0.0
    running_final_mse = 0.0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")

    for batch_idx, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(device), hr.to(device)

        with torch.no_grad():
            stage1_pred = stage1(lr)
            target_residual = (hr - stage1_pred) / epsilon

        stage2_pred = stage2(lr)
        residual_loss = train_criterion(stage2_pred, target_residual)

        optimizer.zero_grad()
        residual_loss.backward()
        torch.nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=0.5)
        optimizer.step()

        with torch.no_grad():
            final_pred = stage1_pred + epsilon * stage2_pred
            final_mse = mse_loss_fn(final_pred, hr)

        running_residual_loss += residual_loss.item()
        running_final_mse += final_mse.item()
        loop.set_postfix(residual_loss=residual_loss.item(), final_mse=final_mse.item())

        image_save_interval = config.get('IMAGE_SAVE_INTERVAL', 10)
        if batch_idx == 0 and epoch % image_save_interval == 0:
            os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/images"), exist_ok=True)

            lr_img = lr.detach().cpu().numpy()[0][0]
            hr_img = hr.detach().cpu().numpy()[0][0]
            stage1_img = stage1_pred.detach().cpu().numpy()[0][0]
            stage2_img = (epsilon * stage2_pred).detach().cpu().numpy()[0][0]
            final_img = final_pred.detach().cpu().numpy()[0][0]

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(
                f"Epoch {epoch} - Batch {batch_idx}\n"
                f"Residual loss: {residual_loss.item():.4f} | Final MSE: {final_mse.item():.6f}",
                fontsize=14, fontweight='bold',
            )
            panels = [
                ("Low Resolution", lr_img),
                (f"Stage 1 Pred ({model_name_display})", stage1_img),
                ("Stage 2 Pred (error map)", stage2_img),
                (f"Stage 1 + Stage 2 ({model_name_display})", final_img),
                ("High Resolution (GT)", hr_img),
            ]
            for ax, (title, img) in zip(axes, panels):
                im = ax.imshow(img, cmap='viridis')
                ax.set_title(title)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_path = os.path.join(config['ROOT_FOLDER'], f"output/images/sample_epoch{epoch}_batch{batch_idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

    n = len(train_loader)
    return {'residual_loss': running_residual_loss / n, 'final_mse': running_final_mse / n}


def validate_stage2(stage1, stage2, epsilon, val_loader, device, train_criterion):
    stage2.eval()
    stage1.eval()
    mse_loss_fn = nn.MSELoss()
    running_residual_loss = 0.0
    running_final_mse = 0.0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            stage1_pred = stage1(lr)
            target_residual = (hr - stage1_pred) / epsilon
            stage2_pred = stage2(lr)

            running_residual_loss += train_criterion(stage2_pred, target_residual).item()
            final_pred = stage1_pred + epsilon * stage2_pred
            running_final_mse += mse_loss_fn(final_pred, hr).item()

    n = len(val_loader)
    return {'residual_loss': running_residual_loss / n, 'final_mse': running_final_mse / n}


def main(config):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    current_model = config.get('MODEL_NAME', 'srcnn').lower()
    upper_model = current_model.upper()
    device = config['DEVICE']

    run_timestamp = datetime.now().strftime('%d_%m_%Y_%H%M%S')
    config['ROOT_FOLDER'] = os.path.join(config['ROOT_FOLDER'].rstrip('/'), run_timestamp) + '/'
    print(f"Run output folder: {config['ROOT_FOLDER']}")

    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/logs"), exist_ok=True)
    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/weights"), exist_ok=True)
    os.makedirs(os.path.join(config['ROOT_FOLDER'], "output/images"), exist_ok=True)

    train_dataset = SuperResDataset(
        hr_files=[f"./hycom_numpy/hycom_{year}.npy" for year in range(2001, 2020)],
        downsample_factor=1, freq_threshold=10,
    )
    val_dataset = SuperResDataset(
        hr_files=["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"],
        downsample_factor=1, freq_threshold=10,
    )
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                               num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                             num_workers=config['NUM_WORKERS'], pin_memory=True)

    loss_norm = config.get('STAGE2_LOSS_NORM', 'mse').lower()
    if loss_norm == 'l1':
        train_criterion = nn.L1Loss()
    elif loss_norm in ('mse', 'l2'):
        train_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown STAGE2_LOSS_NORM: {loss_norm!r}, expected 'mse' or 'l1'")
    print(f"Stage-2 training loss (against the normalized residual): {loss_norm}")

    stage1_checkpoint_path = config['STAGE1_CHECKPOINT']
    print(f"Loading + freezing stage-1 checkpoint: {stage1_checkpoint_path}")
    stage1 = load_stage1(current_model, stage1_checkpoint_path, device)

    print("Computing epsilon (RMS of stage-1's residual) over the training set...")
    epsilon = compute_epsilon(stage1, train_loader, device)
    print(f"epsilon = {epsilon:.6f}")

    with open(os.path.join(config['ROOT_FOLDER'], 'epsilon.txt'), 'w') as f:
        f.write(f"{epsilon}\n")

    # Fresh stage-2 model: same architecture, ordinary (model_factory default) init.
    stage2 = create_model(current_model, in_channels=2, out_channels=2).to(device)

    optimizer = optim.Adam(stage2.parameters(), lr=float(config['LEARNING_RATE']), weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

    best_val_final_mse = float('inf')
    cumulative_time_sec = 0.0

    log_file_path = os.path.join(config['ROOT_FOLDER'], f"output/logs/{upper_model}_multistage_loss.txt")
    best_weight_path = os.path.join(config['ROOT_FOLDER'], f"output/weights/{upper_model}_best.pth")

    for epoch in range(config['NUM_EPOCHS']):
        epoch_start_time = time.perf_counter()

        train_metrics = train_one_epoch_stage2(
            stage1, stage2, epsilon, train_loader, optimizer, device, epoch + 1, config, upper_model,
            train_criterion)
        val_metrics = validate_stage2(stage1, stage2, epsilon, val_loader, device, train_criterion)

        # Select best/schedule on the real objective (HR-space MSE of the
        # combined model), not the auxiliary residual-space training loss.
        scheduler.step(val_metrics['final_mse'])
        current_lr = optimizer.param_groups[0]['lr']

        epoch_duration = time.perf_counter() - epoch_start_time
        cumulative_time_sec += epoch_duration
        cumulative_time_hrs = cumulative_time_sec / 3600

        log_str = (
            f"[Duration: {epoch_duration:.1f}s | Total Duration: {cumulative_time_hrs:.2f}h] "
            f"Epoch [{epoch + 1}/{config['NUM_EPOCHS']}] | LR: {current_lr:.6f}\n"
            f"  Train -> Residual loss: {train_metrics['residual_loss']:.6f} | "
            f"Final MSE: {train_metrics['final_mse']:.6f}\n"
            f"  Val   -> Residual loss: {val_metrics['residual_loss']:.6f} | "
            f"Final MSE: {val_metrics['final_mse']:.6f}\n"
            f"{'-' * 70}\n"
        )
        print(log_str.strip())
        with open(log_file_path, "a") as f:
            f.write(log_str)

        if val_metrics['final_mse'] < best_val_final_mse:
            best_val_final_mse = val_metrics['final_mse']
            save_stage2_checkpoint(
                stage2, optimizer, epoch + 1, train_metrics['final_mse'], val_metrics['final_mse'],
                epsilon, stage1_checkpoint_path, best_weight_path,
            )

        if (epoch + 1) % 50 == 0:
            epoch_weight_path = os.path.join(config['ROOT_FOLDER'], f"output/weights/{upper_model}_epoch{epoch + 1}.pth")
            save_stage2_checkpoint(
                stage2, optimizer, epoch + 1, train_metrics['final_mse'], val_metrics['final_mse'],
                epsilon, stage1_checkpoint_path, epoch_weight_path,
            )

        early_stopper(val_metrics['final_mse'])
        if early_stopper.early_stop:
            print("Early stopping triggered! Validation loss has not improved sufficiently.")
            break


if __name__ == "__main__":
    import argparse
    import traceback
    import yaml

    parser = argparse.ArgumentParser(description="Train stage-2 multistage residual models")
    parser.add_argument('--config', type=str, required=True, nargs='+',
                         help="Path(s) to one or more multistage YAML config files")
    args = parser.parse_args()

    print(f"Queued {len(args.config)} run(s):")
    for config_path in args.config:
        print(f"  - {config_path}")

    queue_start = time.perf_counter()
    failed_configs = []

    for i, config_path in enumerate(args.config, start=1):
        print(f"\n{'=' * 70}\nQueue [{i}/{len(args.config)}]: {config_path}\n{'=' * 70}")

        with open(config_path, 'r') as file:
            experiment_config = yaml.safe_load(file)

        try:
            main(experiment_config)
        except Exception:
            print(f"Run failed for {config_path} -- skipping to the next queued config.")
            traceback.print_exc()
            failed_configs.append(config_path)
        finally:
            torch.cuda.empty_cache()

    queue_duration_hrs = (time.perf_counter() - queue_start) / 3600
    succeeded = len(args.config) - len(failed_configs)
    print(f"\nQueue finished in {queue_duration_hrs:.2f}h. {succeeded}/{len(args.config)} run(s) succeeded.")
    if failed_configs:
        print("Failed configs:")
        for config_path in failed_configs:
            print(f"  - {config_path}")
