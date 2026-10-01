"""Test-set evaluation of trained runs and of the reference baselines (stage 4.5).

    python evaluate_run.py results/<run_id> [results/<run_id> ...]
    python evaluate_run.py --baseline identity wiener
    python evaluate_run.py --split val results/<run_id> ...   # validation set (model selection)

Writes, per run (or results/baseline_<name>/):
    metrics_per_sample.csv  one row per test day: date, MSE, PSNR, SSIM, KE, WKE,
                            per-band errors, k_c, LPIPS (VGG, AlexNet)
    spectra.npz             per-sample radial spectra of the prediction and the
                            target, and rho(k), for the figures and the slope
    eval_summary.json       means over the test set

With --split val the same files get the suffix _val (metrics_per_sample_val.csv,
spectra_val.npz, eval_summary_val.json), so validation metrics, used to select
lambda and the architecture of block E, never mix with the test metrics.

Everything a model needs (architecture, residual, the fixed constants) is read
from the run's config.json. If it is missing, the run is refused rather than
built with guessed settings.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.super_res_dataset import SuperResDataset, NOISE_SEED_OFFSET
from evaluation import baselines
from evaluation import spectral_metrics as sm
from losses.lpips_utils import build_lpips, lpips_distance
from models.model_factory import create_model

TEST_FILES = [f"./hycom_numpy/hycom_{y}.npy" for y in (2022, 2023, 2024)]
VAL_FILES = ["./hycom_numpy/hycom_2020.npy", "./hycom_numpy/hycom_2021.npy"]
DATES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "test_dates.csv")
VAL_DATES_CSV = os.path.join(os.path.dirname(DATES_CSV), "val_dates.csv")
SPLIT_SUFFIX = {'test': '', 'val': '_val'}  # file-name suffix of each split's outputs
CONSTANT_KEYS = ('psnr_ssim_R', 'hr_train_min', 'lpips_scale')


class RunConfigError(RuntimeError):
    pass


def read_run_config(run_dir):
    path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(path):
        raise RunConfigError(f"{path} not found: without it the architecture, residual flag and "
                             "evaluation constants are unknown, so this run is not evaluated.")
    with open(path) as f:
        meta = json.load(f)
    missing = [k for k in ('model', 'residual', *CONSTANT_KEYS) if k not in meta]
    if missing:
        raise RunConfigError(f"{path} lacks {missing}; this run predates the current pipeline.")
    return meta


def load_model(run_dir, device):
    meta = read_run_config(run_dir)
    weights = os.path.join(run_dir, 'best.pt')
    if not os.path.exists(weights):
        raise RunConfigError(f"{weights} not found.")
    if meta.get('multistage'):
        # Stage n >= 2: the frozen chain of the previous stages plus f_n.
        from train import load_run_model
        return load_run_model(run_dir).to(device).eval(), meta
    model = create_model(meta['model'], in_channels=2, out_channels=2, residual=meta['residual'])
    state = torch.load(weights, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), meta


def test_loader(batch_size, noise_std=0.0):
    """Test set; with noise_std > 0 every sample gets its own fixed noise
    (seeded by its index), identical for every model and baseline."""
    ds = SuperResDataset(hr_files=TEST_FILES, downsample_factor=1, freq_threshold=10,
                         noise_std=noise_std, noise_mode='fixed', noise_seed_offset=NOISE_SEED_OFFSET['test'])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def val_loader(batch_size, noise_std=0.0):
    ds = SuperResDataset(hr_files=VAL_FILES, downsample_factor=1, freq_threshold=10,
                         noise_std=noise_std, noise_mode='fixed', noise_seed_offset=NOISE_SEED_OFFSET['val'])
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_noise_std(meta):
    # Runs from before the noisy regime have no key: they were trained without noise.
    return float(meta.get('noise_std', 0.0))


def read_dates(split='test'):
    with open(DATES_CSV if split == 'test' else VAL_DATES_CSV) as f:
        return [row['date'] for row in csv.DictReader(f)]


def evaluate(predict, constants, out_dir, loader, device, with_lpips=True, dates=None, split='test'):
    """Runs predict(lr) over the test loader and writes the per-sample metrics.
    `dates` defaults to the full test set's; pass a slice only with a matching subset loader."""
    os.makedirs(out_dir, exist_ok=True)
    dates = read_dates(split) if dates is None else dates
    suffix = SPLIT_SUFFIX[split]
    lpips_models = {net: build_lpips(net, device) for net in ('vgg', 'alex')} if with_lpips else {}
    rows, pred_spectra, target_spectra, rhos = [], [], [], []
    powers = {'pred': [], 'target': [], 'error': []}
    index = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = predict(lr)
            bands = sm.band_errors(pred, hr)
            rho = sm.spectral_correlation(pred, hr)
            pw = sm.shell_powers(pred, hr)
            batch = {
                'mse': sm.mse(pred, hr),
                'psnr': sm.psnr(pred, hr, constants['psnr_ssim_R']),
                'ssim': sm.ssim(pred, hr, constants['psnr_ssim_R'], constants['hr_train_min']),
                'ke': sm.ke(pred, hr),
                'wke': sm.wke(pred, hr),
                # k_c: largest k with eps(k') <= 1/2 for all k' <= k (this sample);
                # k_c_fase: the phase-coherence version, blind to blur.
                'k_c': sm.effective_resolution_eps(sm.relative_spectral_error(pw['error'], pw['target'])),
                'k_c_fase': sm.effective_resolution(rho),
            }
            # Stricter threshold (eps <= 0.1) and censoring markers (1 = k_c hit
            # the last shell, so it is only a lower bound).
            eps_sample = sm.relative_spectral_error(pw['error'], pw['target'])
            batch['k_c_01'] = sm.effective_resolution_eps(eps_sample, sm.K_C_THRESHOLDS['k_c_01'])
            for name in ('k_c', 'k_c_01'):
                batch[f'{name}_censurado'] = sm.k_c_censored(batch[name], eps_sample.shape[-1]).float()
            # eps_hf: mean of eps(k) over the shells k >= D0 (this sample).
            batch['eps_hf'] = eps_sample[:, int(sm.D0) - 1:].mean(-1)
            for b, name in enumerate(sm.BAND_NAMES):
                batch[f'band_{name}'] = bands[:, b]
            batch['band_k_ge_D0'] = bands[:, 2:].sum(1)
            for net, model in lpips_models.items():
                batch[f'lpips_{net}'] = lpips_distance(model, pred, hr, constants['lpips_scale'], reduction='none')
            for i in range(pred.shape[0]):
                rows.append({'index': index, 'date': dates[index],
                             **{k: float(v[i]) for k, v in batch.items()}})
                index += 1
            pred_spectra.append(sm.radial_spectrum(pred).cpu())
            target_spectra.append(sm.radial_spectrum(hr).cpu())
            rhos.append(rho.cpu())
            for name in powers:
                powers[name].append(pw[name].cpu())

    if index != len(dates):
        raise RuntimeError(f"Evaluated {index} samples but {len(dates)} dates were given.")
    with open(os.path.join(out_dir, f'metrics_per_sample{suffix}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(os.path.join(out_dir, f'spectra{suffix}.npz'),
                        k=np.arange(1, torch.cat(pred_spectra).shape[1] + 1),
                        pred=torch.cat(pred_spectra).numpy(), target=torch.cat(target_spectra).numpy(),
                        rho=torch.cat(rhos).numpy(), dates=np.array(dates),
                        pred_power=torch.cat(powers['pred']).numpy(), target_power=torch.cat(powers['target']).numpy(),
                        error_power=torch.cat(powers['error']).numpy())
    metric_keys = [k for k in rows[0] if k not in ('index', 'date')]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    pooled = pooled_spectral(*(torch.cat(powers[name]) for name in ('pred', 'target', 'error')))
    with open(os.path.join(out_dir, f'eval_summary{suffix}.json'), 'w') as f:
        json.dump({'split': split, 'n_samples': index, 'means': summary, 'pooled': pooled, 'constants': constants}, f, indent=2)
    return summary


def pooled_spectral(pred_power, target_power, error_power):
    """Test-set T(k), eps(k) and k_c from shell sums pooled over all samples,
    plus their summaries over k >= D0 (the restored region)."""
    p, t, e = pred_power.sum(0), target_power.sum(0), error_power.sum(0)
    T, eps = sm.transfer_ratio(p, t), sm.relative_spectral_error(e, t)
    high = torch.arange(1, len(t) + 1) >= sm.D0
    out = {
        'k_c': float(sm.effective_resolution_eps(eps)),
        'T_k': T.tolist(), 'eps_k': eps.tolist(),
        'T_k_ge_D0': float(T[high].mean()),
        'eps_k_ge_D0': float(e[high].sum() / t[high].sum()),
        'k_c_01': float(sm.effective_resolution_eps(eps, sm.K_C_THRESHOLDS['k_c_01'])),
        # eps_hf: mean of the pooled eps(k) over the shells k >= D0 (each shell
        # weighs the same; eps_k_ge_D0 above weighs them by target power).
        'eps_hf': float(eps[high].mean()),
    }
    for name in ('k_c', 'k_c_01'):
        out[f'{name}_censurado'] = bool(sm.k_c_censored(out[name], len(eps)))
    return out


def training_constants():
    from train import dataset_constants
    train = SuperResDataset([f"./hycom_numpy/hycom_{y}.npy" for y in range(2001, 2020)], 1, 10)
    return dataset_constants(train.hr_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', help="results/<run_id> directories")
    parser.add_argument('--baseline', nargs='*', choices=['identity', 'wiener'], default=[])
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-lpips', action='store_true')
    parser.add_argument('--split', choices=SPLIT_SUFFIX, default='test',
                        help="test (default) or val: validation metrics, written to *_val files")
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help="observation noise (m/s) for the baselines; runs use their own config.json value")
    args = parser.parse_args()

    loaders = {}
    make_loader = test_loader if args.split == 'test' else val_loader
    get_loader = lambda noise: loaders.setdefault(noise, make_loader(args.batch_size, noise))  # noqa: E731
    failures = []

    for run_dir in args.runs:
        try:
            model, meta = load_model(run_dir, args.device)
        except RunConfigError as e:
            print(f"SKIPPED {run_dir}: {e}")
            failures.append(run_dir)
            continue
        constants = {k: meta[k] for k in CONSTANT_KEYS}
        summary = evaluate(model, constants, run_dir, get_loader(run_noise_std(meta)), args.device, not args.no_lpips,
                           split=args.split)
        print(f"{run_dir} [{args.split}]: mse {summary['mse']:.4g} psnr {summary['psnr']:.3f} ssim {summary['ssim']:.4f}")

    if args.baseline:
        constants = training_constants()
        noise_tag = f"_noise{args.noise_std:g}" if args.noise_std > 0 else ""
        for name in args.baseline:
            out_dir = os.path.join(args.results_dir, f'baseline_{name}{noise_tag}')
            meta = {'baseline': name, 'noise_std': args.noise_std, **constants}
            if name == 'identity':
                predict = baselines.identity
            else:
                gamma, curve = baselines.choose_gamma(val_loader(args.batch_size, args.noise_std), device=args.device)
                meta.update({'gamma': gamma, 'gamma_selection': 'min validation MSE',
                             'val_mse_by_gamma': curve})
                predict = lambda lr, g=gamma: baselines.wiener(lr, g)  # noqa: E731
            os.makedirs(out_dir, exist_ok=True)
            config_path = os.path.join(out_dir, 'config.json')
            if args.split == 'val' and os.path.exists(config_path):
                # Keep the config of the test evaluation; gamma must be the same.
                with open(config_path) as f:
                    kept = json.load(f)
                if kept.get('gamma') != meta.get('gamma'):
                    raise RuntimeError(f"{out_dir}: gamma {meta.get('gamma')} now vs {kept.get('gamma')} in config.json")
            else:
                with open(config_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            summary = evaluate(predict, constants, out_dir, get_loader(args.noise_std), args.device, not args.no_lpips,
                               split=args.split)
            print(f"baseline {name}: mse {summary['mse']:.4g} psnr {summary['psnr']:.3f} ssim {summary['ssim']:.4f}"
                  + (f" (gamma {meta['gamma']:.3g})" if name == 'wiener' else ""))

    if failures:
        print(f"{len(failures)} run(s) not evaluated: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
