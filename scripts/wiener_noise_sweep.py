"""Identity and Wiener/Tikhonov baselines against the observation-noise level,
for choosing sigma (the user picks it; nothing is fixed here).

    python scripts/wiener_noise_sweep.py [--device cpu]

For each sigma it runs the existing evaluation path
(`evaluate_run.py --baseline identity wiener --noise-std sigma`, gamma chosen
on validation for each sigma, test noise fixed per sample), then writes
    tables/wiener_noise_sweep.csv   one row per (sigma, baseline): gamma, MSE,
                                    PSNR, SSIM, KE, WKE, k_c, k_c_fase, eps/T over k >= D0
    tables/wiener_noise_eps_k.csv   eps(k) and T(k) per shell, per (sigma, baseline)
    figures/wiener_kc_vs_sigma.png/.pdf
and prints sigma as a fraction of the field's standard deviation and mean speed.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIGMAS = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def field_scales():
    """Standard deviation of the velocity field (u and v together), mean speed
    |V| = sqrt(u^2 + v^2), and mean |component|, over the training HR fields."""
    total, total_sq, n, speed, abs_comp = 0.0, 0.0, 0, 0.0, 0.0
    for year in range(2001, 2020):
        a = np.load(os.path.join(ROOT, f"hycom_numpy/hycom_{year}.npy")).astype(np.float64)
        total += a.sum()
        total_sq += (a ** 2).sum()
        n += a.size
        speed += np.sqrt((a ** 2).sum(-1)).sum()
        abs_comp += np.abs(a).sum()
    mean = total / n
    return {'std': float(np.sqrt(total_sq / n - mean ** 2)), 'mean_speed': float(speed / (n / 2)),
            'mean_abs_component': float(abs_comp / n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    args = parser.parse_args()

    scales = field_scales()
    rows, curves = [], []
    for sigma in SIGMAS:
        print(f"sigma = {sigma:g} ...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, 'evaluate_run.py'), '--baseline', 'identity', 'wiener',
                        '--noise-std', repr(sigma), '--results-dir', args.results_dir, '--device', args.device,
                        '--no-lpips'], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        tag = f"_noise{sigma:g}" if sigma > 0 else ""
        for name in ('identity', 'wiener'):
            folder = os.path.join(args.results_dir, f'baseline_{name}{tag}')
            with open(os.path.join(folder, 'eval_summary.json')) as f:
                ev = json.load(f)
            with open(os.path.join(folder, 'config.json')) as f:
                cfg = json.load(f)
            m, p = ev['means'], ev['pooled']
            rows.append({
                'sigma': sigma, 'sigma_over_std': sigma / scales['std'],
                'sigma_over_mean_speed': sigma / scales['mean_speed'],
                'sigma_over_mean_abs_component': sigma / scales['mean_abs_component'],
                'baseline': name, 'gamma': cfg.get('gamma', ''),
                'mse': m['mse'], 'psnr': m['psnr'], 'ssim': m['ssim'], 'ke': m['ke'], 'wke': m['wke'],
                'k_c': p['k_c'], 'k_c_fase': m['k_c_fase'],
                'eps_k_ge_D0': p['eps_k_ge_D0'], 'T_k_ge_D0': p['T_k_ge_D0'],
            })
            for k, (eps, T) in enumerate(zip(p['eps_k'], p['T_k']), start=1):
                curves.append({'sigma': sigma, 'baseline': name, 'k': k, 'eps_k': eps, 'T_k': T})

    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    for path, data in (('tables/wiener_noise_sweep.csv', rows), ('tables/wiener_noise_eps_k.csv', curves)):
        with open(os.path.join(ROOT, path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, style in (('wiener', '-o'), ('identity', '--s')):
        sel = [r for r in rows if r['baseline'] == name and r['sigma'] > 0]
        line, = ax.semilogx([r['sigma'] for r in sel], [r['k_c'] for r in sel], style, label=f'{name} (σ > 0)')
        zero = next(r for r in rows if r['baseline'] == name and r['sigma'] == 0)
        ax.axhline(zero['k_c'], color=line.get_color(), ls=':', lw=1, label=f'{name}, σ = 0: k_c = {zero["k_c"]:g}')
    ax.axhspan(20, 30, color='green', alpha=0.1, label='target band k_c = 20-30')
    ax.set_xlabel('Observation noise σ (m/s), log scale')
    ax.set_ylabel('Effective resolution k_c (cycles per 128-px domain)')
    ax.set_title('Effective resolution against noise: identity and Wiener (γ chosen per σ)')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=8)
    secax = ax.secondary_xaxis('top', functions=(lambda s: s / scales['std'], lambda f: f * scales['std']))
    secax.set_xlabel('σ / std of the velocity field')
    os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
    for ext in ('png', 'pdf'):
        # No embedded dates: re-running must not change a tracked file (it
        # would block real runs through train.py's clean-tree check).
        fig.savefig(os.path.join(ROOT, f'figures/wiener_kc_vs_sigma.{ext}'), dpi=200, bbox_inches='tight',
                    metadata={'png': {'Software': None},
                              'pdf': {'CreationDate': None, 'ModDate': None, 'Producer': None}}[ext])

    print(f"\nfield std {scales['std']:.4f} m/s, mean speed |V| {scales['mean_speed']:.4f} m/s, "
          f"mean |u|,|v| {scales['mean_abs_component']:.4f} m/s (training HR, 2001-2019)")
    print(f"{'sigma':>7s} {'base':>8s} {'gamma':>9s} {'MSE':>10s} {'PSNR':>6s} {'SSIM':>6s} {'KE':>7s} "
          f"{'WKE':>7s} {'k_c':>4s} {'k_cfase':>7s} {'s/std':>7s} {'s/|V|':>7s}")
    for r in rows:
        print(f"{r['sigma']:7g} {r['baseline']:>8s} {r['gamma'] if r['gamma'] != '' else '-':>9} {r['mse']:10.3e} "
              f"{r['psnr']:6.2f} {r['ssim']:6.4f} {r['ke']:7.3f} {r['wke']:7.3f} {r['k_c']:4.0f} {r['k_c_fase']:7.1f} "
              f"{r['sigma_over_std']:7.4f} {r['sigma_over_mean_speed']:7.4f}")


if __name__ == "__main__":
    main()
