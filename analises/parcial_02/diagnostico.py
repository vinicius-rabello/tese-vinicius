"""Partial analysis 02: why eps(k) of the networks is not monotonic at high k.
Read only, CPU. Run from the repository root:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 venv/bin/python analises/parcial_02/diagnostico.py

Writes everything next to this script.
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from netCDF4 import Dataset

REPO = '/home/lcadame/Vinicius_Thesis/Vinicius'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import evaluate_run  # noqa: E402
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from evaluation import baselines  # noqa: E402
from evaluation import spectral_metrics as sm  # noqa: E402

torch.set_grad_enabled(False)
N = 128
K = np.arange(1, N // 2)                     # shells [k, k+1), k = 1..63
RUNS = {'SRCNN': 'results/A_srcnn_res_lam0_seed0', 'DSC/MS': 'results/A_dscms_res_lam0_seed0'}
GAMMA = 1e-12
KS_SHOW = (20, 25, 31, 40, 50, 60)
report = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ helpers --
def predictions(split):
    files = evaluate_run.TEST_FILES if split == 'test' else evaluate_run.VAL_FILES
    ds = SuperResDataset(files, 1, 10)
    lr = torch.stack([ds[i][0] for i in range(len(ds))])
    hr = torch.from_numpy(ds.hr_data).permute(0, 3, 1, 2).contiguous()
    preds = {'identidade': lr, 'Wiener': torch.cat([baselines.wiener(lr[i:i + 64], GAMMA) for i in range(0, len(lr), 64)])}
    for name, run in RUNS.items():
        model, _ = evaluate_run.load_model(run, 'cpu')
        preds[name] = torch.cat([model(lr[i:i + 32]) for i in range(0, len(lr), 32)])
        log(f'  {split}: {name} inference done')
    return lr, hr, preds


def tukey(n, alpha):
    w = np.ones(n)
    if alpha <= 0:
        return w
    edge = int(np.floor(alpha * (n - 1) / 2))
    x = np.arange(edge + 1)
    w[:edge + 1] = 0.5 * (1 + np.cos(np.pi * (2 * x / (alpha * (n - 1)) - 1)))
    w[n - edge - 1:] = w[:edge + 1][::-1]
    return w


def shell_index(n_fft, scale, exclude_axes=False):
    """Shell id per FFT bin in 128-domain units (k = r / scale), -1 = unused."""
    f = np.fft.fftfreq(n_fft) * n_fft
    kx, ky = np.meshgrid(f, f, indexing='ij')
    r = np.sqrt(kx ** 2 + ky ** 2) / scale
    idx = np.floor(r).astype(int) - 1
    idx[(r < 1) | (idx >= len(K))] = -1
    if exclude_axes:
        idx[(np.abs(kx) <= scale) | (np.abs(ky) <= scale)] = -1
    return idx


def transform(x, variant):
    """x: (B, 2, 128, 128) tensor -> field to FFT for the variant."""
    if variant in ('hann', 'tukey25'):
        w1 = tukey(N, 1.0 if variant == 'hann' else 0.25)
        return x * torch.from_numpy(np.outer(w1, w1)).float()
    if variant == 'espelho':
        x = torch.cat([x, x.flip(-1)], -1)
        return torch.cat([x, x.flip(-2)], -2)
    return x


VARIANTS = {'a_como_esta': (1, False), 'b_hann': (1, False), 'b_tukey25': (1, False), 'c_espelho': (2, False),
            'd_sem_eixos': (1, True)}


def pooled(pred, target, variant):
    scale, ex = VARIANTS[variant]
    key = {'b_hann': 'hann', 'b_tukey25': 'tukey25', 'c_espelho': 'espelho'}.get(variant, 'plain')
    idx = torch.from_numpy(shell_index(N * scale, scale, ex)).flatten()
    valid = idx >= 0
    sums = np.zeros((3, len(K)))
    for i in range(0, len(pred), 64):
        p, t = transform(pred[i:i + 64], key), transform(target[i:i + 64], key)
        P, T = torch.fft.fft2(p.double()), torch.fft.fft2(t.double())
        for j, x in enumerate(((P.abs() ** 2).sum(1), (T.abs() ** 2).sum(1), ((P - T).abs() ** 2).sum(1))):
            x = x.sum(0).flatten()
            sums[j] += np.bincount(idx[valid].numpy(), weights=x[valid].numpy(), minlength=len(K))
    pp, tt, ee = sums
    T_k, eps = pp / tt, ee / tt
    coh = (pp + tt - ee) / (2 * np.sqrt(pp * tt))
    return T_k, eps, coh, pp, tt, ee


# ------------------------------------------------------------------ item 1 --
def item1():
    log('item 1')
    out = {}
    tr = np.load('hycom_numpy/hycom_2010.npy', mmap_mode='r')
    te = np.concatenate([np.load(f'hycom_numpy/hycom_{y}.npy', mmap_mode='r') for y in (2022, 2023, 2024)])
    tr_all = np.concatenate([np.load(f'hycom_numpy/hycom_{y}.npy', mmap_mode='r') for y in range(2001, 2020)])
    for name, a in (('treino 2001-2019', tr_all), ('teste 2022-2024', te)):
        zero = (a[..., 0] == 0) & (a[..., 1] == 0)
        always = zero.all(0); frac_day = zero.reshape(len(a), -1).mean(1)
        out[name] = {'n': int(len(a)), 'terra_sempre_zero_px': int(always.sum()),
                     'fracao_terra_por_recorte_min_med_max': [float(frac_day.min()), float(frac_day.mean()), float(frac_day.max())],
                     'zeros_fora_da_mascara_fixa (px-dia)': int((zero & ~always).sum())}
    land = (te[..., 0] == 0).all(0) & (te[..., 1] == 0).all(0)
    np.save(os.path.join(OUT, 'mascara_terra.npy'), land)
    # Raw 1/100 deg mask in the 512 x 512 window: fraction of land per 4 x 4 block.
    with Dataset('/home/lcadame/Vinicius_Thesis/Data/hycom/100/2022/hycom_gom_2022-01.nc') as ds:
        m = np.ma.getmaskarray(ds['u'][0])[512:1024, 794:1306]
    block = m.reshape(128, 4, 128, 4).mean((1, 3))
    out['mascara_bruta'] = {'blocos_terra_total': int((block == 1).sum()), 'blocos_mistos (0<terra<1)': int(((block > 0) & (block < 1)).sum()),
                            'coincide_com_zeros': bool(((block == 1) == land).all())}
    # Edge discontinuity: jump between opposite edges vs. typical neighbour difference.
    f = te.astype(np.float32)
    jump_x = np.abs(f[:, 0] - f[:, -1]).mean(); jump_y = np.abs(f[:, :, 0] - f[:, :, -1]).mean()
    neigh = 0.5 * (np.abs(np.diff(f, axis=1)).mean() + np.abs(np.diff(f, axis=2)).mean())
    out['bordas'] = {'salto_medio_bordas_opostas_linhas (m/s)': float(jump_x), 'salto_medio_bordas_opostas_colunas (m/s)': float(jump_y),
                     'diferenca_media_entre_vizinhos (m/s)': float(neigh),
                     'razao_salto/vizinho': float(0.5 * (jump_x + jump_y) / neigh)}
    # Where is the target energy at k >= 40: on the kx = 0 / ky = 0 axes or off them.
    t = torch.from_numpy(f).permute(0, 3, 1, 2)
    P = (torch.fft.fft2(t.double()).abs() ** 2).sum(1).sum(0).numpy()
    fr = np.fft.fftfreq(N) * N; kx, ky = np.meshgrid(fr, fr, indexing='ij'); r = np.sqrt(kx ** 2 + ky ** 2)
    hi = (r >= 40) & (r < 64); axes = (np.abs(kx) <= 1) | (np.abs(ky) <= 1)
    out['energia_alvo_k40_63'] = {'fracao_nos_eixos': float(P[hi & axes].sum() / P[hi].sum()),
                                  'fracao_de_bins_nos_eixos': float((hi & axes).sum() / hi.sum())}
    report['item1'] = out
    return land


# ------------------------------------------------------------------ item 2 --
def item2(hr, preds):
    log('item 2')
    rows, curves = [], {}
    for var in VARIANTS:
        for name, pred in preds.items():
            T_k, eps, coh, pp, tt, ee = pooled(pred, hr, var)
            curves[(var, name)] = (T_k, eps, coh)
            hi = K >= 40
            rows.append({'variante': var, 'modelo': name,
                         **{f'eps_k{k}': eps[k - 1] for k in KS_SHOW}, **{f'T_k{k}': T_k[k - 1] for k in KS_SHOW},
                         **{f'coer_k{k}': coh[k - 1] for k in KS_SHOW},
                         'eps_max': eps.max(), 'k_eps_max': int(K[eps.argmax()]),
                         'eps_banda_ge40': ee[hi].sum() / tt[hi].sum(), 'eps_hf': eps[K >= 10].mean()})
    with open(os.path.join(OUT, 'item2_variantes.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    styles = {'a_como_esta': '-', 'b_hann': '--', 'b_tukey25': '-.', 'c_espelho': ':', 'd_sem_eixos': (0, (1, 3))}
    for col, name in enumerate(('SRCNN', 'DSC/MS')):
        for var, ls in styles.items():
            T_k, eps, coh = curves[(var, name)]
            axes[0, col].plot(K, eps, ls=ls, label=var); axes[1, col].plot(K, T_k, ls=ls, label=var)
            axes[2, col].plot(K, coh, ls=ls, label=var)
        axes[0, col].set_title(f'{name}: ε(k), test set'); axes[1, col].set_title(f'{name}: T(k)')
        axes[2, col].set_title(f'{name}: coherence c(k) = (E_ŷ+E_y−E_err)/(2√(E_ŷE_y))')
        for r, lab in enumerate(('ε(k) (dimensionless)', 'T(k) (dimensionless)', 'c(k) (dimensionless)')):
            axes[r, col].set_ylabel(lab); axes[r, col].grid(alpha=0.3); axes[r, col].axvline(10, color='gray', lw=0.8, ls=':')
        axes[2, col].set_xlabel('Wavenumber k (cycles per 128-px domain)')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'item2_variantes.{ext}'), dpi=150, metadata={'Software': None} if ext == 'png' else {'CreationDate': None})
    plt.close(fig)
    report['item2'] = rows
    return curves


# ------------------------------------------------------------------ item 3 --
def band_component(x, lo, hi):
    """x: (..., 128, 128) numpy -> component with lo <= |k| < hi."""
    r = sm.radius_grid(N).numpy()
    mask = (r >= lo) & (r < hi)
    return np.fft.ifft2(np.fft.fft2(x) * mask).real


def item3(hr, preds, land, dates):
    log('item 3')
    pick = ['2023-01-15', '2024-02-15', '2022-07-15', '2023-08-15']
    idx = [dates.index(d) for d in pick]
    cols = [('Alvo: velocidade |u| (m/s)', None), ('Alvo: componente k≥40 de u (m/s)', None),
            ('Erro SRCNN ŷ−y, u (m/s)', 'SRCNN'), ('Erro DSC/MS ŷ−y, u (m/s)', 'DSC/MS')]
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    hf_lim = np.percentile(np.abs(band_component(hr[idx, 0].numpy(), 40, 64)), 99.5)
    err_lim = max(np.percentile(np.abs((preds[m][idx, 0] - hr[idx, 0]).numpy()), 99.5) for m in RUNS)
    land_rgba = np.zeros((N, N, 4)); land_rgba[land] = (0.35, 0.35, 0.35, 1)
    for r, (i, d) in enumerate(zip(idx, pick)):
        y = hr[i].numpy()
        panels = [np.hypot(y[0], y[1]), band_component(y[0], 40, 64),
                  (preds['SRCNN'][i, 0] - hr[i, 0]).numpy(), (preds['DSC/MS'][i, 0] - hr[i, 0]).numpy()]
        for c, (img, (title, _)) in enumerate(zip(panels, cols)):
            ax = axes[r, c]
            if c == 0:
                im = ax.imshow(img, origin='lower', cmap='viridis')
            else:
                lim = hf_lim if c == 1 else err_lim
                im = ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.imshow(land_rgba, origin='lower')
            ax.set_title(f'{d} ({"inverno" if r < 2 else "verão"}): {title}', fontsize=9)
            ax.set_xlabel('x (px, 0.04°)'); ax.set_ylabel('y (px, 0.04°)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle('Mapas de erro e da componente k≥40 do alvo (terra em cinza); origem embaixo à esquerda', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'item3_mapas.{ext}'), dpi=110, metadata={'Software': None} if ext == 'png' else {'CreationDate': None})
    plt.close(fig)
    report['item3'] = {'datas': pick, 'indices_teste': idx}


# ------------------------------------------------------------------ item 4 --
def item4(hr, preds, land):
    log('item 4')
    tr = np.concatenate([np.load(f'hycom_numpy/hycom_{y}.npy', mmap_mode='r') for y in range(2001, 2020)])
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(tr), 1000, replace=False))
    tr = torch.from_numpy(np.ascontiguousarray(tr[sel])).permute(0, 3, 1, 2).numpy()
    te = hr.numpy()

    def feats(y, with_grad):
        lf = band_component(y, 0, 20)                       # includes the mean (k = 0)
        hf = band_component(y, 40, 64)
        f = [lf[:, 0], lf[:, 1]]
        if with_grad:
            for c in (0, 1):
                f += [np.roll(lf[:, c], -1, 2) - np.roll(lf[:, c], 1, 2), np.roll(lf[:, c], -1, 1) - np.roll(lf[:, c], 1, 1)]
        return np.stack(f, -1), hf

    # Pixel masks: all; ocean only; ocean >= 8 px from land and from the patch edges.
    from scipy.ndimage import binary_dilation
    near_land = binary_dilation(land, iterations=8)
    edge = np.zeros((N, N), bool); edge[:8] = edge[-8:] = True; edge[:, :8] = edge[:, -8:] = True
    masks = {'todos os pixels': np.ones((N, N), bool), 'oceano': ~land, 'oceano interior (>=8 px de terra e bordas)': ~(near_land | edge)}
    out = {}
    for with_grad in (False, True):
        Xtr, Htr = feats(tr, with_grad); Xte, Hte = feats(te, with_grad)
        for mname, m in masks.items():
            res = {}
            for c, comp in ((0, 'u'), (1, 'v')):
                A = np.c_[Xtr[:, m].reshape(-1, Xtr.shape[-1]), np.ones(m.sum() * len(tr))]
                coef, *_ = np.linalg.lstsq(A, Htr[:, c][:, m].ravel(), rcond=None)
                At = np.c_[Xte[:, m].reshape(-1, Xte.shape[-1]), np.ones(m.sum() * len(te))]
                yt = Hte[:, c][:, m].ravel()
                res[comp] = 1 - ((yt - At @ coef) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
            out[f"{'LF+gradientes' if with_grad else 'LF (u,v)'} | {mname}"] = float(np.mean(list(res.values())))
    # What the networks explain of the same k >= 40 component, in the same pixels.
    for name in ('SRCNN', 'DSC/MS', 'Wiener'):
        hp = band_component(preds[name].numpy(), 40, 64); ht = band_component(te, 40, 64)
        for mname, m in masks.items():
            r2 = [1 - ((hp[:, c][:, m] - ht[:, c][:, m]) ** 2).sum() / ((ht[:, c][:, m] - ht[:, c][:, m].mean()) ** 2).sum()
                  for c in (0, 1)]
            out[f'rede {name} | {mname}'] = float(np.mean(r2))
    # Share of the k >= 40 energy of the target near land / edges.
    ht = band_component(te, 40, 64); e = (ht ** 2).sum((0, 1))
    out['fracao_energia_k40_perto_de_terra(8px)'] = float(e[near_land & ~land].sum() / e.sum())
    out['fracao_energia_k40_nas_bordas(8px)'] = float(e[edge].sum() / e.sum())
    out['fracao_de_pixels_perto_de_terra(8px)'] = float((near_land & ~land).mean())
    out['fracao_de_pixels_nas_bordas(8px)'] = float(edge.mean())
    report['item4'] = out


# ------------------------------------------------------------------ item 5 --
def item5(splits):
    log('item 5')
    rows = []
    for split, (hr, preds) in splits.items():
        for name, pred in preds.items():
            row = {'split': split, 'modelo': name}
            for var in ('a_como_esta', 'b_hann'):
                T_k, eps, coh, pp, tt, ee = pooled(pred, hr, var)
                tag = '' if var == 'a_como_esta' else ' (Hann)'
                row[f'eps_hf media k>=10{tag}'] = eps[K >= 10].mean()
                row[f'eps ponderado k>=10{tag}'] = ee[K >= 10].sum() / tt[K >= 10].sum()
                row[f'eps ponderado 10<=k<40{tag}'] = ee[(K >= 10) & (K < 40)].sum() / tt[(K >= 10) & (K < 40)].sum()
            be = sm.band_errors(pred, hr).mean(0).numpy()
            for b, v in zip(sm.BAND_NAMES, be):
                row[f'MSE {b}'] = v
            row['MSE'] = float(((pred - hr) ** 2).mean())
            rows.append(row)
    with open(os.path.join(OUT, 'item5_metricas.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report['item5'] = rows


def main():
    with open('evaluation/test_dates.csv') as fh:
        dates = [r['date'] for r in csv.DictReader(fh)]
    land = item1()
    _, hr, preds = predictions('test')
    item2(hr, preds)
    item3(hr, preds, land, dates)
    item4(hr, preds, land)
    _, hr_v, preds_v = predictions('val')
    item5({'val': (hr_v, preds_v), 'test': (hr, preds)})
    with open(os.path.join(OUT, 'resultados.json'), 'w') as fh:
        json.dump(report, fh, indent=1, default=float)
    log('done')


if __name__ == '__main__':
    main()
