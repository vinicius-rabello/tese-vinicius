"""Edge-leakage check for spectral comparisons (read only, CPU): the same 50
test days as espectro_medio_hr_teste.json; radial spectra with and without a 2D
Hann window for HR, LR, Wiener (gamma 1e-12) and two models (best and worst by
test MSE among the evaluated runs); ratio E_pred(k)/E_HR(k) over k = 11..46."""
import json, os, sys
import numpy as np, torch
REPO = '/home/lcadame/Vinicius_Thesis/Vinicius'; OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); os.chdir(REPO)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa
import evaluate_run  # noqa
from datasets.super_res_dataset import SuperResDataset  # noqa
from evaluation import baselines, spectral_metrics as sm  # noqa
torch.set_grad_enabled(False); torch.set_num_threads(8)
MODELS = {'pior: A_srcnn (λ=0)': 'results/A_srcnn_res_lam0_seed0', 'melhor avaliado: B_srno λ=0,01': 'results/B_srno_res_lam0.01_seed0'}
idx = json.load(open(os.path.join(OUT, 'espectro_medio_hr_teste.json')))['indices']
test = SuperResDataset(evaluate_run.TEST_FILES, 1, 10)
pairs = [test[i] for i in idx]
lr = torch.stack([p[0] for p in pairs]); hr = torch.stack([p[1] for p in pairs])
fields = {'HR': hr, 'LR': lr, 'Wiener (γ=1e-12)': baselines.wiener(lr, 1e-12)}
for name, run in MODELS.items():
    m, _ = evaluate_run.load_model(run, 'cpu'); fields[name] = torch.cat([m(lr[i:i + 10]) for i in range(0, len(lr), 10)])
w1 = torch.hann_window(128, periodic=False); w2 = w1[:, None] * w1[None, :]; norm = float((w2 ** 2).mean())
k = np.arange(1, 64); sel = (k >= 11) & (k <= 46)
spec = {(n, win): (sm.radial_spectrum(f * w2) / norm if win else sm.radial_spectrum(f)).mean(0).numpy()
        for n, f in fields.items() for win in (False, True)}
res = {}
for n in fields:
    if n == 'HR':
        continue
    for win in (False, True):
        r = spec[(n, win)] / spec[('HR', win)]
        res[f"{n} | {'Hann' if win else 'sem janela'}"] = {
            'razao_media_k11_46': float(r[sel].mean()), 'razao_media_log_k11_46': float(np.exp(np.log(r[sel]).mean())),
            **{f'razao_k{kk}': float(r[kk - 1]) for kk in (11, 15, 20, 25, 30, 40, 46)}}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, win in zip(axes, (False, True)):
    for n, st in zip([n for n in fields if n != 'HR'], ('--', ':', '-', '-')):
        ax.semilogy(k, spec[(n, win)] / spec[('HR', win)], ls=st, lw=1.6, label=n)
    ax.axhline(1, color='black', lw=0.8); ax.axvspan(11, 46, color='gray', alpha=0.08)
    ax.set_ylim(1e-3, 3); ax.set_xlabel('k (ciclos por domínio de 128 px; λ ≈ 515/k km)')
    ax.set_title(f"E_pred(k)/E_HR(k), 50 dias do teste, {'com janela Hann' if win else 'sem janela'}")
    ax.grid(True, which='both', ls='--', alpha=0.3)
axes[0].set_ylabel('E_pred(k) / E_HR(k) (adimensional)'); axes[0].legend(fontsize=8)
fig.tight_layout()
for ext, meta in (('png', {'Software': None}), ('pdf', {'CreationDate': None, 'ModDate': None, 'Producer': None})):
    fig.savefig(os.path.join(OUT, f'razao_espectral_janela.{ext}'), dpi=200, bbox_inches='tight', metadata=meta)
order = {win: sorted([n for n in MODELS], key=lambda n: -res[f"{n} | {win}"]['razao_media_log_k11_46']) for win in ('sem janela', 'Hann')}
json.dump({'dias': [test_d for test_d in json.load(open(os.path.join(OUT, 'espectro_medio_hr_teste.json')))['datas']],
           'modelos': MODELS, 'razoes': res, 'ordem_por_razao_k11_46 (mais energia primeiro)': order},
          open(os.path.join(OUT, 'razao_espectral_janela.json'), 'w'), indent=1, ensure_ascii=False)
for kk, v in res.items(): print(f"{kk:45s} média {v['razao_media_k11_46']:.3f} geom {v['razao_media_log_k11_46']:.3g} k20 {v['razao_k20']:.3f} k30 {v['razao_k30']:.3g} k40 {v['razao_k40']:.3g}")
print(order)
