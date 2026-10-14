"""Hann window for the spectral metrics (evaluation only). CPU:

    CUDA_VISIBLE_DEVICES="" python tests/test_window.py

(1) the training-side KineticEnergySpectrumLoss and every spectral metric with
window='none' are bit-identical to the code before the change (loaded from git
HEAD); (2) 'hann' keeps the energy scale and removes the edge leakage of a
non-periodic field; (3) the 50-day mean HR spectrum with 'hann' has a slope in
[-2.9, -2.4] over k = 11..46 (analises/dissertacao/espectro_medio_hr_teste.json).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation import spectral_metrics as sm  # noqa: E402
from metrics.kinetic_energy_spectrum import KineticEnergySpectrumLoss  # noqa: E402

BASE = os.environ.get('WINDOW_TEST_BASE', 'HEAD')   # commit with the code before the change


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def old_module(path, name):
    src = subprocess.run(['git', 'show', f'{BASE}:{path}'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(f.name)
    return mod


def main():
    results = []
    torch.manual_seed(0)
    x = torch.randn(4, 2, 128, 128).cumsum(-1).cumsum(-2) * 1e-2      # smooth, non-periodic fields
    y = x + 0.01 * torch.randn_like(x)

    old_ke_mod = old_module('metrics/kinetic_energy_spectrum.py', 'old_ke')
    new_loss = KineticEnergySpectrumLoss(size=128)(y, x)
    old_loss = old_ke_mod.KineticEnergySpectrumLoss(size=128)(y, x)
    unchanged = subprocess.run(['git', 'diff', '--quiet', BASE, '--', 'metrics/kinetic_energy_spectrum.py',
                                'losses/losses.py', 'train.py'], cwd=ROOT).returncode == 0
    results.append(check("training loss: KineticEnergySpectrumLoss bit-identical to the code before the change, "
                         "and its file, losses.py and train.py untouched",
                         torch.equal(new_loss, old_loss) and unchanged, f"(loss {new_loss.item():.10g})"))

    old_sm = old_module('evaluation/spectral_metrics.py', 'old_sm')
    same = {
        'ke': torch.equal(sm.ke(y, x), old_sm.ke(y, x)),
        'wke': torch.equal(sm.wke(y, x), old_sm.wke(y, x)),
        'radial_spectrum': torch.equal(sm.radial_spectrum(y), old_sm.radial_spectrum(y)),
        'band_errors': torch.equal(sm.band_errors(y, x), old_sm.band_errors(y, x)),
        'spectral_correlation': torch.equal(sm.spectral_correlation(y, x), old_sm.spectral_correlation(y, x)),
        'shell_powers': all(torch.equal(a, b) for a, b in zip(sm.shell_powers(y, x).values(),
                                                               old_sm.shell_powers(y, x).values())),
    }
    results.append(check("window='none' (default): every spectral metric bit-identical to before",
                         all(same.values()), str({k: v for k, v in same.items() if not v}) if not all(same.values()) else ""))

    # Energy scale: for white noise the windowed spectrum keeps the total energy.
    noise = torch.randn(64, 2, 128, 128)
    ratio = (sm.radial_spectrum(noise, 'hann').sum(1) / sm.radial_spectrum(noise).sum(1)).mean().item()
    results.append(check("hann: energy scale kept (white noise, total E ratio ~ 1)", abs(ratio - 1) < 0.02,
                         f"(ratio {ratio:.4f})"))

    # Leakage: a field with a jump between opposite edges (a linear ramp) puts
    # k^-2 energy at high k without a window and almost none with it.
    ramp = torch.linspace(0, 1, 128).repeat(128, 1)[None, None].repeat(1, 2, 1, 1)
    hi_none = sm.radial_spectrum(ramp)[0, 39:].sum().item()
    hi_hann = sm.radial_spectrum(ramp, 'hann')[0, 39:].sum().item()
    results.append(check("hann: a smooth ramp (edge jump) has ~no energy at k >= 40 with the window",
                         hi_hann < 1e-4 * hi_none, f"(k>=40: {hi_none:.3g} -> {hi_hann:.3g})"))

    same_window = torch.equal(sm.apply_window(x, 'hann')[0], sm.apply_window(x[:1], 'hann')[0])
    results.append(check("hann: the same window for every field (per-sample result independent of the batch)",
                         same_window))

    ref = json.load(open(os.path.join(ROOT, 'analises', 'dissertacao', 'espectro_medio_hr_teste.json')))
    from evaluate_run import TEST_FILES
    hr = np.concatenate([np.load(f, mmap_mode='r') for f in TEST_FILES])[ref['indices']]
    hr = torch.from_numpy(np.ascontiguousarray(hr)).permute(0, 3, 1, 2)
    spec = sm.radial_spectrum(hr, 'hann').mean(0)[None]
    slope = float(sm.spectral_slope(spec, 11, 46))
    results.append(check("hann: 50-day mean HR slope over k = 11..46 in [-2.9, -2.4] (reference -2.66)",
                         -2.9 <= slope <= -2.4, f"(slope {slope:.3f})"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
