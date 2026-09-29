"""Unit tests for the evaluation metrics and baselines (section 6, item 4).
CPU only, synthetic fields or a few real samples. Run from the repo root:

    CUDA_VISIBLE_DEVICES="" python tests/test_metrics.py
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_run  # noqa: E402
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from evaluation import baselines  # noqa: E402
from evaluation import spectral_metrics as sm  # noqa: E402
from losses.losses import MseLpipsLoss  # noqa: E402
from models.model_factory import create_model  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

N = 128
R, OFFSET, LPIPS_SCALE = 5.326794147491455, -2.4576449394226074, 2.8691492080688477  # from the 2001-2019 HR fields


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def cosine_field(k0, amplitude, component=0):
    x = torch.arange(N, dtype=torch.float64)
    field = torch.zeros(1, 2, N, N, dtype=torch.float64)
    field[0, component] = amplitude * torch.cos(2 * math.pi * k0 * x / N)[:, None].expand(N, N)
    return field


def main():
    set_seed(0)
    results = []
    test = SuperResDataset(hr_files=["./hycom_numpy/hycom_2022.npy"], downsample_factor=1, freq_threshold=10)
    lr = torch.stack([test[i][0] for i in range(8)])
    hr = torch.stack([test[i][1] for i in range(8)])

    # (a) H(k) with D0 = 10 reproduces the dataset's LR input
    ours = torch.fft.ifft2(torch.fft.fft2(hr) * sm.gaussian_transfer(N, 10.0)).real
    diff = (ours - lr).abs().max().item()
    results.append(check("(a) H(k)=exp(-k^2/(2*10^2)) applied to HR reproduces the dataset LR", diff < 1e-5,
                         f"(max abs diff {diff:.2e} m/s)"))

    # (b) radial spectrum of a known field: u = A cos(2 pi k0 x / N) puts A^2/4 in shell k0 only
    k0, a = 7, 0.8
    spec = sm.radial_spectrum(cosine_field(k0, a).float())[0]
    others = torch.cat([spec[:k0 - 1], spec[k0:]]).abs().max().item()
    results.append(check("(b) radial spectrum of a single cosine: E(k0) = A^2/4, zero elsewhere",
                         abs(spec[k0 - 1].item() - a ** 2 / 4) < 1e-6 and others < 1e-10,
                         f"(E({k0}) = {spec[k0 - 1].item():.6f}, expected {a ** 2 / 4:.6f})"))

    # (c) KE and WKE of a perfect prediction are zero
    results.append(check("(c) KE(y, y) = 0 and WKE(y, y) = 0",
                         sm.ke(hr, hr).abs().max().item() == 0 and sm.wke(hr, hr).abs().max().item() == 0))

    # (d) rho(k) of a perfect prediction is 1 for every shell, and k_c is the largest k
    rho = sm.spectral_correlation(hr.double(), hr.double())
    kc = sm.effective_resolution(rho)
    results.append(check("(d) rho(k)(y, y) = 1 for every k, k_c = 63", (rho - 1).abs().max().item() < 1e-12
                         and bool((kc == 63).all())))

    # (f) the same seed gives the same initial loss
    losses = []
    for _ in range(2):
        set_seed(0)
        model = create_model("srcnn", residual=True)
        losses.append(MseLpipsLoss(0.0, LPIPS_SCALE, "cpu")(model(lr), hr)[0])
    results.append(check("(f) same seed -> same initial loss", torch.equal(losses[0], losses[1])))

    # (g) PSNR uses one R for every sample: PSNR differences are pure MSE ratios
    pred = lr
    p, m = sm.psnr(pred, hr, R), sm.mse(pred, hr)
    expected = 10 * torch.log10(m[1] / m[0])
    results.append(check("(g) PSNR(s0) - PSNR(s1) = 10 log10(MSE1/MSE0): same R for both samples",
                         abs((p[0] - p[1]) - expected).item() < 1e-4,
                         f"(ranges of the two HR samples: {float(hr[0].max() - hr[0].min()):.3f}, "
                         f"{float(hr[1].max() - hr[1].min()):.3f} m/s; R = {R:.4f})"))

    # extra: band errors partition the MSE (Parseval)
    bands = sm.band_errors(pred.double(), hr.double())
    results.append(check("bands: sum of per-band errors == MSE (Parseval)",
                         torch.allclose(bands.sum(1), sm.mse(pred.double(), hr.double()), rtol=1e-10)))

    # extra: SSIM equals piq.ssim where piq accepts the inputs, and SSIM(y, y) = 1
    import piq
    ours_ssim = sm.ssim(pred, hr, R, OFFSET)
    piq_ssim = torch.stack([
        (piq.ssim((pred[:, c:c + 1] - OFFSET).clamp(0, R), (hr[:, c:c + 1] - OFFSET).clamp(0, R),
                  data_range=R, reduction='none')) for c in (0, 1)]).mean(0)
    in_range = bool(((pred - OFFSET) >= 0).all() and ((pred - OFFSET) <= R).all())
    results.append(check("SSIM matches piq.ssim (inputs inside [0, R] after the fixed offset)",
                         in_range and torch.allclose(ours_ssim, piq_ssim, atol=1e-5),
                         f"(max diff {(ours_ssim - piq_ssim).abs().max().item():.1e})"))
    results.append(check("SSIM(y, y) = 1", torch.allclose(sm.ssim(hr, hr, R, OFFSET), torch.ones(8), atol=1e-6)))

    # extra: Wiener with a small gamma beats the identity on noiseless data
    w = baselines.wiener(lr, 1e-4)
    results.append(check("Wiener (gamma 1e-4) has lower MSE than the identity baseline",
                         bool((sm.mse(w, hr) < sm.mse(lr, hr)).all()),
                         f"(mean MSE {sm.mse(w, hr).mean():.3e} vs {sm.mse(lr, hr).mean():.3e})"))

    # the training loss and the evaluation use the very same LPIPS function (same preprocessing)
    from losses import losses as loss_module, lpips_utils
    results.append(check("LPIPS: loss and evaluation call the same lpips_utils.lpips_distance object",
                         loss_module.lpips_distance is lpips_utils.lpips_distance
                         and evaluate_run.lpips_distance is lpips_utils.lpips_distance))

    # evaluation refuses a run without config.json
    with tempfile.TemporaryDirectory() as run_dir:
        try:
            evaluate_run.load_model(run_dir, "cpu")
            refused = False
        except evaluate_run.RunConfigError as e:
            refused = "config.json" in str(e)
        results.append(check("evaluation refuses a run without config.json (clear error)", refused))

        # end to end on 8 test samples: a fake run dir with a small residual model
        model = create_model("srcnn", residual=True)
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(run_dir, 'best.pt'))
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump({'model': 'srcnn', 'residual': True, 'psnr_ssim_R': R, 'hr_train_min': OFFSET,
                       'lpips_scale': LPIPS_SCALE}, f)
        loaded, meta = evaluate_run.load_model(run_dir, "cpu")
        constants = {k: meta[k] for k in evaluate_run.CONSTANT_KEYS}
        loader = DataLoader(Subset(test, range(8)), batch_size=4, shuffle=False)
        summary = evaluate_run.evaluate(loaded, constants, run_dir, loader, "cpu", with_lpips=False,
                                        dates=evaluate_run.read_dates()[:8])
        rows = open(os.path.join(run_dir, 'metrics_per_sample.csv')).read().splitlines()
        npz = np.load(os.path.join(run_dir, 'spectra.npz'))
        results.append(check("end to end: 8 rows with dates, spectra.npz shapes (8, 63)",
                             len(rows) == 9 and rows[1].split(',')[1] == '2022-01-01'
                             and npz['pred'].shape == (8, 63) and npz['rho'].shape == (8, 63),
                             f"(mean mse {summary['mse']:.3e}, psnr {summary['psnr']:.2f}, ssim {summary['ssim']:.4f})"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
