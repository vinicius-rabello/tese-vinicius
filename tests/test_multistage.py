"""Multistage on the new infrastructure (stage 4.7). CPU, tiny subsets:

    CUDA_VISIBLE_DEVICES="" python tests/test_multistage.py

Unit checks of StageChain and of the normalized-residual loss, then a chain
stage 1 -> 2 -> 3 and the continuous-training control trained through
train.py, a resumed stage, and the evaluation loader.
"""

import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.stage_chain import StageChain  # noqa: E402
from train import NormalizedResidualLoss  # noqa: E402
from losses.losses import MseLpipsLoss  # noqa: E402


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def write_config(folder, name, extra):
    path = os.path.join(folder, f"{name}.yaml")
    with open(path, 'w') as f:
        f.write(f"""MODEL_NAME: "srcnn"
RUN_NAME: "{name}"
RESULTS_DIR: "{folder}/results"
DEVICE: "cpu"
NUM_WORKERS: 0
BATCH_SIZE: 16
LEARNING_RATE: 0.0001
LAMBDA: 0
SEED: 0
IMAGE_SAVE_INTERVAL: 1000
SUBSET_TRAIN: 64
SUBSET_VAL: 32
MIN_EPOCHS: 1
MAX_EPOCHS: 2
""" + extra)
    return path


def train(config, *extra):
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'train.py'), '--config', config, *extra],
                         cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return out.stdout


def read(run, name):
    with open(os.path.join(run, name)) as f:
        return json.load(f)


def main():
    results = []
    torch.manual_seed(0)

    base, f = nn.Conv2d(2, 2, 3, padding=1), nn.Conv2d(2, 2, 3, padding=1)
    nn.init.zeros_(f.weight)
    nn.init.zeros_(f.bias)
    chain = StageChain(base, f, 0.05)
    x = torch.randn(4, 2, 16, 16)
    chain.train()
    results.append(check("f_n = 0 -> the chain returns stage n-1 exactly", torch.equal(chain(x), base(x))))
    results.append(check("stage n-1 frozen and kept in eval mode by model.train()",
                         not any(p.requires_grad for p in base.parameters()) and not base.training and f.training))
    results.append(check("epsilon travels in the state dict", abs(float(chain.state_dict()["epsilon"]) - 0.05) < 1e-8))

    # Gradient of the scaled HR-space loss == gradient of MSE on the normalized residual.
    f2 = nn.Conv2d(2, 2, 3, padding=1)
    chain = StageChain(base, f2, 0.05)
    y = torch.randn(4, 2, 16, 16)
    loss, _ = NormalizedResidualLoss(MseLpipsLoss(0.0, 1.0, device='cpu'), 0.05)(chain(x), y)
    g1 = torch.autograd.grad(loss, f2.weight)[0]
    target = (y - base(x)).detach() / 0.05
    g2 = torch.autograd.grad(nn.functional.mse_loss(f2(x), target), f2.weight)[0]
    results.append(check("training loss = MSE(f_n(x), (y - y_hat(n-1)) / epsilon) (same gradient)",
                         torch.allclose(g1, g2, rtol=1e-5, atol=1e-7),
                         f"(max diff {float((g1 - g2).abs().max()):.2e})"))

    with tempfile.TemporaryDirectory() as folder:
        res = os.path.join(folder, 'results')
        s1 = os.path.join(res, 'S1_res_lam0_seed0')
        s2 = os.path.join(res, 'S2_nores_lam0_seed0')
        s3 = os.path.join(res, 'S3_nores_lam0_seed0')
        train(write_config(folder, 'S1', "RESIDUAL: true\n"))
        bad = train(write_config(folder, 'S2bad', f"RESIDUAL: true\nSTAGE: 2\nSTAGE1_CHECKPOINT: {s1}/best.pt\n"))
        results.append(check("a stage with RESIDUAL: true is refused", 'failed' in bad.split('Queue stopped')[-1]))

        # Stage 2, stopped after its first epoch by the budget, then resumed.
        cfg2 = write_config(folder, 'S2', f"RESIDUAL: false\nSTAGE: 2\nSTAGE1_CHECKPOINT: {s1}/best.pt\n")
        train(cfg2, '--max-hours', '0.000001')
        eps_first = read(s2, 'config.json')['multistage']['epsilon']
        status_first = read(s2, 'summary.json')['status']
        out = train(cfg2)
        meta2 = read(s2, 'config.json')
        results.append(check("stage 2 interrupted by --max-hours resumes and finishes, reusing epsilon",
                             status_first == 'interrupted_max_hours' and 'Resumed from epoch 1' in out
                             and read(s2, 'summary.json')['status'] == 'completed_max_epochs'
                             and meta2['multistage']['epsilon'] == eps_first))

        from train import load_run_model, residual_rms
        from datasets.super_res_dataset import SuperResDataset
        from torch.utils.data import Subset
        ds = Subset(SuperResDataset(hr_files=[f"./hycom_numpy/hycom_{y}.npy" for y in range(2001, 2020)],
                                    downsample_factor=1, freq_threshold=10), range(64))
        cfg = {'BATCH_SIZE': 16, 'NUM_WORKERS': 0, 'DEVICE': 'cpu'}
        stage1 = load_run_model(s1)
        eps_ref = residual_rms(stage1, ds, cfg)
        results.append(check("epsilon_2 = RMS of the stage-1 residual on the training set",
                             abs(eps_ref - eps_first) <= 1e-6 * eps_ref, f"({eps_first:.6g} vs {eps_ref:.6g} m/s)"))
        s2_state = torch.load(os.path.join(s2, 'best.pt'), weights_only=False)['model_state_dict']
        s1_state = torch.load(os.path.join(s1, 'best.pt'), weights_only=False)['model_state_dict']
        results.append(check("stage-1 weights unchanged inside the stage-2 checkpoint",
                             all(torch.equal(s2_state['base.' + k], v) for k, v in s1_state.items())))
        results.append(check("only f_2 counted as trainable parameters",
                             meta2['n_params'] == read(s1, 'config.json')['n_params']))

        train(write_config(folder, 'S3', f"RESIDUAL: false\nSTAGE: 3\nSTAGE1_CHECKPOINT: {s2}/best.pt\n"))
        chain3 = load_run_model(s3)
        stage2 = load_run_model(s2)
        eps3 = read(s3, 'config.json')['multistage']['epsilon']
        with torch.no_grad():
            xb, yb = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
            manual = stage1(xb) + stage2.epsilon * stage2.stage(xb) + eps3 * chain3.stage(xb)
            ok = torch.allclose(chain3(xb), manual, atol=1e-6)
        results.append(check("stage 3 = stage 1 + eps_2 f_2 + eps_3 f_3 on the original input (recursive load)",
                             ok and abs(residual_rms(stage2, ds, cfg) - eps3) <= 1e-6 * eps3))

        from evaluate_run import load_model
        model, _ = load_model(s3, 'cpu')
        with torch.no_grad():
            results.append(check("evaluate_run.load_model rebuilds the stage-3 chain",
                                 torch.allclose(model(xb), chain3(xb))))

        train(write_config(folder, 'CTRL', f"RESIDUAL: true\nSTAGE: control\nFIXED_EPOCHS_FROM: [{s1}, {s2}]\n"))
        ctrl = os.path.join(res, 'CTRL_res_lam0_seed0')
        summary = read(ctrl, 'summary.json')
        results.append(check("control trains exactly the epochs of stage 1 + stage 2, then stops",
                             summary['epochs_completed'] == 4 and summary['status'] == 'completed_max_epochs'
                             and 'multistage' not in read(ctrl, 'config.json')))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
