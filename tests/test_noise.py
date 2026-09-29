"""Checks for the observation-noise option (NOISE_STD). CPU only. Run from the repo root:

    CUDA_VISIBLE_DEVICES="" python tests/test_noise.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.super_res_dataset import NOISE_SEED_OFFSET, SuperResDataset  # noqa: E402
from train import run_id_for  # noqa: E402

REFERENCE_COMMIT = "e98373b"  # last commit before NOISE_STD
FILES = ["./hycom_numpy/hycom_2022.npy"]
SIGMA = 0.01


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def reference_dataset_class():
    source = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:datasets/super_res_dataset.py"]).decode()
    tmp = tempfile.NamedTemporaryFile("w", suffix="_ref_dataset.py", delete=False)
    tmp.write(source)
    tmp.close()
    spec = importlib.util.spec_from_file_location("ref_dataset", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return module.SuperResDataset


def main():
    results = []
    old = reference_dataset_class()(FILES, 1, 10)
    new = SuperResDataset(FILES, 1, 10)
    new_explicit = SuperResDataset(FILES, 1, 10, noise_std=0.0, noise_mode='random')
    indices = (0, 1, 100, 364)

    rng_before = torch.get_rng_state()
    same = all(torch.equal(old[i][0], new[i][0]) and torch.equal(old[i][1], new[i][1])
               and torch.equal(old[i][0], new_explicit[i][0]) for i in indices)
    results.append(check("noise_std = 0: LR and HR bit-identical to the dataset before NOISE_STD", same))
    results.append(check("noise_std = 0: the global RNG is not touched", torch.equal(rng_before, torch.get_rng_state())))

    val_a = SuperResDataset(FILES, 1, 10, noise_std=SIGMA, noise_seed_offset=NOISE_SEED_OFFSET['val'])
    val_b = SuperResDataset(FILES, 1, 10, noise_std=SIGMA, noise_seed_offset=NOISE_SEED_OFFSET['val'])
    test = SuperResDataset(FILES, 1, 10, noise_std=SIGMA, noise_seed_offset=NOISE_SEED_OFFSET['test'])
    results.append(check("fixed noise: same sample, same noise in two dataset instances (every model sees it)",
                         all(torch.equal(val_a[i][0], val_b[i][0]) for i in indices)))
    results.append(check("fixed noise: different samples get different noise",
                         not torch.equal(val_a[0][0] - new[0][0], val_a[1][0] - new[1][0])))
    results.append(check("fixed noise: validation and test offsets give different noise on the same index",
                         not torch.equal(val_a[0][0], test[0][0])))
    results.append(check("fixed noise: the HR target stays clean",
                         all(torch.equal(val_a[i][1], new[i][1]) for i in indices)))
    noise = torch.stack([val_a[i][0] - new[i][0] for i in range(50)])
    results.append(check("fixed noise: mean ~ 0 and std ~ sigma",
                         abs(noise.mean().item()) < 1e-4 and abs(noise.std().item() / SIGMA - 1) < 0.01,
                         f"(mean {noise.mean().item():.2e}, std {noise.std().item():.5f}, sigma {SIGMA})"))

    train = SuperResDataset(FILES, 1, 10, noise_std=SIGMA, noise_mode='random')
    torch.manual_seed(0)
    first, second = train[0][0], train[0][0]
    torch.manual_seed(0)
    again = train[0][0]
    results.append(check("random noise: a new draw on every access (every epoch)", not torch.equal(first, second)))
    results.append(check("random noise: reproducible under the same seed", torch.equal(first, again)))

    base = {'RUN_NAME': 'A_srno', 'RESIDUAL': True, 'LAMBDA': 0.0, 'SEED': 0}
    results.append(check("run name unchanged without NOISE_STD and with NOISE_STD = 0",
                         run_id_for(dict(base), 'x.yaml') == 'A_srno_res_lam0_seed0'
                         and run_id_for({**base, 'NOISE_STD': 0}, 'x.yaml') == 'A_srno_res_lam0_seed0'))
    results.append(check("run name gets _noise<sigma> only when NOISE_STD > 0",
                         run_id_for({**base, 'NOISE_STD': 0.003}, 'x.yaml') == 'A_srno_res_lam0_noise0.003_seed0'))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
