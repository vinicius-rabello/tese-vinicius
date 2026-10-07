"""A priori lambda rule (evaluation/selection.py), lambda families in
make_block_c.py / make_block_n.py, and validation-split evaluation. CPU:

    CUDA_VISIBLE_DEVICES="" python tests/test_selection.py
"""

import json
import os
import subprocess
import sys
import tempfile

import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluate_run  # noqa: E402
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from evaluation.selection import family_lambdas, select_lambda  # noqa: E402
from models.model_factory import create_model  # noqa: E402


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def sweep(values, eps_hf=None):
    """values: {lambda: (mse_val, eps_w_10_40_val)}; eps_hf: {lambda: eps_hf_val}
    (informative criterion, defaults to the same values)."""
    eps_hf = eps_hf or {lam: e for lam, (_, e) in values.items()}
    return [{'run_id': f'r{lam:g}', 'lambda': lam, 'mse_val': m, 'eps_w_val': e, 'eps_hf_val': eps_hf[lam]}
            for lam, (m, e) in values.items()]


def generate(script, *args):
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([sys.executable, f'scripts/{script}', *args], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'BLOCK_CONFIG_ROOT': d})
        files = sorted(f for _, _, fs in os.walk(d) for f in fs)
    return out.returncode, files


def main():
    results = []

    # Lowest eps_hf among lambda > 0 within the MSE bound; 1 exceeds the bound.
    _, chosen, _ = select_lambda('srcnn', sweep({0: (1.0, 0.50), 0.001: (1.01, 0.45), 0.01: (1.05, 0.40),
                                                 0.1: (1.09, 0.30), 1: (1.20, 0.10)}))
    results.append(check("rule: lowest eps_w_10_40 among lambda > 0 with MSE <= 1.10 x lambda = 0 (lambda 1 excluded)",
                         chosen == 0.1, f"(chosen {chosen})"))
    rows, chosen, _ = select_lambda('srcnn', sweep({0: (1.0, 0.50), 0.001: (1.0, 0.50), 0.01: (1.02, 0.304),
                                                    0.1: (1.05, 0.300), 1: (1.08, 0.299)}))
    results.append(check("rule: eps_w_10_40 within 2% of the lowest -> the smallest of the tied lambdas",
                         chosen == 0.01 and [r['tied_with_lowest'] for r in rows] == [False, False, True, True, True],
                         f"(chosen {chosen})"))
    _, inside, _ = select_lambda('srcnn', sweep({0: (1.0, 0.5), 0.01: (1.0, 0.3059), 0.1: (1.0, 0.300)}))
    _, outside, _ = select_lambda('srcnn', sweep({0: (1.0, 0.5), 0.01: (1.0, 0.3061), 0.1: (1.0, 0.300)}))
    results.append(check("rule: 1.97% above the lowest ties (smaller lambda wins), 2.03% does not",
                         inside == 0.01 and outside == 0.1, f"(chosen {inside} / {outside})"))
    _, chosen, why = select_lambda('srno', sweep({0: (1.0, 0.5), 0.01: (1.2, 0.1), 0.1: (1.5, 0.05)}))
    results.append(check("rule: no lambda > 0 within the MSE bound -> no choice (reported)",
                         chosen is None and 'no choice' in why))
    rows, _, _ = select_lambda('srno', sweep({0: (1.0, 0.10), 0.01: (1.05, 0.30)}))
    results.append(check("rule: lambda = 0 is the reference, never a candidate (even with the lowest eps_w_10_40)",
                         rows[0]['candidate'] is False and rows[1]['chosen'] is True))
    try:
        select_lambda('srno', sweep({0.01: (1.0, 0.3)}))
        missing = False
    except ValueError:
        missing = True
    results.append(check("rule: a sweep without its lambda = 0 run is refused", missing))

    rows, chosen, why = select_lambda('srcnn', sweep({0: (1.0, 0.20), 0.01: (1.02, 0.15), 0.1: (1.05, 0.10)},
                                                     eps_hf={0: 0.30, 0.01: 0.20, 0.1: 0.25}))
    results.append(check("amendment: the choice follows eps_w_10_40; eps_hf's choice is only reported (sensitivity)",
                         chosen == 0.1 and [r['chosen_by_eps_hf (info)'] for r in rows] == [False, True, False]
                         and 'eps_hf would choose 0.01' in why))
    _, _, why_same = select_lambda('srcnn', sweep({0: (1.0, 0.2), 0.1: (1.0, 0.1)}))
    results.append(check("amendment: agreement of the two criteria is reported", 'same lambda' in why_same))

    same, args_same = family_lambdas({'srcnn': 0.01, 'srno': 0.01})
    diff, args_diff = family_lambdas({'srcnn': 0.1, 'srno': 0.01})
    results.append(check("families: same lambda -> all architectures; different -> SRCNN's for the CNNs, SRNO's "
                         "for the operators",
                         set(same.values()) == {0.01} and args_same == '--lambda 0.01'
                         and diff == {'srcnn': 0.1, 'dscms': 0.1, 'prusr': 0.1, 'srno': 0.01, 'sofono': 0.01}
                         and args_diff == '--lambda-cnn 0.1 --lambda-op 0.01'))

    rc, files = generate('make_block_c.py', '--lambda-cnn', '0.1', '--lambda-op', '0.01', '--sweep-arch', 'srno', 'srcnn')
    results.append(check("make_block_c.py --lambda-cnn/--lambda-op: 13 configs, CNNs at 0.1, operators at 0.01",
                         rc == 0 and len(files) == 13
                         and all(('lam0.1_' in f) == (f.split('_')[1] in ('srcnn', 'dscms', 'prusr')) for f in files)))
    rc_both, _ = generate('make_block_c.py', '--lambda', '0.1', '--lambda-op', '0.01', '--sweep-arch', 'srno')
    rc_half, _ = generate('make_block_c.py', '--lambda-cnn', '0.1', '--sweep-arch', 'srno')
    results.append(check("make_block_c.py refuses --lambda with a family lambda, and a single family lambda",
                         rc_both != 0 and rc_half != 0))
    rc, files = generate('make_block_n.py', '--noise', '0.01', '--lambda-cnn', '0.1', '--lambda-op', '0.01',
                         '--sweep-arch', 'srcnn')
    results.append(check("make_block_n.py with family lambdas + SRCNN sweep: 13 configs",
                         rc == 0 and len(files) == 13, f"({len(files)})"))

    # Validation split: *_val files, test files untouched.
    val = SuperResDataset(evaluate_run.VAL_FILES, 1, 10)
    with tempfile.TemporaryDirectory() as run_dir:
        model = create_model("srcnn", residual=True)
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(run_dir, 'best.pt'))
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump({'model': 'srcnn', 'residual': True, 'psnr_ssim_R': 5.326794, 'hr_train_min': -2.457645,
                       'lpips_scale': 2.869149}, f)
        loaded, meta = evaluate_run.load_model(run_dir, 'cpu')
        constants = {k: meta[k] for k in evaluate_run.CONSTANT_KEYS}
        loader = DataLoader(Subset(val, range(8)), batch_size=4, shuffle=False)
        evaluate_run.evaluate(loaded, constants, run_dir, loader, 'cpu', with_lpips=False,
                              dates=evaluate_run.read_dates('val')[:8], split='val')
        files = sorted(os.listdir(run_dir))
        rows = open(os.path.join(run_dir, 'metrics_per_sample_val.csv')).read().splitlines()
        summary = json.load(open(os.path.join(run_dir, 'eval_summary_val.json')))
    results.append(check("evaluate(split='val'): *_val files only, validation dates, eps_hf per sample and pooled",
                         files == ['best.pt', 'config.json', 'eval_summary_val.json', 'metrics_per_sample_val.csv',
                                   'spectra_val.npz']
                         and rows[1].split(',')[1] == '2020-01-01' and 'eps_hf' in rows[0].split(',')
                         and summary['split'] == 'val' and 'eps_hf' in summary['pooled']
                         and 'eps_w_10_40' in summary['pooled']))
    results.append(check("validation dates: 731 days, 2020-01-01 .. 2021-12-31",
                         len(evaluate_run.read_dates('val')) == len(val) == 731
                         and evaluate_run.read_dates('val')[-1] == '2021-12-31'))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
